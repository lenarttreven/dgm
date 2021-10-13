from typing import Dict

import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten
import warnings


from dgm.smoother.feature_extractors import get_time_and_states_to_features_extractor, \
    get_features_to_features_extractor
from dgm.smoother.kernel import get_kernel_head
from dgm.utils.representatives import KernelType, FeaturesToFeaturesType, TimeAndStatesToFeaturesType


def get_deep_kernel_feature_creator(
        kernel_feature_creator_type: KernelType, kernel_feature_creator_kwargs: Dict,
        core_type: TimeAndStatesToFeaturesType, core_kwargs: Dict,
        kernel_core_type: FeaturesToFeaturesType, kernel_core_kwargs: Dict,
        kernel_head_type: FeaturesToFeaturesType, kernel_head_kwargs: Dict
):
    """
    Creates kernels independently over output dimensions, where all dimensions share the same feature extractor
    """
    if kernel_feature_creator_type == KernelType.RBF_RFF:
        pure_feature_getter = get_rbf_rff_feature_creator_2(
            kernel_feature_creator_kwargs['feature_rng'],
            kernel_feature_creator_kwargs['n_rff'],
            kernel_feature_creator_kwargs['n_features']
        )
    elif kernel_feature_creator_type == KernelType.LINEAR:
        pure_feature_getter = get_linear_feature_creator()
    else:
        raise KeyError("invalid feature creator type: {}".format(kernel_feature_creator_type))

    kernel_head = get_kernel_head(kernel_head_type, kernel_head_kwargs)

    feature_getter_init, feature_getter_apply, get_feature_getter_regularization = get_headed_feature_creators(
        pure_feature_getter,
        kernel_head
    )

    core_init, core_apply, _ = get_time_and_states_to_features_extractor(core_type, core_kwargs)
    kernel_core_init, kernel_core_apply, kernel_core_get_regularization = get_features_to_features_extractor(
        kernel_core_type, kernel_core_kwargs)

    @jax.jit
    def composite_feature_apply(t0, x0, filtered_params, core_params):
        return feature_getter_apply(kernel_core_apply(core_apply(t0, x0, core_params),
                                                      filtered_params["head_params"]).reshape(-1),
                                    filtered_params["kernel"])

    def composite_feature_init(rng, n_features, n_output_dimensions):
        input_shape = (-1, n_features)
        core_features_shape, _ = core_init(rng, input_shape)  # wasting rng OK, since only shape considered
        all_params = dict()
        rng_kernel, rng_head = jax.random.split(rng, 2)
        head_out_shape, head_parameters = kernel_core_init(rng_head, core_features_shape)
        all_params["head_params"] = head_parameters
        all_params['kernel'] = []

        # get kernel parameters structure
        rng_kernel, curr_rng = jax.random.split(rng_kernel, 2)
        kernel_parameters = feature_getter_init(curr_rng, head_out_shape)[1]

        all_kernel_params = [[leave] for leave in kernel_parameters]

        if n_output_dimensions > 1:
            for output_id in range(1, n_output_dimensions):
                rng_kernel, curr_rng = jax.random.split(rng_kernel, 2)
                _, kernel_parameters = feature_getter_init(curr_rng, head_out_shape)
                all_params["kernel"].append(kernel_parameters)
                for index, leave in enumerate(kernel_parameters):
                    all_kernel_params[index].append(leave)
        all_params['kernel'] = [jnp.stack(leaves) for leaves in all_kernel_params]
        return [], all_params

    @jax.jit
    def get_composite_feature_regularization(parameters, weights):
        num_dim = len(parameters.keys()) - 1
        objective = kernel_core_get_regularization(parameters["head_params"], weights)
        for dim_id in range(num_dim):
            objective += get_feature_getter_regularization([leave[dim_id] for leave in parameters["kernel"]], weights)
        return objective

    def filter_composite_feature_params(all_params, dim_id):
        return {"kernel": [leave[dim_id] for leave in all_params["kernel"]],
                "head_params": all_params["head_params"]}

    return (
        composite_feature_init,
        composite_feature_apply,
        filter_composite_feature_params,
        get_composite_feature_regularization
    )


def get_rbf_rff_feature_creator(rng,
                                n_rff=20,
                                n_features=1):
    rng, omega_key, b_key = jax.random.split(rng, 3)
    # sample omegas
    omegas = jax.random.multivariate_normal(key=omega_key,
                                            mean=jnp.zeros(n_features),
                                            cov=jnp.eye(n_features),
                                            shape=[n_rff]
                                            )
    # sample bs
    biases = jax.random.uniform(b_key, [n_rff])

    @jax.jit
    def creator_apply(x, _):
        features = jnp.cos(omegas @ x.reshape(-1, 1) + biases.reshape(-1, 1))
        return (jnp.sqrt(2) * features).reshape(-1)

    def initialize_parameters(_, __):
        return [], []

    @jax.jit
    def get_creator_regularization(_, __):
        return 0

    return initialize_parameters, creator_apply, get_creator_regularization


def get_rbf_rff_feature_creator_2(rng,
                                  n_rff=20,
                                  n_features=1):
    if n_rff % 2 != 0:
        warnings.warn("Number n_rff is not even - number of features will decrease by 1")
    rng, omega_key = jax.random.split(rng)
    # sample omegas
    omegas = jax.random.multivariate_normal(key=omega_key,
                                            mean=jnp.zeros(n_features),
                                            cov=jnp.eye(n_features),
                                            shape=[n_rff // 2]
                                            )

    @jax.jit
    def creator_apply(x, _):
        cos_features = jnp.cos(omegas @ x.reshape(-1, 1))
        sin_features = jnp.sin(omegas @ x.reshape(-1, 1))
        features = jnp.concatenate([cos_features, sin_features])
        return (jnp.sqrt(2) * features).reshape(-1)

    def initialize_parameters(_, __):
        return [], []

    @jax.jit
    def get_creator_regularization(_, __):
        return 0

    return initialize_parameters, creator_apply, get_creator_regularization


def get_linear_feature_creator():
    @jax.jit
    def creator_apply(x, _):
        return x

    def initialize_parameters(_, __):
        return [], []

    @jax.jit
    def get_creator_regularization(_, __):
        return 0

    return initialize_parameters, creator_apply, get_creator_regularization


def get_headed_feature_creators(pure_feature_getter, kernel_head):
    pure_feature_init, pure_feature_apply, pure_feature_regularization = pure_feature_getter
    kernel_head_init, kernel_head_apply, kernel_head_regularization = kernel_head

    def _init_with_structure(rng, input_shape):
        rng, *keys = jax.random.split(rng, 3)
        parameters = dict()
        feature_creator_output, parameters["individual_head"] = kernel_head_init(keys[0], input_shape)
        _, parameters["pure_feature_creator"] = pure_feature_init(keys[1], feature_creator_output)
        leaves, structure = tree_flatten(parameters)
        return [], leaves, structure

    def init(rng, input_shape):
        return _init_with_structure(rng, input_shape)[:2]

    global_structure = _init_with_structure(jax.random.PRNGKey(0), (-1, 1))[2]

    @jax.jit
    def apply(x, parameters):
        parameters = tree_unflatten(global_structure, parameters)
        input_features = kernel_head_apply(x, parameters["individual_head"])
        return pure_feature_apply(input_features, parameters["pure_feature_creator"])

    @jax.jit
    def regularization(parameters, weights):
        parameters = tree_unflatten(global_structure, parameters)
        objective = kernel_head_regularization(parameters["individual_head"], weights)
        objective += pure_feature_regularization(parameters["pure_feature_creator"], weights)
        return objective

    return init, apply, regularization
