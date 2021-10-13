from typing import Dict

import jax
import jax.numpy as jnp

from dgm.smoother.feature_extractors import get_time_and_states_to_features_extractor
from dgm.smoother.feature_extractors import get_features_to_features_extractor
from dgm.smoother.kernel import get_kernel
from dgm.utils.representatives import FeaturesToFeaturesType, TimeAndStatesToFeaturesType
from dgm.utils.representatives import KernelType


def get_deep_kernel(
        kernel: KernelType, kernel_kwargs: Dict,
        core_type: TimeAndStatesToFeaturesType, core_kwargs: Dict,
        kernel_core_type: FeaturesToFeaturesType, kernel_core_kwargs: Dict,
        kernel_head_type: FeaturesToFeaturesType, kernel_head_kwargs: Dict
):
    """
    creates kernels independently over output dimensions, where all dimensions share the same feature extractor
    """
    kernel_init, kernel_apply, get_kernel_regularization = get_kernel(kernel, kernel_kwargs,
                                                                      kernel_head_type, kernel_head_kwargs)
    core_init, core_apply, _ = get_time_and_states_to_features_extractor(core_type, core_kwargs)
    kernel_core_init, kernel_core_apply, kernel_core_regularization = get_features_to_features_extractor(kernel_core_type,
                                                                                                         kernel_core_kwargs)

    @jax.jit
    def deep_kernel_apply(t0, x0, t1, x1, filtered_params, core_params):
        features0 = core_apply(t0, x0, core_params)
        features1 = core_apply(t1, x1, core_params)
        return kernel_apply(
            kernel_core_apply(features0, filtered_params["kernel_core_params"]),
            kernel_core_apply(features1, filtered_params["kernel_core_params"]),
            filtered_params["kernel_head"]
        ).reshape()

    def deep_kernel_init(rng, n_features, n_output_dimensions):
        input_shape = (-1, n_features)
        # Only need the n_core_out
        core_features_shape, _ = core_init(rng, input_shape)  # wasting rng OK, since only shape considered
        all_params = dict()

        # Create feature_head_params
        rng_kernel, rng_head = jax.random.split(rng, 2)
        head_core_shape, kernel_core_parameters = kernel_core_init(rng_head, core_features_shape)
        all_params["kernel_core_params"] = kernel_core_parameters

        rng_kernel, curr_rng = jax.random.split(rng_kernel, 2)
        kernel_parameters = kernel_init(curr_rng, head_core_shape)[1]

        all_kernel_params = [[leave] for leave in kernel_parameters]

        if n_output_dimensions >= 2:
            for output_id in range(1, n_output_dimensions):
                rng_kernel, curr_rng = jax.random.split(rng_kernel, 2)
                kernel_parameters = kernel_init(curr_rng, head_core_shape)[1]
                for index, leave in enumerate(kernel_parameters):
                    all_kernel_params[index].append(leave)
        all_params['kernel_head'] = [jnp.stack(leaves) for leaves in all_kernel_params]
        return [], all_params

    def deep_kernel_regularization(parameters, weights):
        num_dim = len(parameters.keys()) - 1
        objective = kernel_core_regularization(parameters["kernel_core_params"], weights)
        for dim_id in range(num_dim):
            objective += get_kernel_regularization([leave[dim_id] for leave in parameters["kernel_head"]], weights)
        return objective

    def filter_deep_kernel_params(all_params, dim_id):
        return {
            "kernel_head": [leave[dim_id] for leave in all_params["kernel_head"]],
            "kernel_core_params": all_params["kernel_core_params"]
        }

    return deep_kernel_init, deep_kernel_apply, filter_deep_kernel_params, deep_kernel_regularization
