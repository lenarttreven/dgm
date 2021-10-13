from typing import Dict

import jax.numpy as jnp
import jax.random
from jax import jit
from jax.tree_util import tree_flatten, tree_unflatten

from dgm.smoother.feature_extractors import get_features_to_features_extractor
from dgm.utils.helper_functions import make_positive, negative_log_likelihood_normal
from dgm.utils.representatives import KernelType, FeaturesToFeaturesType


def get_kernel(kernel_type: KernelType, kernel_kwargs: Dict,
               kernel_head_type: FeaturesToFeaturesType, kernel_head_kwargs: Dict):
    """We select pure kernel"""
    pure_kernel = get_pure_kernel(kernel_type, kernel_kwargs)
    """We select kernel head map"""
    kernel_head = get_kernel_head(kernel_head_type, kernel_head_kwargs)

    return headed_kernel(kernel_head, pure_kernel)


def get_pure_kernel(kernel_type: KernelType, kernel_kwargs: Dict):
    if kernel_type == KernelType.RBF:
        kernel = rbf_kernel(kernel_kwargs)
    elif kernel_type == KernelType.RBF_NO_LENGTHSCALES:
        kernel = rbf_kernel_no_lengthscales(kernel_kwargs)
    elif kernel_type == KernelType.RBF_NO_VARIANCE:
        kernel = rbf_kernel_no_variance(kernel_kwargs)
    elif kernel_type == KernelType.RBF_NO_LENGTHSCALES_NO_VARIANCE:
        kernel = rbf_kernel_no_variance_no_lengthscales(kernel_kwargs)
    else:
        raise NotImplementedError("Chosen kernel has not been implemented yet.")
    return kernel


def get_kernel_head(kernel_head_type: FeaturesToFeaturesType, kernel_head_kwargs: Dict):
    return get_features_to_features_extractor(kernel_head_type, kernel_head_kwargs)


# TODO: set hyperparameters reasonably or make them tunable via weights dict
def _regularize_lengthscale(lengthscale):
    return negative_log_likelihood_normal(lengthscale, mu=-2.25, sigma=4)


# TODO: set hyperparameters reasonably or make them tunable via weights dict
def _regularize_kernel_std(std):
    return negative_log_likelihood_normal(std, mu=-2.25, sigma=4)


"""Pure kernels"""


def rbf_kernel(_):
    def initialize_parameters(_, input_shape):
        # we need 1 + input_shape parameters_for_dgm - inverse bandwidths and kernel std
        parameters = dict()
        parameters["lengthscales"] = jnp.ones(input_shape[1])
        parameters["kernel std"] = jnp.ones(1)
        return [], parameters

    @jit
    def kernel_apply(x, y, kernel_parameters):
        pseudo_lengthscales = kernel_parameters["lengthscales"]
        lengthscales = make_positive(pseudo_lengthscales)
        pseudo_kernel_std = kernel_parameters["kernel std"]
        kernel_std = make_positive(pseudo_kernel_std)
        dif = jnp.squeeze(x.reshape(-1, 1) - y.reshape(-1, 1))
        exponent = jnp.dot(dif ** 2, 1 / lengthscales ** 2)
        return kernel_std * jnp.exp(-exponent / 2).reshape()

    @jit
    def get_kernel_regularization(parameters, weights):
        objective = weights['kernel_variance'] * _regularize_kernel_std(parameters["kernel std"])
        for lengthscale_id in range(parameters["lengthscales"].size):
            objective += weights["kernel_lengthscale"] \
                         * _regularize_lengthscale(parameters["lengthscales"][lengthscale_id])
        return objective

    return initialize_parameters, kernel_apply, get_kernel_regularization


def rbf_kernel_no_variance(_):
    @jit
    def kernel_apply(x, y, kernel_parameters):
        pseudo_lengthscales = kernel_parameters["lengthscales"]
        lengthscales = make_positive(pseudo_lengthscales)
        dif = jnp.squeeze(x.reshape(-1, 1) - y.reshape(-1, 1))
        exponent = jnp.dot(dif ** 2, 1 / lengthscales ** 2)
        return jnp.exp(-exponent / 2).reshape()

    def initialize_parameters(_, input_shape):
        # we need 1 + input_shape parameters_for_dgm - inverse bandwidths and kernel std
        parameters = dict()
        parameters["lengthscales"] = jnp.ones(input_shape[1])
        return [], parameters

    @jit
    def get_kernel_regularization(parameters, weights):
        objective = 0
        for lengthscale_id in range(parameters["lengthscales"].size):
            objective += weights["kernel_lengthscale"] * _regularize_lengthscale(
                parameters["lengthscales"][lengthscale_id])
        return objective

    return initialize_parameters, kernel_apply, get_kernel_regularization


def rbf_kernel_no_lengthscales(_):
    @jit
    def kernel_apply(x, y, kernel_parameters):
        pseudo_kernel_std = kernel_parameters["kernel std"]
        kernel_std = make_positive(pseudo_kernel_std)
        dif = jnp.squeeze(x.reshape(-1, 1) - y.reshape(-1, 1))
        exponent = jnp.sum(dif ** 2)
        return kernel_std * jnp.exp(-exponent / 2).reshape()

    def initialize_parameters(_, __):
        # we need 1 + input_shape parameters_for_dgm - inverse bandwidths and kernel std
        parameters = dict()
        parameters["kernel std"] = jnp.ones(1)
        return [], parameters

    @jit
    def get_kernel_regularization(parameters, weights):
        objective = weights['kernel_variance'] * _regularize_kernel_std(parameters["kernel std"])
        return objective

    return initialize_parameters, kernel_apply, get_kernel_regularization


def rbf_kernel_no_variance_no_lengthscales(_):
    @jit
    def kernel_apply(x, y, _):
        dif = jnp.squeeze(x.reshape(-1, 1) - y.reshape(-1, 1))
        exponent = jnp.sum(dif ** 2)
        return jnp.exp(-exponent / 2).reshape()

    def initialize_parameters(_, __):
        return [], dict()

    @jit
    def get_kernel_regularization(_, __):
        return 0

    return initialize_parameters, kernel_apply, get_kernel_regularization


"""Join kernel head with pure kernel"""


def headed_kernel(kernel_head, pure_kernel):
    kernel_head_init, kernel_head_apply, kernel_head_regularization = kernel_head
    pure_kernel_init, pure_kernel_apply, pure_kernel_regularization = pure_kernel

    def _init_with_structure(rng, input_shape):
        rng, *keys = jax.random.split(rng, 3)
        parameters = dict()
        kernel_feature_output, parameters["kernel_head"] = kernel_head_init(keys[0], input_shape)
        _, parameters["pure_kernel"] = pure_kernel_init(keys[1], kernel_feature_output)
        leaves, _structure = tree_flatten(parameters)
        return [], leaves, _structure

    def init(rng, input_shape):
        return _init_with_structure(rng, input_shape)[:2]

    structure = _init_with_structure(jax.random.PRNGKey(0), (-1, 1))[2]

    @jax.jit
    def apply(x, y, parameters):
        parameters = tree_unflatten(structure, parameters)
        features_x = kernel_head_apply(x, parameters["kernel_head"])
        features_y = kernel_head_apply(y, parameters["kernel_head"])
        return pure_kernel_apply(features_x, features_y, parameters["pure_kernel"])

    @jax.jit
    def regularization(parameters, weights):
        parameters = tree_unflatten(structure, parameters)
        objective = kernel_head_regularization(parameters["kernel_head"], weights)
        objective += pure_kernel_regularization(parameters["pure_kernel"], weights)
        return objective

    return init, apply, regularization
