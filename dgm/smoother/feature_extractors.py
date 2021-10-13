import operator
from functools import reduce
from typing import Dict, Callable, Tuple

import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental.stax import Dense, Sigmoid, Identity, FanOut, FanInSum

from dgm.utils.helper_functions import squared_l2_norm
from dgm.utils.representatives import FeaturesToFeaturesType, TimeAndStatesToFeaturesType

""" Feature to feature maps"""


def get_features_to_features_extractor(map_type: FeaturesToFeaturesType, kwargs: Dict) -> Tuple[Callable, ...]:
    if map_type == FeaturesToFeaturesType.LINEAR:
        return _get_linear_head(kwargs)
    elif map_type == FeaturesToFeaturesType.LINEAR_WITH_SIGMOID:
        return _get_linear_head_with_sigmoid(kwargs)
    elif map_type == FeaturesToFeaturesType.FIRST_FEATURE:
        return _get_first_feature_head(kwargs)
    elif map_type == FeaturesToFeaturesType.ZERO:
        return _get_zero_head(kwargs)
    elif map_type == FeaturesToFeaturesType.NEURAL_NET:
        return _get_feature_to_feature_nn(kwargs)
    elif map_type == FeaturesToFeaturesType.IDENTITY:
        return _get_feature_to_feature_identity(kwargs)
    elif map_type == FeaturesToFeaturesType.NEURAL_NET_WITH_SERIAL_SPECIFICATION:
        return _get_feature_to_feature_nn_with_serial_specification(kwargs)
    else:
        raise NotImplementedError("This head has not been implemented.")


def _get_linear_head_with_sigmoid(kwargs):
    net_init, raw_net_apply = stax.serial(Dense(kwargs["n_out"]), Sigmoid)

    @jax.jit
    def net_apply(x, parameters):
        return raw_net_apply(parameters, x)

    @jax.jit
    def get_regularization(parameters, weights):
        return weights[kwargs['weight_key']] * squared_l2_norm(parameters)

    return net_init, net_apply, get_regularization


def _get_linear_head(kwargs):
    net_init, raw_net_apply = Dense(kwargs["n_out"])

    @jax.jit
    def net_apply(x, parameters):
        return raw_net_apply(parameters, x)

    @jax.jit
    def get_regularization(parameters, weights):
        return weights[kwargs['weight_key']] * squared_l2_norm(parameters)

    return net_init, net_apply, get_regularization


def _get_first_feature_head(_):
    def init(_, __):
        return (-1, 1), None

    @jax.jit
    def apply(features, _):
        # We assume that the first layer of features is time - the only time we apply this function is when we use
        # identity as a core layer - in this case time is the first column of the core output
        return features[:, 0]

    @jax.jit
    def regularization(_, __):
        return 0

    return init, apply, regularization


def _get_zero_head(kwargs):
    def init(_, __):
        return (-1, kwargs['n_out']), None

    @jax.jit
    def apply(features, _):
        # We assume that the first layer of features is time - the only time we apply this function is when we use
        # identity as a core layer - in this case time is the first column of the core output
        return jnp.zeros(shape=(features.shape[0], kwargs['n_out']))

    @jax.jit
    def regularization(_, __):
        return 0

    return init, apply, regularization


def _get_feature_to_feature_identity(_):
    def init(_, input_shape):
        return input_shape, []

    @jax.jit
    def apply(x, _):
        return x

    @jax.jit
    def regularization(_, __):
        return 0

    return init, apply, regularization


def _get_feature_to_feature_nn(kwargs):
    features_init, raw_features_apply = stax.serial(*kwargs["serial_input"])

    @jax.jit
    def features_apply(x, parameters):
        return raw_features_apply(parameters, x)

    @jax.jit
    def regularization(parameters, weights):
        return weights[kwargs['weight_key']] * squared_l2_norm(parameters)

    return features_init, features_apply, regularization


def _get_feature_to_feature_nn_with_serial_specification(kwargs):
    stax_input = []
    for n_neurons in kwargs['hidden_layers']:
        stax_input.append(Dense(n_neurons))
        stax_input.append(Sigmoid)
    stax_input.append(Dense(kwargs['n_out']))
    features_init, raw_features_apply = stax.serial(*stax_input)

    @jax.jit
    def features_apply(x, parameters):
        return raw_features_apply(parameters, x)

    @jax.jit
    def regularization(parameters, weights):
        return weights[kwargs['weight_key']] * squared_l2_norm(parameters)

    return features_init, features_apply, regularization


""" Concatenated (time, state) to feature maps"""


def get_time_and_states_to_features_extractor(map_type: TimeAndStatesToFeaturesType, kwargs: Dict):
    if map_type == TimeAndStatesToFeaturesType.NEURAL_NET:
        return _get_nn_core(kwargs)
    elif map_type == TimeAndStatesToFeaturesType.NN_CORE_WITH_SERIAL_SPECIFICATION:
        return _get_nn_core_with_serial_specification(kwargs)
    elif map_type == TimeAndStatesToFeaturesType.IDENTITY:
        return _get_identity_core(kwargs)
    elif map_type == TimeAndStatesToFeaturesType.JUST_TIME:
        return _get_just_time(kwargs)
    else:
        raise NotImplementedError("This core has not been implemented.")


def _get_nn_core(kwargs):
    net_init, net_apply = stax.serial(*kwargs["serial_input"])

    @jax.jit
    def core_apply(t, x0, parameters):
        features = jnp.concatenate([t.reshape(-1, 1), x0.reshape(t.size, -1)], axis=1)
        return net_apply(parameters, features).reshape([t.size, -1])

    @jax.jit
    def get_regularization(parameters, weights):
        return weights[kwargs['weight_key']] * squared_l2_norm(parameters)

    return net_init, core_apply, get_regularization


def _get_nn_core_with_serial_specification(kwargs):
    stax_input = []
    for n_neurons in kwargs['hidden_layers']:
        stax_input.append(Dense(n_neurons))
        stax_input.append(Sigmoid)

    net_init, net_apply = stax.serial(*stax_input)

    @jax.jit
    def core_apply(t, x0, parameters):
        features = jnp.concatenate([t.reshape(-1, 1), x0.reshape(t.size, -1)], axis=1)
        return net_apply(parameters, features).reshape([t.size, -1])

    @jax.jit
    def get_regularization(parameters, weights):
        return weights[kwargs['weight_key']] * squared_l2_norm(parameters)

    return net_init, core_apply, get_regularization


def _get_just_time(_):

    def init(_, __):
        return (-1, 1), None

    @jax.jit
    def core_apply(t, _, __):
        return t.reshape(-1, 1)

    @jax.jit
    def get_regularization(_, __):
        return 0

    return init, core_apply, get_regularization


def _get_identity_core(_):
    init, apply = stax.serial(Identity)

    @jax.jit
    def core_apply(t, x0, parameters):
        features = jnp.concatenate([t.reshape(-1, 1), x0.reshape(t.size, -1)], axis=1)
        return apply(parameters, features).reshape([t.size, -1])

    @jax.jit
    def get_regularization(_, __):
        return 0

    return init, core_apply, get_regularization


def _get_mean_with_x0_bias(kwargs):
    # We assume that the first dimension is time dimension, other dimensions are state dimensions
    main_net = stax.serial(Dense(10), Sigmoid, Dense(5), Sigmoid, Dense(kwargs["n_out_dim"]))
    perturbation_net = stax.serial(FanOut(2), stax.parallel(extract_time, main_net), FanInProd)
    net_init, net_apply = stax.serial(FanOut(2), stax.parallel(perturbation_net, extract_state), FanInSum)

    @jax.jit
    def mean_apply(times, initial_conditions, parameters, _):
        new_times = times.reshape(-1, 1)
        features = jnp.concatenate([new_times, initial_conditions.reshape(-1, kwargs["n_out_dim"])], axis=1)
        return jax.jit(net_apply)(parameters, features)

    def mean_init(rng, n_dim):
        in_shape = (-1, n_dim + 1)
        return net_init(rng, in_shape)

    @jax.jit
    def get_regularization(parameters, weights):
        return weights[kwargs['weight_key']] * squared_l2_norm(parameters)

    return mean_init, mean_apply


""" Helper functions """


def _extract_state():
    def init_fun(_, input_shape):
        output_shape = input_shape[:-1] + (input_shape[-1] - 1,)
        return output_shape, ()

    @jax.jit
    def apply_fun(_, inputs, **__):
        return inputs[:, 1:]

    return init_fun, apply_fun


extract_state = _extract_state()


def _extract_time():
    def init_fun(_, input_shape):
        output_shape = input_shape[:-1] + (1,)
        return output_shape, ()

    def apply_fun(_, inputs, **__):
        return inputs[:, :1]

    return init_fun, apply_fun


extract_time = _extract_time()


def _fan_in_prod():
    def init(_, input_shape):
        return input_shape[-1], ()

    def apply(_, inputs, **__):
        _prod(inputs)

    return init, apply


FanInProd = _fan_in_prod()


def _prod(iterable):
    return reduce(operator.mul, iterable, 1)
