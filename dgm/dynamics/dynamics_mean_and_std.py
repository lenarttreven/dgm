from typing import Tuple, Any, Callable

import jax.numpy as jnp
import jax.random
import numpy as np
from jax import jit
from jax.experimental import stax
from jax.experimental.stax import Dense, Sigmoid

from dgm.utils.helper_functions import squared_l2_norm

pytree = Any
input_shape = Tuple[int, int]
output_shape = Tuple[int, int]

"""Example structure of the functions in the file"""


def example_dynamics() -> Tuple[
    Callable[[pytree, jnp.array], jnp.array], Callable[[np.ndarray, input_shape], Tuple[output_shape, pytree]],
    Callable[[pytree, pytree], float]]:
    def dynamics_apply(parameters: pytree, x: jnp.array) -> jnp.array:
        """

        Parameters
        ----------
        parameters_for_dgm          joint parameters_for_dgm of mean and std
        x                   jnp.array(num_samples, num_dimension)

        Returns
        -------
        predicted_dynamics  mean or std or both. Both are of shape jnp.array(num_samples, num_dimension)
        """
        pass

    def dynamics_init(rng: np.ndarray, in_shape: input_shape) -> Tuple[output_shape, pytree]:
        """

        Parameters
        ----------
        rng         jax random number generator

        Returns
        -------
        output shape    tuple of output shape
        parameters_for_dgm      parameters_for_dgm presented as pytree
        """
        pass

    def get_regularization(parameters: pytree, weights: pytree) -> float:
        """
        Parameters
        ----------
        parameters: joint parameters_for_dgm of mean and std
        weights: dict containing the weights
        """
        pass

    return dynamics_init, dynamics_apply, get_regularization


"""Mean functions"""


def parametric_quadrocopter_mean(u=2 * jnp.array([0.248, 0.2475, 0.24775, 0.24775], dtype=jnp.float32)):
    def mean_apply(parameters, x):
        ub, vb, wb = x[:, 0], x[:, 1], x[:, 2]
        p, q, r = x[:, 3], x[:, 4], x[:, 5]
        phi, theta_sys, psi = x[:, 6], x[:, 7], x[:, 8]
        xE, yE, hE = x[:, 9], x[:, 10], x[:, 11]

        m = parameters['m']  # kg
        Ixx = parameters['Ixx'] * 1e-3  # kg-m^2
        Iyy = parameters['Iyy'] * 1e-3  # kg-m^2
        Izz = parameters['Izz'] * (Ixx + Iyy)  # kg-m^2 (Assume nearly flat object, z=0)
        dx = parameters['dx']  # m
        dy = parameters['dy'] * 1e-2  # m
        g = parameters['g']  # m/s/s

        # Directly get forces as inputs
        F1, F2, F3, F4 = u
        Fz = F1 + F2 + F3 + F4
        L = (F2 + F3) * dy - (F1 + F4) * dy
        M = (F1 + F3) * dx - (F2 + F4) * dx
        N = 0  # -T(F1,dx,dy) - T(F2,dx,dy) + T(F3,dx,dy) + T(F4,dx,dy)

        # Pre-calculate trig values
        cphi = jnp.cos(phi)
        sphi = jnp.sin(phi)
        cthe = jnp.cos(theta_sys)
        sthe = jnp.sin(theta_sys)
        cpsi = jnp.cos(psi)
        spsi = jnp.sin(psi)

        # Calculate the derivative of the state matrix using EOM
        x0_dot = -g * sthe + r * vb - q * wb  # = udot
        x1_dot = g * sphi * cthe - r * ub + p * wb  # = vdot
        x2_dot = 1 / m * (-Fz) + g * cphi * cthe + q * ub - p * vb  # = wdot
        x3_dot = 1 / Ixx * (L + (Iyy - Izz) * q * r)  # = pdot
        x4_dot = 1 / Iyy * (M + (Izz - Ixx) * p * r)  # = qdot
        x5_dot = 1 / Izz * (N + (Ixx - Iyy) * p * q)  # = rdot
        x6_dot = p + (q * sphi + r * cphi) * sthe / cthe  # = phidot
        x7_dot = q * cphi - r * sphi  # = thetadot
        x8_dot = (q * sphi + r * cphi) / cthe  # = psidot
        x9_dot = cthe * cpsi * ub + (-cphi * spsi + sphi * sthe * cpsi) * vb + \
                 (sphi * spsi + cphi * sthe * cpsi) * wb  # = xEdot
        x10_dot = cthe * spsi * ub + (cphi * cpsi + sphi * sthe * spsi) * vb + \
                  (-sphi * cpsi + cphi * sthe * spsi) * wb  # = yEdot
        x11_dot = -1 * (-sthe * ub + sphi * cthe * vb + cphi * cthe * wb)  # = hEdot

        x_dot = jnp.stack(
            [x0_dot, x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot, x7_dot, x8_dot, x9_dot, x10_dot, x11_dot], axis=1)
        return x_dot

    def mean_init(rng, input_shape):
        parameters = dict()
        parameters['m'] = jnp.array(0.1, dtype=jnp.float32)
        parameters['Ixx'] = jnp.array(0.62, dtype=jnp.float32)
        parameters['Iyy'] = jnp.array(1.13, dtype=jnp.float32)
        parameters['Izz'] = jnp.array(0.9, dtype=jnp.float32)
        parameters['dx'] = jnp.array(0.114, dtype=jnp.float32)
        parameters['dy'] = jnp.array(8.25, dtype=jnp.float32)
        parameters['g'] = jnp.array(9.85, dtype=jnp.float32)
        return (-1, 12), parameters

    @jit
    def get_regularization(parameters, weights):
        return 0

    return mean_init, mean_apply, get_regularization


def parametric_double_pendulum_mean(g=9.81):
    def mean_apply(parameters, x):
        x0, x1, = x[:, 0].reshape(-1, 1), x[:, 1].reshape(-1, 1),
        x2, x3 = x[:, 2].reshape(-1, 1), x[:, 3].reshape(-1, 1)
        x1_dot = 6 / (parameters['m'] * parameters['l'] ** 2) * (2 * x2 - 3 * jnp.cos(x0 - x1) * x3) / (
                16 - 9 * jnp.cos(x0 - x1) ** 2)
        x2_dot = 6 / (parameters['m'] * parameters['l'] ** 2) * (8 * x3 - 3 * jnp.cos(x0 - x1) * x2) / (
                16 - 9 * jnp.cos(x0 - x1) ** 2)
        x3_dot = (-0.5) * parameters['m'] * parameters['l'] ** 2 * (
                x1_dot * x2_dot * jnp.sin(x0 - x1) + 3 * g / parameters['l'] * jnp.sin(x0))
        x4_dot = (-0.5) * parameters['m'] * parameters['l'] ** 2 * (
                -x1_dot * x2_dot * jnp.sin(x0 - x1) + 3 * g / parameters['l'] * jnp.sin(x1))

        x_dot = jnp.concatenate(
            [x1_dot.reshape(-1, 1), x2_dot.reshape(-1, 1), x3_dot.reshape(-1, 1), x4_dot.reshape(-1, 1)],
            axis=1)
        return x_dot

    def mean_init(rng, input_shape):
        parameters = dict()
        parameters['m'] = jnp.array(1, dtype=jnp.float32)
        parameters['l'] = jnp.array(1, dtype=jnp.float32)
        return (-1, 4), parameters

    @jit
    def get_regularization(parameters, weights):
        return 0

    return mean_init, mean_apply, get_regularization


def parametric_lotka_volterra_mean():
    def mean_apply(parameters, x):
        x_dot = jnp.concatenate(
            [(parameters['a'] * x[:, 0] - parameters['b'] * x[:, 0] * x[:, 1]).reshape(-1, 1),
             (parameters['c'] * x[:, 0] * x[:, 1] - parameters['d'] * x[:, 1]).reshape(-1, 1)],
            axis=1)
        return x_dot

    def mean_init(rng, input_shape):
        parameters = dict()
        for i in "abcd":
            parameters[i] = jnp.array(1, dtype=jnp.float32)
        return (-1, 2), parameters

    @jit
    def get_regularization(parameters, weights):
        return 0

    return mean_init, mean_apply, get_regularization


def parametric_perfectly_adapted_mean(s=1):
    def mean_apply(parameters, x):
        x_dot = jnp.concatenate(
            [(parameters['k_1'] * s - parameters['k_2'] * x[:, 0] * x[:, 1]).reshape(-1, 1),
             (parameters['k_3'] * s - parameters['k_4'] * x[:, 1]).reshape(-1, 1)], axis=1)
        return x_dot

    def mean_init(rng, input_shape):
        parameters = dict()
        for i in range(1, 5):
            key = "k_" + str(i)
            if i < 3:
                parameters[key] = jnp.array(2, dtype=jnp.float32)
            else:
                parameters[key] = jnp.array(1, dtype=jnp.float32)
        return (-1, 2), parameters

    @jit
    def get_regularization(parameters, weights):
        return 0

    return mean_init, mean_apply, get_regularization


def parametric_negative_feedback_oscilator_mean(s=1):
    def mean_apply(parameters, x):
        x0, x1, x2 = x[:, 0].reshape(-1, 1), x[:, 1].reshape(-1, 1), x[:, 2].reshape(-1, 1)
        x_dot = jnp.concatenate(
            [parameters['k_0'] + parameters['k_1'] * s - parameters['k_2'] * x0 +
             parameters['k_prime_2'] * x2 * x0,
             parameters['k_3'] * x0 * (parameters['y_t'] - x1) / (
                     parameters['k_m3'] + parameters['y_t'] - x1) -
             parameters['k_4'] * x1 / (parameters['k_m4'] + x1),
             parameters['k_5'] * x1 * (parameters['r_t'] - x2) / (
                     parameters['k_m5'] + parameters['r_t'] - x2) -
             parameters['k_6'] * x2 / (parameters['k_m6'] + x2)], axis=1)
        return x_dot

    def mean_init(rng, input_shape):
        parameters = dict()
        parameters['k_0'] = jnp.array(0, dtype=jnp.float32)
        parameters['k_1'] = jnp.array(1, dtype=jnp.float32)
        parameters['k_2'] = jnp.array(0.01, dtype=jnp.float32)
        parameters['k_prime_2'] = jnp.array(10, dtype=jnp.float32)
        parameters['k_3'] = jnp.array(0.1, dtype=jnp.float32)
        parameters['k_4'] = jnp.array(0.2, dtype=jnp.float32)
        parameters['k_5'] = jnp.array(0.1, dtype=jnp.float32)
        parameters['k_6'] = jnp.array(0.05, dtype=jnp.float32)
        parameters['y_t'] = jnp.array(1, dtype=jnp.float32)
        parameters['r_t'] = jnp.array(1, dtype=jnp.float32)
        parameters['k_m3'] = jnp.array(0.01, dtype=jnp.float32)
        parameters['k_m4'] = jnp.array(0.01, dtype=jnp.float32)
        parameters['k_m5'] = jnp.array(0.01, dtype=jnp.float32)
        parameters['k_m6'] = jnp.array(0.01, dtype=jnp.float32)
        return (-1, 3), parameters

    @jit
    def get_regularization(parameters, weights):
        return 0

    return mean_init, mean_apply, get_regularization


def parametric_lorenz_mean():
    def mean_apply(parameters, x):
        x0, x1, x2 = x[:, 0].reshape(-1, 1), x[:, 1].reshape(-1, 1), x[:, 2].reshape(-1, 1)
        x_dot = jnp.concatenate(
            [parameters['sigma'] * (x1 - x0), x0 * (parameters['rho'] - x2) - x1,
             x0 * x1 - parameters['beta'] * x2], axis=1)
        return x_dot

    def mean_init(rng, input_shape):
        parameters = dict()
        parameters['sigma'] = jnp.array(10, dtype=jnp.float32)
        parameters['rho'] = jnp.array(28, dtype=jnp.float32)
        parameters['beta'] = jnp.array(8 / 3, dtype=jnp.float32)
        return (-1, 3), parameters

    @jit
    def get_regularization(parameters, weights):
        return 0

    return mean_init, mean_apply, get_regularization


def parametric_cstr_mean():
    def mean_apply(parameters, x):
        x0, x1 = x[:, 0].reshape(-1, 1), x[:, 1].reshape(-1, 1)
        x_dot = jnp.concatenate(
            [-x0 + parameters['d'] * (1 - x0) * jnp.exp(x1 / (1 + x1 / parameters['phi'])),
             -(1 + parameters['delta']) * x1 + parameters['b'] * parameters['d'] * (
                     1 - x0) * jnp.exp(x1 / (1 + x1 / parameters['phi']))], axis=1)
        return x_dot

    def mean_init(rng, input_shape):
        parameters = dict()
        parameters['d'] = jnp.array(1, dtype=jnp.float32)
        parameters['b'] = jnp.array(1, dtype=jnp.float32)
        parameters['phi'] = jnp.array(1, dtype=jnp.float32)
        parameters['delta'] = jnp.array(1, dtype=jnp.float32)
        return (-1, 2), parameters

    @jit
    def get_regularization(parameters, weights):
        return 0

    return mean_init, mean_apply, get_regularization


def small_node_mean(out_dimension: int):
    mean_init, mean_apply = stax.serial(Dense(10), Sigmoid, Dense(10), Sigmoid, Dense(out_dimension))

    @jit
    def get_regularization(parameters: pytree, weights: pytree) -> float:
        return weights['dynamics'] * squared_l2_norm(parameters)

    return mean_init, mean_apply, get_regularization


def linear_prod_mean(sys_dim: int):
    def mean_apply(parameters, x):
        matrices = parameters['system_matrices']
        derivative = x
        for matrix in matrices:
            derivative = jnp.dot(derivative, matrix)
        return derivative

    def mean_init(rng, input_shape):
        parameters = dict()
        shapes = [(sys_dim, 6), (6, 9), (9, 6), (6, sys_dim)]
        matrices = []
        for shape in shapes:
            rng, subkey = jax.random.split(rng)
            matrices.append(0.1 * jax.random.normal(key=subkey, shape=shape))
        parameters['system_matrices'] = matrices
        return (-1, sys_dim), parameters

    @jit
    def get_regularization(parameters, weights):
        return 0

    return mean_init, mean_apply, get_regularization


def linear_mean(sys_dim: int):
    def mean_apply(parameters, x):
        return jnp.dot(x, parameters['system_matrix'])

    def mean_init(rng, input_shape):
        parameters = dict()
        parameters['system_matrix'] = 0.1 * jax.random.normal(key=rng, shape=(sys_dim, sys_dim))
        return (-1, sys_dim), parameters

    @jit
    def get_regularization(parameters, weights):
        return 0

    return mean_init, mean_apply, get_regularization


"""Std functions"""


def parametric_lotka_volterra_std():
    def std_apply(parameters, x):
        num_observations = x.shape[0]
        stds = jnp.concatenate(
            [jnp.array([parameters['0']] * num_observations, dtype=jnp.float32).reshape(-1, 1),
             jnp.array([parameters["1"]] * num_observations, dtype=jnp.float32).reshape(-1, 1)], axis=1)
        return stds

    def std_init(rng, input_shape):
        parameters = dict()
        parameters["0"] = jnp.array(1, jnp.float32)
        parameters["1"] = jnp.array(1, jnp.float32)
        return (-1, 2), parameters

    @jit
    def get_regularization(parameters, weights):
        return 0  # TODO: maybe add some meaningful prior

    return std_init, std_apply, get_regularization


def heteroscedastic_std(out_dimension: int):
    std_init, std_apply = stax.serial(Dense(10), Sigmoid, Dense(10), Sigmoid, Dense(out_dimension))

    @jit
    def get_regularization(parameters, weights):
        return weights['dynamics'] * squared_l2_norm(parameters)

    return std_init, std_apply, get_regularization


"""Joint functions"""


def small_dynamics(out_dimension):
    dynamics_init, dynamics_apply = stax.serial(Dense(20), Sigmoid, Dense(20), Sigmoid, Dense(2 * out_dimension))

    @jit
    def get_regularization(parameters, weights):
        return weights['dynamics'] * squared_l2_norm(parameters)

    return dynamics_init, dynamics_apply, get_regularization


def medium_dynamics(out_dimension):
    dynamics_init, dynamics_apply = stax.serial(Dense(40), Sigmoid, Dense(40), Sigmoid, Dense(40), Sigmoid,
                                                Dense(2 * out_dimension))

    @jit
    def get_regularization(parameters, weights):
        return weights['dynamics'] * squared_l2_norm(parameters)

    return dynamics_init, dynamics_apply, get_regularization


def big_dynamics(out_dimension):
    dynamics_init, dynamics_apply = stax.serial(Dense(80), Sigmoid, Dense(80), Sigmoid, Dense(80), Sigmoid, Dense(80),
                                                Sigmoid, Dense(2 * out_dimension))

    @jit
    def get_regularization(parameters, weights):
        return weights['dynamics'] * squared_l2_norm(parameters)

    return dynamics_init, dynamics_apply, get_regularization


def joint_nn_dynamics(out_dimension, hidden_layers):
    stax_input = []
    for n_neurons in hidden_layers:
        stax_input.append(Dense(n_neurons))
        stax_input.append(Sigmoid)
    stax_input.append(Dense(2 * out_dimension))
    dynamics_init, dynamics_apply = stax.serial(*stax_input)

    @jit
    def get_regularization(parameters, weights):
        return weights['dynamics'] * squared_l2_norm(parameters)

    return dynamics_init, dynamics_apply, get_regularization
