import os

os.environ['JAX_ENABLE_X64'] = 'True'

import time
from abc import ABC, abstractmethod
from typing import Tuple, Callable, Optional, List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.experimental.ode import odeint

from dgm.simulator.prepare_matrix import create_matrix
from dgm.utils.representatives import SimulatorType

import numpy as np
from scipy.integrate import solve_ivp


class Simulator(ABC):
    def __init__(self, state_dimension: int):
        self.state_dimension = state_dimension
        pass

    ''' The following is faster jax integrator. Need to test why is it so much faster compared to standard scipy
        integrator. The results can be reproduced with integration of the old one.'''

    def simulate_trajectory(self, initial_condition: jnp.array, times: jnp.array, sigma: Optional[float],
                            key: Optional[jnp.array]) -> Tuple[jnp.array, jnp.array, jnp.array]:
        trajectory = odeint(self._compute_derivative, initial_condition, times.reshape(-1))
        ground_truth = jnp.array(trajectory.copy())
        ground_truth_derivatives = self.get_derivatives(ground_truth)
        if sigma is not None:
            key, subkey = jax.random.split(key)
            noise = sigma * jax.random.normal(key=subkey, shape=trajectory.shape, dtype=jnp.float64)
            trajectory += noise
        return jnp.array(trajectory, dtype=jnp.float64), ground_truth, ground_truth_derivatives

    # def simulate_trajectory(self, initial_condition: jnp.array, times: jnp.array, sigma: Optional[float],
    #                         key: Optional[np.ndarray]) -> Tuple[jnp.array, jnp.array, jnp.array]:
    #     # trajectory = odeint(self._compute_derivative, initial_condition, times.reshape(-1))
    #     integrand = solve_ivp(self._compute_derivative, (times[0], times[-1]), initial_condition, t_eval=times.reshape(-1), method='Radau')
    #     trajectory = integrand['y'].T
    #     ground_truth = jnp.array(trajectory.copy())
    #     ground_truth_derivatives = self.get_derivatives(ground_truth)
    #     if sigma is not None:
    #         for i in range(self.state_dimension):
    #             key, subkey = jax.random.split(key)
    #             trajectory[:, i] = trajectory[:, i] + sigma[i] * jax.random.normal(key=subkey,
    #                                                                                shape=trajectory[:, i].shape,
    #                                                                                dtype=jnp.float64)
    #     return jnp.array(trajectory, dtype=jnp.float64), ground_truth, ground_truth_derivatives

    def simulate_trajectories(self, initial_conditions, times, sigmas, rng) -> Tuple[
        List[jnp.array], List[jnp.array], List[jnp.array]]:
        trajectories = []
        ground_truth_states = []
        ground_truth_derivatives = []
        num_trajectories = len(initial_conditions)
        rng, *key = jax.random.split(rng, num_trajectories + 1)
        for i in range(num_trajectories):
            trajectory, ground_truth_state, ground_truth_derivative = self.simulate_trajectory(
                initial_condition=initial_conditions[i], times=times[i], sigma=sigmas[i], key=key[i])
            trajectories.append(trajectory)
            ground_truth_states.append(ground_truth_state)
            ground_truth_derivatives.append(ground_truth_derivative)
        return trajectories, ground_truth_states, ground_truth_derivatives

    @abstractmethod
    def get_derivatives(self, states):
        pass

    @abstractmethod
    def _compute_derivative(self, x: jnp.array, t: jnp.array) -> jnp.array:
        pass

    @abstractmethod
    def prepare_vector_field_for_plotting(self, x, y) -> Tuple[jnp.array, jnp.array, jnp.array]:
        pass


class ThreeBody(Simulator):
    """
        Lorenz 96
    """

    def __init__(self, m=jnp.array([1, 1, 1], dtype=jnp.float64), G=6.67 * 1e-2):
        """
        Initialization of LV dynamics
        Parameters
        ----------
        state_dimension state_dimension
        f               forcing constant

        """
        super().__init__(state_dimension=12)
        self.m = m
        self.G = G

    def _compute_derivative(self, x, _):
        r0 = x[0:2]
        r1 = x[2:4]
        r2 = x[4:6]
        v0 = x[6:8]
        v1 = x[8:10]
        v2 = x[10:12]

        r0_dot = v0
        r1_dot = v1
        r2_dot = v2
        v0_dot = -self.G * self.m[1] * (r0 - r1) / jnp.linalg.norm(r0 - r1) ** 3 - self.G * self.m[2] * (
                r0 - r2) / jnp.linalg.norm(r0 - r2) ** 3
        v1_dot = -self.G * self.m[2] * (r1 - r2) / jnp.linalg.norm(r1 - r2) ** 3 - self.G * self.m[0] * (
                r1 - r0) / jnp.linalg.norm(r1 - r0) ** 3
        v2_dot = -self.G * self.m[0] * (r2 - r0) / jnp.linalg.norm(r2 - r0) ** 3 - self.G * self.m[1] * (
                r2 - r1) / jnp.linalg.norm(r2 - r1) ** 3

        return jnp.concatenate([r0_dot, r1_dot, r2_dot, v0_dot, v1_dot, v2_dot])

    def get_derivatives(self, states):
        r0 = states[:, 0:2]
        r1 = states[:, 2:4]
        r2 = states[:, 4:6]
        v0 = states[:, 6:8]
        v1 = states[:, 8:10]
        v2 = states[:, 10:12]

        r0_dot = v0
        r1_dot = v1
        r2_dot = v2
        v0_dot = -self.G * self.m[1] * (r0 - r1) / jnp.linalg.norm(r0 - r1, axis=1).reshape(-1, 1) ** 3 - self.G * \
                 self.m[2] * (
                         r0 - r2) / jnp.linalg.norm(r0 - r2, axis=1).reshape(-1, 1) ** 3
        v1_dot = -self.G * self.m[2] * (r1 - r2) / jnp.linalg.norm(r1 - r2, axis=1).reshape(-1, 1) ** 3 - self.G * \
                 self.m[0] * (
                         r1 - r0) / jnp.linalg.norm(r1 - r0, axis=1).reshape(-1, 1) ** 3
        v2_dot = -self.G * self.m[0] * (r2 - r0) / jnp.linalg.norm(r2 - r0, axis=1).reshape(-1, 1) ** 3 - self.G * \
                 self.m[1] * (
                         r2 - r1) / jnp.linalg.norm(r2 - r1, axis=1).reshape(-1, 1) ** 3

        return jnp.concatenate([r0_dot, r1_dot, r2_dot, v0_dot, v1_dot, v2_dot])

    def prepare_vector_field_for_plotting(self, x, y):
        raise NotImplementedError('State dimension is not 2. We do not provide plotting of vector field in this case.')


class Lorenz96(Simulator):
    """
        Lorenz 96
    """

    def __init__(self, state_dimension, f):
        """
        Initialization of LV dynamics
        Parameters
        ----------
        state_dimension state_dimension
        f               forcing constant

        """
        super().__init__(state_dimension=state_dimension)
        self.f = f

    def _compute_derivative(self, x, _):
        """
        Compute the derivative at state observations_train and time _
        Parameters
        ----------
        observations_train    state
        _               time (non important since we have autonomous system)

        Returns
        -------

        """
        return (jnp.roll(x, -1) - jnp.roll(x, 2)) * jnp.roll(x, 1) - x + self.f

    def get_derivatives(self, states):
        return jnp.dot(states, self.system_matrix)

    def prepare_vector_field_for_plotting(self, x, y):
        raise NotImplementedError('State dimension is not 2. We do not provide plotting of vector field in this case.')


class Linear(Simulator):
    """
    Linear dynamics
    """

    def __init__(self, triplet=(2, 2, 2), key=12345, matrix=None):
        """
        Initialization of LV dynamics
        Parameters
        ----------
        triplet (a, b, c)
        a   number of stable modes
        b   number of marginally stable modes
        c   number of unstable modes
        key random number for jax randomness

        """
        self.triplet = triplet
        super().__init__(state_dimension=sum(self.triplet))
        if matrix is None:
            matrix = create_matrix(triplet, key)
        self.system_matrix = matrix

    def _compute_derivative(self, x, _):
        """
        Compute the derivative at state observations_train and time _
        Parameters
        ----------
        observations_train    state
        _               time (non important since we have autonomous system)

        Returns
        -------

        """
        return jnp.dot(x, self.system_matrix)

    def get_derivatives(self, states):
        return jnp.dot(states, self.system_matrix)

    def prepare_vector_field_for_plotting(self, x, y):
        raise NotImplementedError('State dimension is not 2. We do not provide plotting of vector field in this case.')


class LotkaVolterra(Simulator):
    """
    Lotka Volterra dynamics
    """

    def __init__(self, params=jnp.array([1, 1, 1, 1])):
        """
        Initialization of LV dynamics
        Parameters
        ----------
        a   parameter a of the LV dynamics
        b   parameter b of the LV dynamics
        c   parameter c of the LV dynamics
        d   parameter d of the LV dynamics
        """
        super().__init__(state_dimension=2)
        self.params = params

    def _compute_derivative(self, x, _):
        """
        Compute the derivative at state observations_train and time _
        Parameters
        ----------
        observations_train    state
        _               time (non important since we have autonomous system)

        Returns
        -------

        """
        return jnp.array([self.params[0] * x[0] - self.params[1] * x[0] * x[1],
                          self.params[2] * x[0] * x[1] - self.params[3] * x[1]])

    def get_derivatives(self, states):
        x0, x1 = states[:, 0].reshape(-1, 1), states[:, 1].reshape(-1, 1)
        return jnp.concatenate([self.params[0] * x0 - self.params[1] * x0 * x1,
                                self.params[2] * x0 * x1 - self.params[2] * x1], axis=1)

    def prepare_vector_field_for_plotting(self, x, y):
        u = self.params[0] * x - self.params[1] * x * y
        v = self.params[2] * x * y - self.params[3] * y
        norm = jnp.sqrt(u ** 2 + v ** 2)
        return u, v, norm


class DoublePendulum(Simulator):
    def __init__(self, m=1, l=1, g=9.81):
        """
        Initialization of Double Pendulum dynamics
        Parameters
        ----------
        m   mass of th the rods
        l   length of the rods
        g   gravity
        """
        super().__init__(state_dimension=4)
        self.m = m
        self.l = l
        self.g = g

    def _compute_derivative(self, x, _):
        x1_dot = 6 / (self.m * self.l ** 2) * (2 * x[2] - 3 * jnp.cos(x[0] - x[1]) * x[3]) / (
                16 - 9 * jnp.cos(x[0] - x[1]) ** 2)
        x2_dot = 6 / (self.m * self.l ** 2) * (8 * x[3] - 3 * jnp.cos(x[0] - x[1]) * x[2]) / (
                16 - 9 * jnp.cos(x[0] - x[1]) ** 2)
        x3_dot = (-0.5) * self.m * self.l ** 2 * (
                x1_dot * x2_dot * jnp.sin(x[0] - x[1]) + 3 * self.g / self.l * jnp.sin(x[0]))
        x4_dot = (-0.5) * self.m * self.l ** 2 * (
                -x1_dot * x2_dot * jnp.sin(x[0] - x[1]) + 3 * self.g / self.l * jnp.sin(x[1]))
        return jnp.array([x1_dot, x2_dot, x3_dot, x4_dot])

    def get_derivatives(self, states):
        x0, x1, = states[:, 0].reshape(-1, 1), states[:, 1].reshape(-1, 1),
        x2, x3 = states[:, 2].reshape(-1, 1), states[:, 3].reshape(-1, 1)
        x1_dot = 6 / (self.m * self.l ** 2) * (2 * x2 - 3 * jnp.cos(x0 - x1) * x3) / (
                16 - 9 * jnp.cos(x0 - x1) ** 2)
        x2_dot = 6 / (self.m * self.l ** 2) * (8 * x3 - 3 * jnp.cos(x0 - x1) * x2) / (
                16 - 9 * jnp.cos(x0 - x1) ** 2)
        x3_dot = (-0.5) * self.m * self.l ** 2 * (
                x1_dot * x2_dot * jnp.sin(x0 - x1) + 3 * self.g / self.l * jnp.sin(x0))
        x4_dot = (-0.5) * self.m * self.l ** 2 * (
                -x1_dot * x2_dot * jnp.sin(x0 - x1) + 3 * self.g / self.l * jnp.sin(x1))

        return jnp.concatenate([x1_dot, x2_dot, x3_dot, x4_dot], axis=1)

    def prepare_vector_field_for_plotting(self, x, y):
        raise NotImplementedError('State dimension is not 2. We do not provide plotting of vector field in this case.')


class Quadrocopter(Simulator):
    def __init__(self, theta=jnp.array([0.1, 0.62, 1.13, 0.9, 0.114, 8.25, 9.85]),
                 u=2 * jnp.array([0.248, 0.2475, 0.24775, 0.24775])):
        """
        Initialization of Double Pendulum dynamics
        Parameters
        ----------
        theta   parameters_for_dgm of the Quadrocopter
        u       control parameters_for_dgm
        """
        super().__init__(state_dimension=12)
        self.theta = theta
        self.u = u

    def _compute_derivative(self, x, _):
        ub, vb, wb = x[0], x[1], x[2]
        p, q, r = x[3], x[4], x[5]
        phi, theta_sys, psi = x[6], x[7], x[8]
        xE, yE, hE = x[9], x[10], x[11]

        m = self.theta[0]  # kg
        Ixx = self.theta[1] * 1e-3  # kg-m^2
        Iyy = self.theta[2] * 1e-3  # kg-m^2
        Izz = self.theta[3] * (Ixx + Iyy)  # kg-m^2 (Assume nearly flat object, z=0)
        dx = self.theta[4]  # m
        dy = self.theta[5] * 1e-2  # m
        g = self.theta[6]  # m/s/s

        # Directly get forces as inputs
        F1, F2, F3, F4 = self.u
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
        return jnp.array(
            [x0_dot, x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot, x7_dot, x8_dot, x9_dot, x10_dot, x11_dot])

    def get_derivatives(self, x):
        ub, vb, wb = x[:, 0].reshape(-1, 1), x[:, 1].reshape(-1, 1), x[:, 2].reshape(-1, 1)
        p, q, r = x[:, 3].reshape(-1, 1), x[:, 4].reshape(-1, 1), x[:, 5].reshape(-1, 1)
        phi, theta_sys, psi = x[:, 6].reshape(-1, 1), x[:, 7].reshape(-1, 1), x[:, 8].reshape(-1, 1)
        xE, yE, hE = x[:, 9].reshape(-1, 1), x[:, 10].reshape(-1, 1), x[:, 11].reshape(-1, 1)

        m = self.theta[0]  # kg
        Ixx = self.theta[1] * 1e-3  # kg-m^2
        Iyy = self.theta[2] * 1e-3  # kg-m^2
        Izz = self.theta[3] * (Ixx + Iyy)  # kg-m^2 (Assume nearly flat object, z=0)
        dx = self.theta[4]  # m
        dy = self.theta[5] * 1e-2  # m
        g = self.theta[6]  # m/s/s

        # Directly get forces as inputs
        F1, F2, F3, F4 = self.u
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

        return jnp.concatenate(
            [x0_dot, x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot, x7_dot, x8_dot, x9_dot, x10_dot, x11_dot], axis=1)

    def prepare_vector_field_for_plotting(self, x, y):
        raise NotImplementedError('State dimension is not 2. We do not provide plotting of vector field in this case.')


class PerfectlyAdapted(Simulator):
    """
    System from paper "Sniffers, buzzers, toggles and blinkers: dynamics of regulatory
    and signaling pathways in the cell", presented at Figure 1d
    """

    def __init__(self, k_1=2, k_2=2, k_3=1, k_4=1, s=1):
        super().__init__(state_dimension=2)
        self.k_1 = k_1
        self.k_2 = k_2
        self.k_3 = k_3
        self.k_4 = k_4
        self.s = s

    def _compute_derivative(self, x, _):
        """

        Parameters
        ----------
        x   Concatenation of parameters_for_dgm x = [R, X]
        _   time, but we don't need it since system is autonomous

        Returns
        -------
        The derivative of the system
        """
        return jnp.array([self.k_1 * self.s - self.k_2 * x[0] * x[1], self.k_3 * self.s - self.k_4 * x[1]])

    def get_derivatives(self, states):
        x0, x1 = states[:, 0].reshape(-1, 1), states[:, 1].reshape(-1, 1)
        return jnp.concatenate([self.k_1 * self.s - self.k_2 * x0 * x1, self.k_3 * self.s - self.k_4 * x1], axis=1)

    def prepare_vector_field_for_plotting(self, x, y):
        u = self.k_1 * self.s - self.k_2 * x * y
        v = self.k_3 * self.s - self.k_4 * y
        norm = jnp.sqrt(u ** 2 + v ** 2)
        return u, v, norm


class NegativeFeedbackOscilator(Simulator):
    def __init__(self, k_0=0.0, k_1=1.0, k_2=0.01, k_prime_2=10.0, k_3=0.1, k_4=0.2, k_5=0.1, k_6=0.05, y_t=1.0,
                 r_t=1.0,
                 k_m3=0.01, k_m4=0.01, k_m5=0.01, k_m6=0.01, s=2.0):
        super().__init__(state_dimension=3)
        self.k_0 = k_0
        self.k_1 = k_1
        self.k_2 = k_2
        self.k_prime_2 = k_prime_2
        self.k_3 = k_3
        self.k_4 = k_4
        self.k_5 = k_5
        self.k_6 = k_6
        self.y_t = y_t
        self.r_t = r_t
        self.k_m3 = k_m3
        self.k_m4 = k_m4
        self.k_m5 = k_m5
        self.k_m6 = k_m6
        self.s = s

    def _compute_derivative(self, x, _):
        """

        Parameters
        ----------
        x   Concatenation of parameters_for_dgm x = [X, Y_p, R_p]
        _   time, but we don't need it since system is autonomous

        Returns
        -------
        The derivative of the system
        """
        X, Y_P, R_P = x[0], x[1], x[2]
        return jnp.array([self.k_0 + self.k_1 * self.s - self.k_2 * X + self.k_prime_2 * R_P * X,
                          self.k_3 * X * (self.y_t - Y_P) / (self.k_m3 + self.y_t - Y_P) - self.k_4 * Y_P / (
                                  self.k_m4 + Y_P),
                          self.k_5 * Y_P * (self.r_t - R_P) / (self.k_m5 + self.r_t - R_P) - self.k_6 * R_P / (
                                  self.k_m6 + R_P)])

    def get_derivatives(self, states):
        x0, x1, x2 = states[:, 0].reshape(-1, 1), states[:, 1].reshape(-1, 1), states[:, 2].reshape(-1, 1)
        return jnp.array([self.k_0 + self.k_1 * self.s - self.k_2 * x0 + self.k_prime_2 * x2 * x0,
                          self.k_3 * x0 * (self.y_t - x1) / (self.k_m3 + self.y_t - x1) - self.k_4 * x1 / (
                                  self.k_m4 + x1),
                          self.k_5 * x1 * (self.r_t - x2) / (self.k_m5 + self.r_t - x2) - self.k_6 * x2 / (
                                  self.k_m6 + x2)])

    def prepare_vector_field_for_plotting(self, x, y):
        raise NotImplementedError('State dimension is not 2. We do not provide plotting of vector field in this case.')


class Lorenz(Simulator):
    def __init__(self, sigma=10, rho=28, beta=8 / 3):
        super().__init__(state_dimension=3)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def _compute_derivative(self, x, _):
        return jnp.array([self.sigma * (x[1] - x[0]), x[0] * (self.rho - x[2]) - x[1], x[0] * x[1] - self.beta * x[2]])

    def get_derivatives(self, states):
        x0, x1, x2 = states[:, 0].reshape(-1, 1), states[:, 1].reshape(-1, 1), states[:, 2].reshape(-1, 1)
        return jnp.concatenate(
            [self.sigma * (x1 - x0), x0 * (self.rho - x2) - x1, x0 * x1 - self.beta * x2], axis=1
        )

    def prepare_vector_field_for_plotting(self, x, y):
        raise NotImplementedError('State dimension is not 2. We do not provide plotting of vector field in this case.')


class CSTR(Simulator):
    def __init__(self, phi=1, delta=1, d=1, b=1):
        super().__init__(state_dimension=2)
        self.phi = phi
        self.delta = delta
        self.d = d
        self.b = b

        def u(x):
            return 0

        self.u: Callable = u

    def _compute_derivative(self, x, _):
        return jnp.array([-x[0] + self.d * (1 - x[0]) * jnp.exp(x[1] / (1 + x[1] / self.phi)),
                          -(1 + self.delta) * x[1] + self.b * self.d * (1 - x[0]) * jnp.exp(
                              x[1] / (1 + x[1] / self.phi)) + self.delta * self.u(x)])

    def get_derivatives(self, states):
        x0, x1 = states[:, 0].reshape(-1, 1), states[:, 1].reshape(-1, 1)
        return jnp.concatenate([-x0 + self.d * (1 - x0) * jnp.exp(x1 / (1 + x1 / self.phi)),
                                -(1 + self.delta) * x1 + self.b * self.d * (1 - x0) * jnp.exp(
                                    x1 / (1 + x1 / self.phi)) + self.delta * self.u(states)], axis=1)

    def prepare_vector_field_for_plotting(self, x, y):
        raise NotImplementedError('State dimension is not 2. We do not provide plotting of vector field in this case.')


def get_simulator(simulator: SimulatorType, **kwargs) -> Simulator:
    if simulator == SimulatorType.LOTKA_VOLTERRA:
        return LotkaVolterra(**kwargs)
    elif simulator == SimulatorType.PERFECTLY_ADAPTED:
        return PerfectlyAdapted()
    elif simulator == SimulatorType.NEGATIVE_FEEDBACK_OSCILATOR:
        return NegativeFeedbackOscilator()
    elif simulator == SimulatorType.LORENZ:
        return Lorenz()
    elif simulator == SimulatorType.CSTR:
        return CSTR()
    elif simulator == SimulatorType.DOUBLE_PENDULUM:
        return DoublePendulum()
    elif simulator == SimulatorType.QUADROCOPTER:
        return Quadrocopter()
    elif simulator == SimulatorType.LINEAR:
        return Linear(**kwargs)


def simulate_tree_body():
    seed = 0
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    my_simulator = ThreeBody(m=jnp.array([10, 10, 10], dtype=jnp.float64))

    num_trajectories = 1
    my_test_times = [jnp.linspace(0, 100, 1000, dtype=jnp.float64) for _ in range(num_trajectories)]
    my_initial_conditions = [jnp.concatenate(
        [jnp.array([3, -2, -3, -1, 3, 5], dtype=jnp.float64), jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.float64)])]
    my_stds_for_simulation = [None for _ in range(num_trajectories)]
    trajectories, ground_truth_state, ground_truth_derivatives = my_simulator.simulate_trajectories(
        initial_conditions=my_initial_conditions, times=my_test_times, sigmas=my_stds_for_simulation, rng=key)
    fig, axs = plt.subplots(1, 12, figsize=(16, 1 * 4))
    axs = axs.reshape(1, 12)
    for j in range(num_trajectories):
        for i in range(12):
            axs[0, i].plot(my_test_times[j], trajectories[j][:, i])
        # axs[1, i].plot(my_test_times[0], ground_truth_derivatives[0][:, i])
    plt.show()

    fig, axs = plt.subplots(1, 1, figsize=(16, 1 * 4))
    axs.plot(trajectories[0][:, 0], trajectories[0][:, 1], color='red', label=r'$r_0$')
    axs.plot(trajectories[0][:, 2], trajectories[0][:, 3], color='blue', label=r'$r_1$')
    axs.plot(trajectories[0][:, 4], trajectories[0][:, 5], color='green', label=r'$r_2$')
    plt.show()


def simulate_double_pendulum():
    seed = 0
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)

    my_simulator = DoublePendulum()

    num_x_points = 10
    num_y_points = 10
    num_trajectories = num_x_points * num_y_points
    my_test_times = [jnp.linspace(0, 1, 100, dtype=jnp.float64) for _ in range(num_trajectories)]

    points_x = jnp.linspace(-jnp.pi / 6, jnp.pi / 6, num_x_points)
    points_y = jnp.linspace(-jnp.pi / 6, jnp.pi / 6, num_x_points)
    out = jnp.stack(jnp.meshgrid(points_x, points_y))
    my_initial_conditions = [out[:, i, j] for i in range(num_x_points) for j in range(num_y_points)]
    my_initial_conditions = [jnp.concatenate([ic, jnp.array([0, 0])]) for ic in my_initial_conditions]

    my_stds_for_simulation = [None for _ in range(num_trajectories)]

    # For one dimensional case
    # my_initial_conditions = [jnp.array([jnp.pi / 6, jnp.pi / 6, 0, 0], dtype=jnp.float54)]
    # my_test_times = [jnp.linspace(0, 3, 100, dtype=jnp.float64)]

    trajectories, ground_truth_state, ground_truth_derivatives = my_simulator.simulate_trajectories(
        initial_conditions=my_initial_conditions, times=my_test_times, sigmas=my_stds_for_simulation, rng=key)
    fig, axs = plt.subplots(1, 4, figsize=(16, 1 * 4))
    axs = axs.reshape(1, 4)
    for j in range(num_trajectories):
        for i in range(4):
            axs[0, i].plot(my_test_times[j], trajectories[j][:, i])
        # axs[1, i].plot(my_test_times[0], ground_truth_derivatives[0][:, i])
    plt.show()


def simulate_quadrocopter():
    seed = 0
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)

    my_simulator = Quadrocopter()

    num_dimensions = 12
    # num_trajectories = 1
    # my_initial_conditions = [jnp.zeros(num_dimensions)]

    num_x_points = 4
    num_y_points = 4
    num_z_points = 4
    num_trajectories = num_x_points * num_y_points * num_z_points
    points_x = jnp.linspace(-jnp.pi / 18, jnp.pi / 18, num_x_points)
    points_y = jnp.linspace(-jnp.pi / 18, jnp.pi / 18, num_y_points)
    points_z = jnp.linspace(-jnp.pi / 18, jnp.pi / 18, num_z_points)
    my_initial_conditions = [jnp.stack(jnp.meshgrid(points_x, points_y, points_z))[:, i, j, k] for i in
                             range(num_x_points) for j in range(num_y_points) for k in range(num_z_points)]
    my_initial_conditions = [jnp.concatenate([jnp.zeros(6), ic, jnp.zeros(3)]) for ic in my_initial_conditions]

    my_stds_for_simulation = [None for _ in range(num_trajectories)]
    my_test_times = [jnp.linspace(0, 10, 100, dtype=jnp.float64) for _ in range(num_trajectories)]

    start_time = time.time()
    trajectories, ground_truth_state, ground_truth_derivatives = my_simulator.simulate_trajectories(
        initial_conditions=my_initial_conditions, times=my_test_times, sigmas=my_stds_for_simulation, rng=key)
    fig, axs = plt.subplots(1, num_dimensions, figsize=(4 * num_dimensions, 4))
    axs = axs.reshape(1, num_dimensions)
    for j in range(num_trajectories):
        for i in range(num_dimensions):
            axs[0, i].plot(my_test_times[j], trajectories[j][:, i])
        # axs[1, i].plot(my_test_times[0], ground_truth_derivatives[0][:, i])
    plt.tight_layout()
    plt.savefig('/Users/lenarttreven/Desktop/quadrocopter.png')
    print("Elapsed time: ", time.time() - start_time, " seconds")


if __name__ == '__main__':
    simulate_tree_body()
