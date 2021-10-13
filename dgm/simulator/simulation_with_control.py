import os

# os.environ['JAX_ENABLE_X64'] = 'True'
from jax.experimental.ode import odeint
import jax.numpy as jnp
from typing import Optional, Tuple, Callable
from jax import vmap
import matplotlib.pyplot as plt


class Pendulum:
    def __init__(self, m=1, l=1, g=9.81):
        self.m = m
        self.l = l
        self.g = g
        self.theta = 0
        self.omega = 0
        self.stable_energy = 2 * self.m * self.g * self.l
        self.moment_of_inertia = 0.5 * self.m * self.l ** 2
        self._compute_derivative = self._prepare_compute_derivative()

    def _energy(self, x):
        # x = (theta, omega)
        return 0.5 * self.moment_of_inertia * x[1] ** 2 + self.m * self.g * self.l * (1 - jnp.cos(x[0]))

    def _control(self, x):
        a = 0.1
        b = 5
        return b * jnp.tanh(a * (self._energy(x) - self.stable_energy) * x[1] * jnp.cos(x[0]))
        # return 5 * jnp.tanh((self._energy(x) - self.stable_energy) * x[1] * jnp.cos(x[0]))

    # def _control(self, x, P=0.2, D=0.05):
    #     return (P / 2 * (jnp.cos(x[0]) - jnp.cos(jnp.pi)) ** 2 + P / 2 * x[1] ** 2 + D * (jnp.cos(x[0]) - jnp.cos(jnp.pi)) * x[1] + D * x[
    #         1] * self.g / self.l * jnp.sin(x[0])) / (1 - D * x[1])

    def _prepare_compute_derivative(self) -> Callable:
        def compute_derivative(x, t):
            x0_dot = x[1]
            x1_dot = self.g / self.l * jnp.sin(x[0]) + self._control(x)
            return jnp.array([x0_dot, x1_dot])

        return compute_derivative

    def simulate_trajectory(self, initial_condition: jnp.array, times: jnp.array) -> jnp.array:
        trajectory = odeint(self._compute_derivative, initial_condition, times.reshape(-1))
        return trajectory


def sign(x):
    if x >= 0:
        return 1
    return -1


if __name__ == '__main__':
    system = Pendulum()
    initial_condition = jnp.array([0, 0.1])
    times = jnp.linspace(0, 10, 100)
    trajectory = system.simulate_trajectory(initial_condition, times)
    figure, axs = plt.subplots(1, 3, figsize=(16, 4))
    axs[0].plot(times, trajectory[:, 0], label='Angle')
    axs[0].plot(times, trajectory[:, 1], label='Angular velocity')
    axs[0].legend()

    axs[1].plot(times, 1 - jnp.cos(trajectory[:, 0]), label='Potential energy')
    axs[1].legend()

    axs[2].plot(times, system._control(trajectory.T), label='Control')
    axs[2].legend()
    plt.show()
