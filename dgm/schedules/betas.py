from enum import Enum, auto
from typing import Callable, Any

import jax.numpy as jnp
from jax.experimental.optimizers import constant, piecewise_constant

BetaSchedule = Callable[[int], jnp.array]


def polynomial_decay(step_size, decay_steps, final_step_size, power):
    def schedule(step_num):
        step_num = jnp.minimum(step_num, decay_steps)
        step_mult = jnp.maximum(1 - step_num / decay_steps, 0) ** power
        return step_mult * (step_size - final_step_size) + final_step_size

    return schedule


class BetasType(Enum):
    PIECEWISE_CONSTANT = auto()
    CONSTANT = auto()
    TRANSITION_BETWEEN_VALUES = auto()


def get_betas(beta_type: BetasType, kwargs: dict) -> BetaSchedule:
    if beta_type == BetasType.PIECEWISE_CONSTANT:
        betas = beta_piecewise_constant(**kwargs)
    elif beta_type == BetasType.CONSTANT:
        betas = beta_constant(**kwargs)
    elif beta_type == BetasType.TRANSITION_BETWEEN_VALUES:
        betas = beta_transition_between_values(**kwargs)
    return betas


def beta_piecewise_constant(boundaries: Any, values: Any, num_dim: int) -> BetaSchedule:
    one_dim_beta = piecewise_constant(boundaries, values)

    def beta_schedule(i):
        return jnp.repeat(one_dim_beta(i), num_dim)

    return beta_schedule


def beta_constant(value: float, num_dim: int) -> BetaSchedule:
    one_dim_beta = constant(value)

    def beta_schedule(i):
        return jnp.repeat(one_dim_beta(i), num_dim)

    return beta_schedule


def indicator(x):
    return (jnp.sign(x) + 1) / 2


def beta_transition_between_values(transition_start, step_size, decay_steps, final_step_size, num_dim,
                                   power=1.0) -> BetaSchedule:
    initial_beta = constant(step_size)
    later_beta = polynomial_decay(step_size, decay_steps, final_step_size, power=power)

    def beta_schedule(i):
        return indicator(transition_start - i) * jnp.repeat(initial_beta(i), num_dim) + indicator(
            i - transition_start) * jnp.repeat(later_beta(i - transition_start), num_dim)

    return beta_schedule
