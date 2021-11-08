from enum import Enum, auto
from typing import Callable

from jax.experimental.optimizers import constant, piecewise_constant
from dgm.schedules.betas import polynomial_decay

Schedule = Callable[[int], float]


class WeightDecayType(Enum):
    PIECEWISE_CONSTANT = auto()
    CONSTANT = auto()
    POLYNOMIAL_DECAY = auto()


def get_weight_decay(wd_type: WeightDecayType, kwargs: dict) -> Schedule:
    if wd_type == WeightDecayType.PIECEWISE_CONSTANT:
        weight_decay = piecewise_constant(**kwargs)
    elif wd_type == WeightDecayType.CONSTANT:
        weight_decay = constant(**kwargs)
    elif wd_type == WeightDecayType.POLYNOMIAL_DECAY:
        weight_decay = polynomial_decay(**kwargs)
    return weight_decay
