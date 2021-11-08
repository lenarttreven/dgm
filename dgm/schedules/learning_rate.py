from enum import Enum, auto
from typing import Callable

from jax.experimental.optimizers import constant, piecewise_constant
from dgm.schedules.betas import polynomial_decay

Schedule = Callable[[int], float]


class LearningRateType(Enum):
    PIECEWISE_CONSTANT = auto()
    CONSTANT = auto()
    POLYNOMIAL_DECAY = auto()


def get_learning_rate(lr_type: LearningRateType, kwargs: dict) -> Schedule:
    if lr_type == LearningRateType.PIECEWISE_CONSTANT:
        learning_rate = piecewise_constant(**kwargs)
    elif lr_type == LearningRateType.CONSTANT:
        learning_rate = constant(**kwargs)
    elif lr_type == LearningRateType.POLYNOMIAL_DECAY:
        learning_rate = polynomial_decay(**kwargs)
    return learning_rate

