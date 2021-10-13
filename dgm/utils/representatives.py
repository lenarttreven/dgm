from enum import Enum, auto


class KernelType(Enum):
    RBF = auto()
    RBF_NO_LENGTHSCALES = auto()
    RBF_NO_LENGTHSCALES_NO_VARIANCE = auto()
    RBF_NO_VARIANCE = auto()
    RBF_RFF = auto()
    LINEAR = auto()


class Space(Enum):
    STATE = auto()
    DERIVATIVE = auto()


class Optimizer(Enum):
    ADAM = auto()
    LBFGS = auto()
    SGD = auto()


class DynamicsModel(Enum):
    PARAMETRIC_LOTKA_VOLTERRA = auto()
    LOTKA_VOLTERRA_WITH_HETEROSCEDASTIC_NOISE = auto()
    PERFECTLY_ADAPTED_HETEROSCEDASTIC_NOISE = auto()
    NEGATIVE_FEEDBACK_OSCILATOR_HETEROSCEDASTIC_NOISE = auto()
    LORENZ_HETEROSCEDASTIC_NOISE = auto()
    CSTR_HETEROSCEDASTIC_NOISE = auto()
    TWO_DIMENSIONAL_NODE = auto()
    JOINT_SMALL_DYNAMICS = auto()
    DOUBLE_PENDULUM_HETEROSCEDASTIC_NOISE = auto()
    QUADROCOPTER_HETEROSCEDASTIC_NOISE = auto()
    LINEAR_HETEROSCEDASTIC_NOISE = auto()
    LINEAR_PROD_HETEROSCEDASTIC_NOISE = auto()
    JOINT_MEDIUM_DYNAMICS = auto()
    JOINT_BIG_DYNAMICS = auto()
    JOINT_NN = auto()


class SimulatorType(Enum):
    LOTKA_VOLTERRA = auto()
    PERFECTLY_ADAPTED = auto()
    NEGATIVE_FEEDBACK_OSCILATOR = auto()
    LORENZ = auto()
    CSTR = auto()
    DOUBLE_PENDULUM = auto()
    QUADROCOPTER = auto()
    LINEAR = auto()


class Statistics(Enum):
    MEDIAN = auto()
    MEAN = auto()


class KernelFeatureCreatorType(Enum):
    RBF_RFF = auto()
    LINEAR = auto()


class FeaturesToFeaturesType(Enum):
    LINEAR = auto()
    LINEAR_WITH_SIGMOID = auto()
    FIRST_FEATURE = auto()
    ZERO = auto()
    IDENTITY = auto()
    NEURAL_NET = auto()
    NEURAL_NET_WITH_SERIAL_SPECIFICATION = auto()


class TimeAndStatesToFeaturesType(Enum):
    NEURAL_NET = auto()
    IDENTITY = auto()
    NN_CORE_WITH_SERIAL_SPECIFICATION = auto()
    JUST_TIME = auto()
