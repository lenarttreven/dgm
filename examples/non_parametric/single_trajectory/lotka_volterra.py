from typing import Callable
import os
os.environ['JAX_ENABLE_X64'] = 'True'

import jax.numpy as jnp
import jax

import wandb
from dgm.main.learn_system import LearnSystem
from dgm.schedules.betas import BetasType
from dgm.schedules.learning_rate import LearningRateType
from dgm.schedules.weight_decay import WeightDecayType
from dgm.utils.representatives import KernelType, Optimizer, DynamicsModel, SimulatorType, FeaturesToFeaturesType
from dgm.utils.representatives import TimeAndStatesToFeaturesType

Schedule = Callable[[int], float]

if __name__ == '__main__':
    weight_decay = 1
    lr = 0.1
    seed = 0

    num_points_on_trajectory = 100
    num_der_points = 100
    num_trajectories = 1

    kernel_seed = 135
    kernel_rng = jax.random.PRNGKey(kernel_seed)

    my_times = [jnp.linspace(0, 10, num_points_on_trajectory, dtype=jnp.float64) for _ in range(num_trajectories)]
    my_test_times = [jnp.linspace(0, 10, 200, dtype=jnp.float64) for _ in range(num_trajectories)]

    my_initial_conditions = [jnp.array([1, 2], dtype=jnp.float64)]
    my_stds_for_simulation = [jnp.array([0.1, 0.1], dtype=jnp.float64) for _ in range(num_trajectories)]

    my_simulator_parameters = {"params": jnp.array([1, 1, 1, 1])}

    track_wandb = True
    track_just_loss = True
    der_conditioned_on_states = False
    numerical_correction = 1e-3

    final_beta = 1
    transition_len = 1200
    boundary = 1500

    num_iterations = boundary + 1000

    run_dict = {
        'seed': seed,
        'data_generation': {
            'type': SimulatorType.LOTKA_VOLTERRA,
            'parameters': my_simulator_parameters,
            'noise': my_stds_for_simulation,
            'times': my_times,
            'test_times': my_test_times,
            'initial_conditions': my_initial_conditions
        },
        'smoother': {
            'kernel': {
                'type': KernelType.RBF_RFF,
                'kwargs': {
                    'feature_rng': kernel_rng,
                    'n_rff': 40,
                    'n_features': 1,  # depends on the feature extractor chosen
                },
            },
            'core': {
                'type': TimeAndStatesToFeaturesType.IDENTITY,
                'kwargs': {}
            },
            'kernel_core': {
                'type': FeaturesToFeaturesType.FIRST_FEATURE,
                'kwargs': {}
            },
            'kernel_head': {
                'type': FeaturesToFeaturesType.LINEAR,
                'kwargs': {'n_out': 1}
            },
            'mean_head': {
                'type': FeaturesToFeaturesType.ZERO,
                'kwargs': {'n_out': 2}
            }
        },
        'dynamics': {
            'type': DynamicsModel.JOINT_SMALL_DYNAMICS,
            'kwargs': {}
        },
        'betas': {
            'type': BetasType.TRANSITION_BETWEEN_VALUES,
            'kwargs': {'transition_start': 0, 'step_size': 0, 'decay_steps': transition_len,
                       'final_step_size': final_beta, 'power': 0.8, 'num_dim': 2},
        },
        'optimizer': {
            'type': Optimizer.ADAM,
            'learning_rate': {
                'type': LearningRateType.PIECEWISE_CONSTANT,
                'kwargs': {'boundaries': [boundary], 'values': [lr, 0.01]},
            },
        },
        'priors': {
            'wd_core': {
                'type': WeightDecayType.CONSTANT,
                'kwargs': {'step_size': weight_decay}
            },
            'wd_mean_head': {
                'type': WeightDecayType.CONSTANT,
                'kwargs': {'step_size': weight_decay}
            },
            'wd_kernel_core': {
                'type': WeightDecayType.CONSTANT,
                'kwargs': {'step_size': weight_decay}
            },
            'wd_kernel_head': {
                'type': WeightDecayType.CONSTANT,
                'kwargs': {'step_size': weight_decay}
            },
            'wd_pure_kernel': {
                'kernel_variance': {
                    'type': WeightDecayType.CONSTANT,
                    'kwargs': {'step_size': 0}
                },
                'kernel_lengthscale': {
                    'type': WeightDecayType.CONSTANT,
                    'kwargs': {'step_size': 0}
                },
                "observation_noise": {
                    'type': WeightDecayType.CONSTANT,
                    'kwargs': {'step_size': 0}
                }
            },
            'wd_dynamics': {
                'type': WeightDecayType.CONSTANT,
                'kwargs': {'step_size': weight_decay}
            }
        },
        'logging': {
            'track_wandb': track_wandb,
            'track_just_loss': track_just_loss,
        },
        'numerical_correction': numerical_correction,
        'num_derivative_points_per_trajectory': num_der_points,
    }

    if track_wandb:
        wandb.init(
            project="Lotka Volterra",
            config=run_dict,
            name="Single trajectory",
        )
        config = wandb.config

    model = LearnSystem(**run_dict)
    model.train(num_iterations)
