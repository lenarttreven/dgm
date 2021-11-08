import argparse
import time
from typing import Callable
import os
os.environ['JAX_ENABLE_X64'] = 'True'
import jax
import jax.numpy as jnp
import wandb
from jax.experimental.stax import Dense, Identity, Sigmoid

from dgm.main.learn_system import LearnSystem
from dgm.schedules.betas import BetasType
from dgm.schedules.learning_rate import LearningRateType
from dgm.schedules.weight_decay import WeightDecayType
from dgm.utils.representatives import FeaturesToFeaturesType, TimeAndStatesToFeaturesType
from dgm.utils.representatives import KernelType, Optimizer, DynamicsModel, SimulatorType

Schedule = Callable[[int], float]

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--n_points_per_dimension', action='store', type=int, default=1)
    args = my_parser.parse_args()

    kernel_seed = 135
    kernel_rng = jax.random.PRNGKey(kernel_seed)

    weight_decay = 1
    lr = 0.1
    seed = 0

    num_points_on_trajectory = 100
    num_der_points = 100
    n_points_per_dimension = args.n_points_per_dimension

    num_x_points = n_points_per_dimension
    num_y_points = n_points_per_dimension

    num_trajectories = num_x_points * num_y_points
    my_times = [jnp.linspace(0, 10, num_points_on_trajectory, dtype=jnp.float64) for _ in range(num_trajectories)]
    my_test_times = [jnp.linspace(0, 10, 100, dtype=jnp.float64) for _ in range(num_trajectories)]

    points_x = jnp.linspace(0.5, 1.5, num_x_points)
    points_y = jnp.linspace(0.5, 1.5, num_y_points)
    out = jnp.stack(jnp.meshgrid(points_x, points_y))
    my_initial_conditions = [out[:, i, j] for i in range(num_x_points) for j in range(num_y_points)]

    my_stds_for_simulation = [jnp.array([0.1, 0.1], dtype=jnp.float64) for _ in range(num_trajectories)]
    my_simulator_parameters = {"params": jnp.array([1, 1, 1, 1])}

    track_wandb = True
    track_just_loss = True
    numerical_correction = 1e-3

    final_beta = 1
    transition_len = 500
    boundary = 1000

    num_iterations = boundary + 500

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
                    'n_features': 5,  # depends on the feature extractor chosen
                },
            },
            'core': {
                'type': TimeAndStatesToFeaturesType.NEURAL_NET,
                'kwargs': {'serial_input': [Dense(20), Sigmoid, Dense(20), Sigmoid, Dense(20), Sigmoid]}
            },
            'kernel_core': {
                'type': FeaturesToFeaturesType.NEURAL_NET,
                'kwargs': {'serial_input': [Identity]}
            },
            'kernel_head': {
                'type': FeaturesToFeaturesType.NEURAL_NET,
                'kwargs': {'serial_input': [Dense(5)]}
            },
            'mean_head': {
                'type': FeaturesToFeaturesType.NEURAL_NET,
                'kwargs': {'serial_input': [Dense(2)]}
            }
        },
        'dynamics': {
            'type': DynamicsModel.JOINT_SMALL_DYNAMICS,
            'kwargs': {}
        },
        'betas': {
            'type': BetasType.TRANSITION_BETWEEN_VALUES,
            'kwargs': {'transition_start': 0, 'step_size': 0, 'decay_steps': transition_len,
                       'final_step_size': final_beta, 'power': 1, 'num_dim': 2},
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
                'type': WeightDecayType.POLYNOMIAL_DECAY,
                'kwargs': {'step_size': 0, 'decay_steps': transition_len, 'final_step_size': weight_decay, 'power': 1}
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
            # dir='/cluster/scratch/trevenl',
            project="RFF sacling test",
            config=run_dict,
            name="RFF".format(run_dict["smoother"]["kernel"]["kwargs"]["n_rff"]) + "num points {}".format(num_x_points),
            group="Scaling training test."
        )
        config = wandb.config

    model = LearnSystem(**run_dict)
    model.train(num_iterations)
    model.plot_trajectories_at_times()

    start_time = time.time()

    with jax.disable_jit():
        quantile = 0.7
        key = jax.random.PRNGKey(0)
        num_test_points = 10
        test_times = [jnp.linspace(0, 10, 100) for _ in range(num_test_points)]
        initial_conditions = []
        for _ in range(num_test_points):
            key, subkey = jax.random.split(key)
            initial_condition = jax.random.uniform(key=subkey, shape=(2,), minval=0.5, maxval=1.5)
            initial_conditions.append(initial_condition)
        print('Before evaluating models')
        evaluation_start_time = time.time()
        # We evaluate prediction.
        model.evaluate_models(ground_truth=False, initial_conditions=initial_conditions, times=test_times,
                              quantile=quantile)
        print('Evaluation time: ', time.time() - evaluation_start_time, ' seconds.')
        print('After evaluating models')

    total_script_time = time.time() - start_time
    print("Total script time:", total_script_time, "seconds")
