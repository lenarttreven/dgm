import os
import pickle
import time
from functools import partial
from typing import List, Optional, Dict, Tuple, Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from jax import jit, value_and_grad
from jax.experimental.optimizers import adam, sgd
from jax.tree_util import tree_leaves
from sklearn.preprocessing import StandardScaler

from dgm.dynamics.dynamics_model import get_dynamics
from dgm.objectives.objective_builder import get_objective_builder
from dgm.plotter.plotter import Plotter
from dgm.schedules.betas import get_betas, BetaSchedule
from dgm.schedules.learning_rate import get_learning_rate
from dgm.schedules.weight_decay import get_weight_decay
from dgm.simulator.simulator import get_simulator
from dgm.smoother.smoother import get_smoother
from dgm.utils.helper_functions import unroll_dictionary, replace_str
from dgm.utils.representatives import Optimizer

Schedule = Callable[[int], float]
Range = Optional[Tuple[float, float]]
pytree = Any


class LearnSystem:
    def __init__(
            self,
            seed: int,
            data_generation: Dict,
            smoother: Dict,
            dynamics: Dict,
            betas: Dict,
            optimizer: Dict,
            priors: Dict,
            logging: Dict,
            numerical_correction: float,
            num_derivative_points_per_trajectory: int
    ):
        self.numerical_correction = numerical_correction
        self.track_just_loss = logging["track_just_loss"]
        self.core_type = smoother["core"]["type"]
        self.core_kwargs = smoother["core"]["kwargs"]
        # Adding kwargs here seems a bit hacky, it works but should you
        # think of a better solution don't hesitate to implement it
        self.core_kwargs['weight_key'] = "core"
        self.mean_head_type = smoother["mean_head"]["type"]
        self.mean_head_kwargs = smoother["mean_head"]["kwargs"]
        self.mean_head_kwargs['weight_key'] = "mean_head"
        self.kernel_core_type = smoother["kernel_core"]["type"]
        self.kernel_core_kwargs = smoother["kernel_core"]["kwargs"]
        self.kernel_core_kwargs['weight_key'] = "kernel_core"
        self.kernel_head_type = smoother['kernel_head']['type']
        self.kernel_head_kwargs = smoother["kernel_head"]["kwargs"]
        self.kernel_head_kwargs['weight_key'] = "kernel_head"

        self.learning_rate: Schedule = get_learning_rate(optimizer["learning_rate"]["type"],
                                                         optimizer["learning_rate"]["kwargs"])
        self.time_normalizer = StandardScaler()
        self.state_normalizer = StandardScaler()
        self.parameters = None
        self.kernel_type = smoother["kernel"]["type"]
        self.kernel_kwargs = smoother["kernel"]["kwargs"]
        self.optimizer_type = optimizer["type"]
        self.dynamics_model = dynamics["type"]
        self.dynamics_kwargs = dynamics['kwargs']
        self.simulator_type = data_generation["type"]

        self.times = data_generation["times"]
        self.test_times = data_generation["test_times"]
        self.initial_conditions = data_generation["initial_conditions"]
        self.num_trajectories = len(self.initial_conditions)
        self.state_dimension = len(self.initial_conditions[0])
        simulation_noise = data_generation['noise']
        self.simulation_noise = [None] * self.num_trajectories if simulation_noise is None else simulation_noise
        self.simulator_parameters = data_generation["parameters"]
        self.current_rng = jax.random.PRNGKey(seed)
        self.betas: BetaSchedule = get_betas(betas["type"], betas["kwargs"])

        self.wd_core = get_weight_decay(priors['wd_core']['type'], priors['wd_core']['kwargs'])
        self.wd_kernel_core = get_weight_decay(priors['wd_kernel_core']['type'], priors['wd_core']['kwargs'])
        self.wd_kernel_head = get_weight_decay(priors['wd_kernel_head']['type'], priors['wd_core']['kwargs'])
        self.wd_mean_head = get_weight_decay(priors['wd_mean_head']['type'], priors['wd_core']['kwargs'])
        self.wd_obs_noise = get_weight_decay(priors['wd_pure_kernel']['observation_noise']['type'],
                                             priors['wd_pure_kernel']['observation_noise']['kwargs'])
        self.wd_kernel_variance = get_weight_decay(priors['wd_pure_kernel']['kernel_variance']['type'],
                                                   priors['wd_pure_kernel']['kernel_variance']['kwargs'])
        self.wd_kernel_lengthscales = get_weight_decay(priors['wd_pure_kernel']['kernel_lengthscale']['type'],
                                                       priors['wd_pure_kernel']['kernel_lengthscale']['kwargs'])
        self.wd_dynamics: Schedule = get_weight_decay(priors["wd_dynamics"]['type'], priors["wd_dynamics"]['kwargs'])

        self.track_wandb = logging["track_wandb"]
        self.num_derivative_points_per_trajectory = num_derivative_points_per_trajectory
        self.simulator = get_simulator(simulator=self.simulator_type, **self.simulator_parameters)
        self._prepare_observed_data()
        self._normalize_data()
        self._prepare_data_for_training()
        self._prepare_smoother()
        self._prepare_dynamics_model()
        self._prepare_objective_builder()
        self._prepare_optimizer()
        self.plotter = Plotter(simulator=self.simulator, initial_conditions=self.initial_conditions)

    def _prepare_observed_data(self):
        self.current_rng, key = jax.random.split(self.current_rng)
        time_before_data = time.time()
        self.observations, self.ground_truth_states, self.ground_truth_derivatives = self.simulator.simulate_trajectories(
            initial_conditions=self.initial_conditions, times=self.times, sigmas=self.simulation_noise, rng=key)
        print("Time for data preparation", time.time() - time_before_data)

    def _normalize_data(self):
        time_before_normalization = time.time()
        all_times = jnp.concatenate(self.times, axis=0)
        all_observations = jnp.concatenate(self.observations, axis=0)
        self.time_normalizer.fit(all_times.reshape(-1, 1))
        self.state_normalizer.fit(all_observations)
        self.normalized_times = []
        self.normalized_observations = []
        self.normalized_initial_conditions = []
        self.normalized_test_times = []
        self.normalized_ground_truth_states = []
        self.normalized_ground_truth_derivatives = []
        derivative_scale = self.time_normalizer.scale_ / self.state_normalizer.scale_
        for i in range(self.num_trajectories):
            current_normalized_times = self.time_normalizer.transform(self.times[i].reshape(-1, 1))
            current_normalized_test_times = self.time_normalizer.transform(self.test_times[i].reshape(-1, 1))
            current_normalized_states = self.state_normalizer.transform(self.observations[i])
            current_normalized_initial_conditions = self.state_normalizer.transform(
                self.initial_conditions[i].reshape(1, -1))
            current_normalized_ground_truth_states = self.state_normalizer.transform(self.ground_truth_states[i])
            current_normalized_ground_truth_derivatives = derivative_scale * self.ground_truth_derivatives[i]
            self.normalized_times.append(jnp.array(current_normalized_times).reshape(-1))
            self.normalized_test_times.append(jnp.array(current_normalized_test_times).reshape(-1))
            self.normalized_observations.append(jnp.array(current_normalized_states))
            self.normalized_initial_conditions.append(jnp.array(current_normalized_initial_conditions.reshape(-1)))
            self.normalized_ground_truth_states.append(jnp.array(current_normalized_ground_truth_states))
            self.normalized_ground_truth_derivatives.append(jnp.array(current_normalized_ground_truth_derivatives))
        print("Time for normalization", time.time() - time_before_normalization)

    def _prepare_data_for_training(self):
        self.joint_normalized_test_times = jnp.concatenate(self.normalized_test_times)
        self.joint_normalized_times = jnp.concatenate(self.normalized_times)
        self.joint_normalized_observations = jnp.concatenate(self.normalized_observations)

        times_for_derivatives = []
        for traj_id in range(self.num_trajectories):
            min_time, max_time = jnp.min(self.normalized_times[traj_id]), jnp.max(self.normalized_times[traj_id])
            times_for_derivatives.append(jnp.linspace(min_time, max_time, self.num_derivative_points_per_trajectory))

        self.joint_normalized_times_for_derivatives = jnp.concatenate(times_for_derivatives)

        initial_conditions_to_pass = []
        initial_conditions_for_derivatives = []
        initial_conditions_for_test = []

        for traj_id in range(self.num_trajectories):
            initial_conditions_to_pass.append(
                jnp.repeat(self.normalized_initial_conditions[traj_id].reshape(1, -1),
                           self.normalized_times[traj_id].size, axis=0)
            )
            initial_conditions_for_derivatives.append(
                jnp.repeat(self.normalized_initial_conditions[traj_id].reshape(1, -1),
                           times_for_derivatives[traj_id].size, axis=0)
            )
            initial_conditions_for_test.append(
                jnp.repeat(self.normalized_initial_conditions[traj_id].reshape(1, -1),
                           self.test_times[traj_id].size, axis=0)
            )

        self.joint_repeated_normalized_initial_conditions = jnp.concatenate(initial_conditions_to_pass, axis=0)
        self.joint_repeated_normalized_initial_conditions_derivatives = jnp.concatenate(
            initial_conditions_for_derivatives,
            axis=0)
        self.joint_repeated_normalized_test_initial_conditions = jnp.concatenate(initial_conditions_for_test, axis=0)

    def _prepare_smoother(self):
        time_smoother = time.time()
        (
            self.smoother_init,
            self.smoother_apply,
            self.smoother_get_means_and_covariances_test,
            self.get_smoother_regularization,
        ) = get_smoother(kernel=self.kernel_type, kernel_kwargs=self.kernel_kwargs,
                         core_type=self.core_type, core_kwargs=self.core_kwargs,
                         mean_head_type=self.mean_head_type, mean_head_kwargs=self.mean_head_kwargs,
                         kernel_core_type=self.kernel_core_type, kernel_core_kwargs=self.kernel_core_kwargs,
                         kernel_head_type=self.kernel_head_type, kernel_head_kwargs=self.kernel_head_kwargs,
                         n_dim=self.state_dimension, numerical_correction=self.numerical_correction)
        print("Time for smoother preparation: ", time.time() - time_smoother)

    def _prepare_dynamics_model(self):
        time_dynamics = time.time()
        (
            self.dynamics_model_init,
            self.dynamics_model_apply,
            self.dynamics_for_plotting,
            self.dynamics_sample_trajectories,
            self.get_dynamics_regularization
        ) = get_dynamics(dynamics_model=self.dynamics_model, state_normalizer=self.state_normalizer,
                         time_normalizer=self.time_normalizer, state_dimension=self.state_dimension,
                         dynamics_kwargs=self.dynamics_kwargs)
        print("Time for dynamics preparation: ", time.time() - time_dynamics)

    def _prepare_objective_builder(self):
        time_objective_builder = time.time()
        self.current_rng, *keys = jax.random.split(self.current_rng, 3)
        dynamics_parameters = self.dynamics_model_init(keys[0])
        smoother_parameters = self.smoother_init(keys[1], self.state_dimension)
        self.parameters = {"smoother": smoother_parameters, "dynamics": dynamics_parameters, }
        self.num_dynamics_parameters = 0
        self.num_smoother_parameters = 0

        for leave in tree_leaves(dynamics_parameters):
            self.num_dynamics_parameters += leave.size
        for leave in tree_leaves(smoother_parameters):
            self.num_smoother_parameters += leave.size
        self.num_parameters = self.num_smoother_parameters + self.num_dynamics_parameters
        self.objective_builder = get_objective_builder(apply_smoother=self.smoother_apply,
                                                       apply_dynamics=self.dynamics_model_apply,
                                                       get_dynamics_regularization=self.get_dynamics_regularization,
                                                       get_smoother_regularization=self.get_smoother_regularization)
        print("Time to prepare objective builder", time.time() - time_objective_builder)
        time_objective_builder = time.time()
        self.values_and_grad = jit(value_and_grad(self.objective_builder, 0))
        print("Time to jit: ", time.time() - time_objective_builder)

    def _prepare_optimizer(self):
        if self.optimizer_type == Optimizer.ADAM:
            self.optimizer = adam
        elif self.optimizer_type == Optimizer.SGD:
            self.optimizer = sgd

    def train(self, number_of_steps):
        current_time = time.time()
        initial_time = current_time
        opt_init, opt_update, get_params = self.optimizer(self.learning_rate)
        params = opt_init(self.parameters)

        @jit
        def do_step(step, params):
            weights = {
                "kernel_variance": self.wd_kernel_variance(step),
                "kernel_lengthscale": self.wd_kernel_lengthscales(step),
                "obs_noise": self.wd_obs_noise(step),
                "dynamics": self.wd_dynamics(step),
                "core": self.wd_core(step),
                'kernel_core': self.wd_kernel_core(step),
                "kernel_head": self.wd_kernel_head(step),
                "mean_head": self.wd_mean_head(step)
            }
            loss, params_grad = self.values_and_grad(
                get_params(params),
                self.joint_normalized_times,
                self.joint_normalized_times_for_derivatives,
                self.joint_repeated_normalized_initial_conditions,
                self.joint_repeated_normalized_initial_conditions_derivatives,
                self.joint_normalized_observations,
                self.betas(step),
                weights
            )
            return loss, opt_update(step, params_grad, params)

        for step in range(number_of_steps):
            if step < 10:
                next_time = time.time()
                print("Time for step {}:".format(step), next_time - current_time)
                current_time = next_time
            loss, params = do_step(step, params)
            if self.track_wandb:
                if self.track_just_loss:
                    variables_dict = dict()
                    variables_dict["Loss"] = float(loss)
                else:
                    variables_dict = unroll_dictionary(get_params(params))
                    variables_dict["Loss"] = float(loss)
                wandb.log(variables_dict)
        time_spent_for_training = time.time() - initial_time
        print("Time spent for training:", time_spent_for_training, "seconds")
        self.parameters = get_params(params)
        # Save parameters_for_dgm
        if self.track_wandb:
            directory = os.path.join(wandb.run.dir, 'models')
            if not os.path.exists(directory):
                os.makedirs(directory)
            model_path = os.path.join('models', 'final_parameters.pkl')
            with open(os.path.join(wandb.run.dir, model_path), 'wb') as handle:
                pickle.dump(get_params(params), handle)
            wandb.save(os.path.join(wandb.run.dir, model_path), wandb.run.dir)

    def _compute_nll_per_dimension(self, denormalized_state_means, denormalized_state_variances,
                                   denormalized_derivative_means, denormalized_derivative_variances,
                                   denormalized_dynamics_means, denormalized_dynamics_variances,
                                   test_states, test_derivatives):
        nll_state = []
        nll_derivatives_smoother = []
        nll_derivatives_dynamics = []

        # Compute average (over dimension) average NLL score
        # Not over the range of self.num_trajectories but over the range of number of evaluated trajectories
        for i in range(len(denormalized_state_means)):
            nll_state.append(
                self._mean_nll(test_states[i], denormalized_state_means[i], denormalized_state_variances[i]))
            nll_derivatives_smoother.append(self._mean_nll(test_derivatives[i], denormalized_derivative_means[i],
                                                           denormalized_derivative_variances[i]))
            nll_derivatives_dynamics.append(self._mean_nll(test_derivatives[i], denormalized_dynamics_means[i],
                                                           denormalized_dynamics_variances[i]))
        return nll_state, nll_derivatives_smoother, nll_derivatives_dynamics

    @staticmethod
    def _mean_nll(test, mean_prediction, variance_prediction):
        mean_diff = (test - mean_prediction)
        nll_state_current = 0.5 * jnp.mean(mean_diff * mean_diff / variance_prediction)
        nll_state_current += 0.5 * jnp.mean(jnp.log(variance_prediction))
        nll_state_current += 0.5 * jnp.log(2 * jnp.pi)
        return nll_state_current

    @staticmethod
    def _prepare_nll_for_wandb(nll_state, nll_derivatives_smoother, nll_derivatives_dynamics, quantile):
        nll_state = jnp.array(nll_state)
        nll_derivatives_smoother = jnp.array(nll_derivatives_smoother)
        nll_derivatives_dynamics = jnp.array(nll_derivatives_dynamics)

        nll_state_median = jnp.median(nll_state)
        nll_derivatives_smoother_median = jnp.median(nll_derivatives_smoother)
        nll_derivatives_dynamics_median = jnp.median(nll_derivatives_dynamics)

        nll_state_lower_q = jnp.quantile(nll_state, q=1 - quantile)
        nll_derivatives_smoother_lower_q = jnp.quantile(nll_derivatives_smoother, q=1 - quantile)
        nll_derivatives_dynamics_lower_q = jnp.quantile(nll_derivatives_dynamics, q=1 - quantile)

        nll_state_upper_q = jnp.quantile(nll_state, q=quantile)
        nll_derivatives_smoother_upper_q = jnp.quantile(nll_derivatives_smoother, q=quantile)
        nll_derivatives_dynamics_upper_q = jnp.quantile(nll_derivatives_dynamics, q=quantile)

        variables_dict = dict()

        variables_dict['nll_state_mean'] = float(jnp.mean(nll_state))
        variables_dict['nll_derivatives_smoother_mean'] = float(jnp.mean(nll_derivatives_smoother))
        variables_dict['nll_derivatives_dynamics_mean'] = float(jnp.mean(nll_derivatives_dynamics))

        variables_dict['nll_state_median'] = float(nll_state_median)
        variables_dict['nll_derivatives_smoother_median'] = float(nll_derivatives_smoother_median)
        variables_dict['nll_derivatives_dynamics_median'] = float(nll_derivatives_dynamics_median)

        variables_dict['nll_state_lower_q'] = float(nll_state_lower_q)
        variables_dict['nll_derivatives_smoother_lower_q'] = float(nll_derivatives_smoother_lower_q)
        variables_dict['nll_derivatives_dynamics_lower_q'] = float(nll_derivatives_dynamics_lower_q)

        variables_dict['nll_state_upper_q'] = float(nll_state_upper_q)
        variables_dict['nll_derivatives_smoother_upper_q'] = float(nll_derivatives_smoother_upper_q)
        variables_dict['nll_derivatives_dynamics_upper_q'] = float(nll_derivatives_dynamics_upper_q)
        return variables_dict

    def _denormalize(self, state_means, state_variances, derivative_means, derivative_variances, dynamics_means,
                     dynamics_variances):
        denormalized_state_means = self.state_normalizer.inverse_transform(state_means)
        denormalized_state_variances = self.state_normalizer.scale_ ** 2 * state_variances

        derivative_scale = self.state_normalizer.scale_ / self.time_normalizer.scale_

        denormalized_derivative_means = derivative_scale * derivative_means
        denormalized_derivative_variances = derivative_scale ** 2 * derivative_variances
        denormalized_dynamics_means = derivative_scale * dynamics_means
        denormalized_dynamics_variances = derivative_scale ** 2 * dynamics_variances
        return denormalized_state_means, denormalized_state_variances, denormalized_derivative_means, \
               denormalized_derivative_variances, denormalized_dynamics_means, denormalized_dynamics_variances

    @staticmethod
    def join_trajectories(initial_conditions: List[jnp.array], times: List[jnp.array]) -> Tuple[
        pytree, jnp.array, jnp.array]:
        # initial_conditions are of shape (num_dim, )
        # times are of shape (num_times, )
        n_trajectories = len(times)
        joint_times = jnp.concatenate(times)
        joint_initial_conditions = []

        for traj_id in range(n_trajectories):
            joint_initial_conditions.append(
                jnp.repeat(initial_conditions[traj_id].reshape(1, -1), times[traj_id].size, axis=0))
        joint_initial_conditions = jnp.concatenate(joint_initial_conditions, axis=0)
        return list(map(len, times)), joint_times, joint_initial_conditions

    @staticmethod
    def split_trajectories(trajectory_lengths, *data) -> List[List[jnp.array]]:
        start_index = 0
        n_data = len(data)
        separated_data = [[] for _ in range(n_data)]
        for length in trajectory_lengths:
            for index, datum in enumerate(data):
                separated_data[index].append(datum[start_index: start_index + length, :])
            start_index += length
        return separated_data

    def _all_predictions(self, joint_normalized_times, joint_repeated_normalized_initial_conditions,
                         trajectory_lengths):
        state_means, state_variances, \
        derivative_means, derivative_variances = self.smoother_get_means_and_covariances_test(
            joint_normalized_times,
            self.joint_normalized_times,
            joint_repeated_normalized_initial_conditions,
            self.joint_repeated_normalized_initial_conditions,
            self.joint_normalized_observations,
            self.parameters["smoother"],
        )
        dynamics_means, dynamics_variances = self.dynamics_model_apply(self.parameters["dynamics"], state_means)

        # Denormalize everything
        denormalized_state_means, denormalized_state_variances, denormalized_derivative_means, \
        denormalized_derivative_variances, denormalized_dynamics_means, denormalized_dynamics_variances = self._denormalize(
            state_means, state_variances, derivative_means, derivative_variances, dynamics_means, dynamics_variances)

        # Here all data are one big jnp.array now we split it since we would like to perform per trajectory analysis
        return self.split_trajectories(trajectory_lengths, denormalized_state_means, denormalized_state_variances,
                                       denormalized_derivative_means, denormalized_derivative_variances,
                                       denormalized_dynamics_means, denormalized_dynamics_variances)

    def evaluate_models(self, ground_truth: bool = True, initial_conditions: Optional = None, times: Optional = None,
                        quantile=0.8):
        if ground_truth:
            initial_conditions = self.initial_conditions
            times = self.test_times
        trajectory_lengths, joint_times, joint_repeated_initial_conditions = self.join_trajectories(initial_conditions,
                                                                                                    times)

        joint_normalized_times = self.time_normalizer.transform(joint_times.reshape(-1, 1)).reshape(-1)
        joint_repeated_normalized_initial_conditions = self.state_normalizer.transform(
            joint_repeated_initial_conditions)

        denormalized_state_means, denormalized_state_variances, \
        denormalized_derivative_means, denormalized_derivative_variances, \
        denormalized_dynamics_means, denormalized_dynamics_variances = self._all_predictions(
            joint_normalized_times,
            joint_repeated_normalized_initial_conditions,
            trajectory_lengths)

        # Prepare (not normalized) ground truth prediction
        self.current_rng, subkey = jax.random.split(self.current_rng)
        test_states, test_derivatives = self.simulator.simulate_trajectories(
            initial_conditions=initial_conditions, times=times, sigmas=[None] * len(times), rng=subkey)[1:]

        # Compute average (per dimension) average NLL score
        nll_state, nll_derivatives_smoother, nll_derivatives_dynamics = self._compute_nll_per_dimension(
            denormalized_state_means, denormalized_state_variances,
            denormalized_derivative_means, denormalized_derivative_variances,
            denormalized_dynamics_means, denormalized_dynamics_variances,
            test_states, test_derivatives)

        variables_dict = self._prepare_nll_for_wandb(nll_state, nll_derivatives_smoother, nll_derivatives_dynamics,
                                                     quantile)
        if self.track_wandb:
            wandb.log(variables_dict)
        return variables_dict

    def plot_learned_vector_field(self, x_range: Range = None, y_range: Range = None):
        get_dynamics_derivatives = partial(self.dynamics_for_plotting, self.parameters["dynamics"])
        num_trajectories = len(self.observations)
        max_per_trajectory = [jnp.max(self.observations[i], 0) for i in range(num_trajectories)]
        max_all_trajectories = jnp.max(jnp.vstack(max_per_trajectory), 0)
        if x_range is None:
            x_range = (0, max_all_trajectories[0] * 1.1)
        if y_range is None:
            y_range = (0, max_all_trajectories[1] * 1.1)
        x, y = jnp.meshgrid(
            jnp.linspace(x_range[0], x_range[1], 20),
            jnp.linspace(y_range[0], y_range[1], 20),
        )
        u_mean_learned, v_mean_learned, norm_mean_learned, volume_covariance_learned, max_covariance_eigenvalue = get_dynamics_derivatives(
            x, y)
        u_true, v_true, norm_true = self.simulator.prepare_vector_field_for_plotting(x, y)

        fig = self.plotter.plot_learned_vector_field(
            initial_conditions=self.initial_conditions,
            observations=self.observations,
            grid=(x, y),
            true_vector_field=(u_true, v_true, norm_true),
            learned_vector_field_mean=(u_mean_learned, v_mean_learned, norm_mean_learned),
            volume_covariance_learned=volume_covariance_learned,
            max_covariance_eigenvalue=max_covariance_eigenvalue,
        )
        fig.tight_layout()
        if self.track_wandb:
            wandb.log({'vector_field_plot': wandb.Image(fig)})

    def plot_trajectories_at_times(self, add_all_trajectories: bool = False):
        print('Before computing the values for plotting')
        trajectory_lengths, joint_normalized_test_times, joint_repeated_normalized_test_initial_conditions = self.join_trajectories(
            self.normalized_initial_conditions,
            self.normalized_test_times)
        denormalized_state_means, denormalized_state_variances, \
        denormalized_derivative_means, denormalized_derivative_variances, \
        denormalized_dynamics_means, denormalized_dynamics_variances = self._all_predictions(
            joint_normalized_test_times,
            joint_repeated_normalized_test_initial_conditions,
            trajectory_lengths)

        print("Plotting")
        figure_smoother_states, figure_smoother_derivatives, figure_dynamics_derivatives = self.plotter.plot_at_times(
            self.test_times,
            denormalized_state_means,
            denormalized_state_variances,
            denormalized_derivative_means,
            denormalized_derivative_variances,
            denormalized_dynamics_means,
            denormalized_dynamics_variances,
            train_times=self.times,
            observations=self.observations,
            all_initial_conditions=self.initial_conditions if add_all_trajectories else None
        )
        figure_smoother_states.tight_layout()
        figure_smoother_derivatives.tight_layout()
        figure_dynamics_derivatives.tight_layout()

        state_filename = 'smoother_states_with_all_trajectories' if add_all_trajectories else 'smoother_states'
        if self.track_wandb:
            wandb.log({state_filename: wandb.Image(figure_smoother_states),
                       'smoother_derivatives': wandb.Image(figure_smoother_derivatives),
                       'dynamics_derivatives': wandb.Image(figure_dynamics_derivatives)})

    def save_data_for_plotting(self):
        trajectory_lengths, joint_normalized_test_times, joint_repeated_normalized_test_initial_conditions = self.join_trajectories(
            self.normalized_initial_conditions,
            self.normalized_test_times)
        denormalized_state_means, denormalized_state_variances, \
        denormalized_derivative_means, denormalized_derivative_variances, \
        denormalized_dynamics_means, denormalized_dynamics_variances = self._all_predictions(
            joint_normalized_test_times,
            joint_repeated_normalized_test_initial_conditions,
            trajectory_lengths)

        data = self.plotter.save_data(
            self.test_times,
            denormalized_state_means,
            denormalized_state_variances,
            denormalized_derivative_means,
            denormalized_derivative_variances,
            denormalized_dynamics_means,
            denormalized_dynamics_variances,
            train_times=self.times,
            observations=self.observations,
            all_initial_conditions=self.initial_conditions
        )
        if self.track_wandb:
            directory = os.path.join(wandb.run.dir, 'data')
            if not os.path.exists(directory):
                os.makedirs(directory)
            data_path = os.path.join('data', 'test_plot_data.pkl')
            with open(os.path.join(wandb.run.dir, data_path), 'wb') as handle:
                pickle.dump(data, handle)
            wandb.save(os.path.join(wandb.run.dir, data_path), wandb.run.dir)

        return data

    def bayesian_path_prediction_from_dynamics(self, rng: np.ndarray, initial_condition: jnp.array, times: jnp.array,
                                               num_samples: int, q: float = 0.7, add_all_trajectories: bool = False):
        normalized_times = self.time_normalizer.transform(times.reshape(-1, 1)).reshape(-1)
        normalized_initial_conditions = self.state_normalizer.transform(initial_condition.reshape(1, -1)).reshape(-1)
        median, upper_quantile, lower_quantile = self.dynamics_sample_trajectories(rng, self.parameters['dynamics'],
                                                                                   normalized_initial_conditions,
                                                                                   normalized_times, num_samples, q)

        denormalized_mean = self.state_normalizer.inverse_transform(median)
        denormalized_lower_quantile = self.state_normalizer.inverse_transform(lower_quantile)
        denormalized_upper_quantile = self.state_normalizer.inverse_transform(upper_quantile)
        figure = self.plotter.plot_sample_trajectories(initial_condition, times, denormalized_mean,
                                                       denormalized_lower_quantile, denormalized_upper_quantile, q,
                                                       all_initial_conditions=self.initial_conditions if add_all_trajectories else None)
        if self.track_wandb:
            key = 'Bayesian integration from initial condition: ' + replace_str(str(initial_condition))
            wandb.log({key: wandb.Image(figure)})

    def predict_trajectory(self, initial_condition, times, add_all_trajectories: bool = False):
        normalized_initial_condition = self.state_normalizer.transform(initial_condition.reshape(1, -1))
        normalized_times = self.time_normalizer.transform(times.reshape(-1, 1)).reshape(-1)
        repeated_normalized_initial_condition = jnp.repeat(normalized_initial_condition, normalized_times.size, axis=0)
        denormalized_state_means, denormalized_state_variances, \
        denormalized_derivative_means, denormalized_derivative_variances, \
        denormalized_dynamics_means, denormalized_dynamics_variances = self._all_predictions(
            normalized_times,
            repeated_normalized_initial_condition,
            [len(times)])

        figure_prediction = self.plotter.plot_predicted_trajectory(
            initial_condition=[initial_condition],
            test_times=[times],
            state_means=denormalized_state_means,
            state_variances=denormalized_state_variances,
            derivative_means=denormalized_derivative_means,
            derivative_variances=denormalized_derivative_variances,
            dynamics_means=denormalized_dynamics_means,
            dynamics_variances=denormalized_dynamics_variances,
            all_initial_conditions=self.initial_conditions if add_all_trajectories else None
        )
        if self.track_wandb:
            key = 'Initial condition: ' + replace_str(str(initial_condition))
            wandb.log({key: wandb.Image(figure_prediction)})

    def save_predicted_trajectory(self, initial_condition, times):
        # initial condition is one dimensional jnp.array, times is one dimensional jnp.array
        normalized_initial_condition = self.state_normalizer.transform(initial_condition.reshape(1, -1))
        normalized_times = self.time_normalizer.transform(times.reshape(-1, 1)).reshape(-1)
        repeated_normalized_initial_condition = jnp.repeat(normalized_initial_condition, normalized_times.size, axis=0)

        denormalized_state_means, denormalized_state_variances, \
        denormalized_derivative_means, denormalized_derivative_variances, \
        denormalized_dynamics_means, denormalized_dynamics_variances = self._all_predictions(
            normalized_times,
            repeated_normalized_initial_condition,
            [len(times)])

        data = self.plotter.save_plot_predicted_trajectory(
            initial_condition=[initial_condition],
            test_times=[times],
            state_means=denormalized_state_means,
            state_variances=denormalized_state_variances,
            derivative_means=denormalized_derivative_means,
            derivative_variances=denormalized_derivative_variances,
            dynamics_means=denormalized_dynamics_means,
            dynamics_variances=denormalized_dynamics_variances,
            all_initial_conditions=self.initial_conditions
        )

        if self.track_wandb:
            directory = os.path.join(wandb.run.dir, 'data')
            if not os.path.exists(directory):
                os.makedirs(directory)
            data_path = os.path.join('data', 'predicted_trajectory{}.pkl'.format(replace_str(str(initial_condition))))
            with open(os.path.join(wandb.run.dir, data_path), 'wb') as handle:
                pickle.dump(data, handle)
            wandb.save(os.path.join(wandb.run.dir, data_path), wandb.run.dir)
        return data


if __name__ == "__main__":
    pass
