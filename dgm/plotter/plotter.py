from typing import List, Optional

import jax.numpy as jnp
import matplotlib.pyplot as plt

from dgm.utils.representatives import Space
from dgm.simulator.simulator import Simulator
from dgm.utils.representatives import Statistics


class Plotter:
    def __init__(self, simulator: Simulator, initial_conditions: List[jnp.array]):
        self.simulator = simulator
        self.initial_conditions = initial_conditions

    def plot_at_times(
            self,
            prediction_times: List[jnp.array],  # jnp.array of shape (n,)
            state_means: List[jnp.array],  # jnp.array of shape (n, num_dimensions)
            state_variances: List[jnp.array],  # jnp.array of shape (n, num_dimensions)
            derivatives_means: List[jnp.array],  # jnp.array of shape (n, num_dimensions)
            derivative_variances: List[jnp.array],  # jnp.array of shape (n, num_dimensions)
            dynamics_means: List[jnp.array],  # jnp.array of shape (n, num_dimensions)
            dynamics_variances: List[jnp.array],  # jnp.array of shape (n, num_dimensions)
            train_times: List[jnp.array],  # jnp.array of shape (n, 1)
            observations: List[jnp.array],  # jnp.array of shape (n, num_dimensions)
            all_initial_conditions: Optional[List[jnp.array]] = None
    ):
        # Plot states
        num_trajectories = len(state_means)
        num_dimensions = state_means[0].shape[1]
        figure_smoother_states, ax_states = plt.subplots(
            num_trajectories,
            num_dimensions,
            figsize=(5 * num_dimensions + 4, 4 * num_trajectories),
        )
        ax_states = ax_states.reshape(num_trajectories, num_dimensions)

        state_stds = [jnp.sqrt(jnp.clip(x, a_min=0)) for x in state_variances]

        values_lower = [state_means[i] - 1.960 * state_stds[i] for i in range(len(state_means))]
        values_upper = [state_means[i] + 1.960 * state_stds[i] for i in range(len(state_means))]

        print('Plot states')
        self._create_plot(
            axs=ax_states,
            plot_times=prediction_times,
            plot_values=state_means,
            plot_values_lower=values_lower,
            plot_values_upper=values_upper,
            q=95,
            statistics_type=Statistics.MEAN,
            times_train=train_times,
            observations=observations,
            add_data=True,
            space=Space.STATE,
            all_initial_conditions=all_initial_conditions
        )
        print('After plot states')
        figure_smoother_derivatives, ax_derivatives = plt.subplots(
            num_trajectories,
            num_dimensions,
            figsize=(5 * num_dimensions + 4, 4 * num_trajectories),
        )
        ax_derivatives = ax_derivatives.reshape(num_trajectories, num_dimensions)
        # figure_smoother_derivatives.suptitle("Trajectories Derivatives", fontsize=20)

        derivative_stds = [jnp.sqrt(jnp.clip(x, a_min=0)) for x in derivative_variances]
        derivative_values_lower = [derivatives_means[i] - 1.960 * derivative_stds[i] for i in
                                   range(len(derivatives_means))]
        derivative_values_upper = [derivatives_means[i] + 1.960 * derivative_stds[i] for i in
                                   range(len(derivatives_means))]

        print('Plot smoother derivatives')
        self._create_plot(
            axs=ax_derivatives,
            plot_times=prediction_times,
            plot_values=derivatives_means,
            plot_values_lower=derivative_values_lower,
            plot_values_upper=derivative_values_upper,
            q=95,
            statistics_type=Statistics.MEAN,
            times_train=train_times,
            observations=observations,
            add_data=False,
            space=Space.DERIVATIVE,
        )
        # Plot dynamics derivatives
        figure_dynamics_derivatives, ax_dynamics = plt.subplots(num_trajectories, num_dimensions,
                                                                figsize=(5 * num_dimensions + 4, 4 * num_trajectories))
        ax_dynamics = ax_dynamics.reshape(num_trajectories, num_dimensions)

        # figure_dynamics_derivatives.suptitle("Dynamics Derivatives", fontsize=20)
        dynamics_stds = [jnp.sqrt(jnp.clip(x, a_min=0)) for x in dynamics_variances]
        dynamics_values_lower = [dynamics_means[i] - 1.960 * dynamics_stds[i] for i in
                                 range(len(dynamics_means))]
        dynamics_values_upper = [dynamics_means[i] + 1.960 * dynamics_stds[i] for i in
                                 range(len(dynamics_means))]

        print('Plot dynamics derivatives')
        self._create_plot(
            axs=ax_dynamics,
            plot_times=prediction_times,
            plot_values=dynamics_means,
            plot_values_lower=dynamics_values_lower,
            plot_values_upper=dynamics_values_upper,
            q=95,
            statistics_type=Statistics.MEAN,
            times_train=train_times,
            observations=observations,
            add_data=False,
            space=Space.DERIVATIVE,
        )
        return figure_smoother_states, figure_smoother_derivatives, figure_dynamics_derivatives

    def save_data(
            self,
            prediction_times: List[jnp.array],  # jnp.array of shape (n,)
            state_means: List[jnp.array],  # jnp.array of shape (n, num_dimensions)
            state_variances: List[jnp.array],  # jnp.array of shape (n, num_dimensions)
            derivatives_means: List[jnp.array],  # jnp.array of shape (n, num_dimensions)
            derivative_variances: List[jnp.array],  # jnp.array of shape (n, num_dimensions)
            dynamics_means: List[jnp.array],  # jnp.array of shape (n, num_dimensions)
            dynamics_variances: List[jnp.array],  # jnp.array of shape (n, num_dimensions)
            train_times: List[jnp.array],  # jnp.array of shape (n, 1)
            observations: List[jnp.array],  # jnp.array of shape (n, num_dimensions)
            all_initial_conditions: Optional[List[jnp.array]] = None
    ):
        data = dict()
        # Create smoother states data

        state_stds = [jnp.sqrt(jnp.clip(x, a_min=0)) for x in state_variances]
        values_lower = [state_means[i] - 1.960 * state_stds[i] for i in range(len(state_means))]
        values_upper = [state_means[i] + 1.960 * state_stds[i] for i in range(len(state_means))]

        smoother_states = self.save_plot(
            plot_times=prediction_times,
            plot_values=state_means,
            plot_values_lower=values_lower,
            plot_values_upper=values_upper,
            q=95,
            statistics_type=Statistics.MEAN,
            times_train=train_times,
            observations=observations,
            add_data=True,
            space=Space.STATE,
            all_initial_conditions=all_initial_conditions
        )
        data['smoother_states'] = smoother_states

        # Create smoother derivatives data

        derivative_stds = [jnp.sqrt(jnp.clip(x, a_min=0)) for x in derivative_variances]
        derivative_values_lower = [derivatives_means[i] - 1.960 * derivative_stds[i] for i in
                                   range(len(derivatives_means))]
        derivative_values_upper = [derivatives_means[i] + 1.960 * derivative_stds[i] for i in
                                   range(len(derivatives_means))]

        smoother_derivatives = self.save_plot(
            plot_times=prediction_times,
            plot_values=derivatives_means,
            plot_values_lower=derivative_values_lower,
            plot_values_upper=derivative_values_upper,
            q=95,
            statistics_type=Statistics.MEAN,
            times_train=train_times,
            observations=observations,
            add_data=False,
            space=Space.DERIVATIVE,
        )
        data['smoother_derivatives'] = smoother_derivatives

        # Create dynamics derivatives data

        dynamics_stds = [jnp.sqrt(jnp.clip(x, a_min=0)) for x in dynamics_variances]
        dynamics_values_lower = [dynamics_means[i] - 1.960 * dynamics_stds[i] for i in
                                 range(len(dynamics_means))]
        dynamics_values_upper = [dynamics_means[i] + 1.960 * dynamics_stds[i] for i in
                                 range(len(dynamics_means))]

        dynamics_derivatives = self.save_plot(
            plot_times=prediction_times,
            plot_values=dynamics_means,
            plot_values_lower=dynamics_values_lower,
            plot_values_upper=dynamics_values_upper,
            q=95,
            statistics_type=Statistics.MEAN,
            times_train=train_times,
            observations=observations,
            add_data=False,
            space=Space.DERIVATIVE,
        )
        data['dynamics_derivatives'] = dynamics_derivatives
        return data

    def save_plot(
            self,
            plot_times: List[jnp.array],  # jnp.array of shape (n, )
            plot_values: List[jnp.array],  # jnp.array of shape (n, num_dimension)
            plot_values_lower: List[jnp.array],  # jnp.array of shape (n, num_dimensions)
            plot_values_upper: List[jnp.array],  # jnp.array of shape (n, num_dimensions)
            q: float,
            statistics_type: Statistics,
            initial_conditions: Optional[List[jnp.array]] = None,
            times_train: Optional[List[jnp.array]] = None,  # jnp.array of shape (n, 1)
            observations: Optional[List[jnp.array]] = None,  # jnp.array of shape (n, num_dimensions)
            add_data: bool = False,
            space: Space = Space.STATE,
            all_initial_conditions: Optional[List[jnp.array]] = None
    ):
        data = dict()

        num_dimensions = plot_values[0].shape[1]
        num_trajectories = len(plot_times)
        if all_initial_conditions is not None:
            # We assume the time horizon is the same for every trajectory,
            # otherwise we need to do a lot of integration
            all_trajectories = []
            times = jnp.linspace(
                float(jnp.min(plot_times[0])),
                float(jnp.max(plot_times[0])),
                100,
            )
            for ic in all_initial_conditions:
                current_trajectory = self.simulator.simulate_trajectory(
                    initial_condition=ic.reshape(-1),
                    times=times,
                    sigma=None,
                    key=None
                )[0]
                all_trajectories.append(current_trajectory)

        if initial_conditions is None:
            initial_conditions = self.initial_conditions
        for trajectory in range(num_trajectories):
            times_for_true_points = jnp.linspace(
                float(jnp.min(plot_times[trajectory])),
                float(jnp.max(plot_times[trajectory])),
                100,
            )
            values_for_true_points = self.simulator.simulate_trajectory(
                initial_condition=initial_conditions[trajectory].reshape(-1),
                times=times_for_true_points,
                sigma=None,
                key=None
            )[0]
            if space == Space.DERIVATIVE:
                values_for_true_points = self.simulator.get_derivatives(
                    values_for_true_points
                )
            for dimension in range(num_dimensions):
                values = plot_values[trajectory][:, dimension]
                values_lower = plot_values_lower[trajectory][:, dimension]
                values_upper = plot_values_upper[trajectory][:, dimension]

                plot_name = 'trajectory {}, dimension {}'.format(trajectory, dimension)
                plot_dict = {}
                plot_dict['plot_times'] = plot_times[trajectory]
                plot_dict['values'] = values
                plot_dict['plot_values_lower'] = values_lower
                plot_dict['plot_values_upper'] = values_upper
                plot_dict['q'] = q
                plot_dict['statistics_type'] = statistics_type
                plot_dict['times_from_train_dataset'] = times_train[trajectory] if add_data else None
                plot_dict['observations'] = observations[trajectory][:, dimension] if add_data else None
                plot_dict['statistics_type'] = statistics_type
                if all_initial_conditions is not None:
                    plot_dict['ground_truth_all_times'] = times
                    plot_dict['ground_truth_all_values'] = [traj[:, dimension] for traj in all_trajectories]
                plot_dict['ground_truth_times'] = times_for_true_points
                plot_dict['ground_truth_values'] = values_for_true_points[:, dimension]
                data[plot_name] = plot_dict
        return data

    def save_plot_predicted_trajectory(self,
                                       initial_condition,
                                       test_times,
                                       state_means,
                                       state_variances,
                                       derivative_means,
                                       derivative_variances,
                                       dynamics_means,
                                       dynamics_variances,
                                       all_initial_conditions: Optional[List[jnp.array]] = None):
        num_dimensions = state_means[0].shape[1]

        data = dict()

        state_stds = [jnp.sqrt(jnp.clip(x, a_min=0)) for x in state_variances]
        values_lower = [state_means[i] - 1.960 * state_stds[i] for i in range(len(state_means))]
        values_upper = [state_means[i] + 1.960 * state_stds[i] for i in range(len(state_means))]

        smoother_states = self.save_plot(
            plot_times=test_times,
            plot_values=state_means,
            plot_values_lower=values_lower,
            plot_values_upper=values_upper,
            q=95,
            statistics_type=Statistics.MEAN,
            initial_conditions=initial_condition,
            space=Space.STATE,
            all_initial_conditions=all_initial_conditions
        )
        data['smoother_states'] = smoother_states

        derivative_stds = [jnp.sqrt(jnp.clip(x, a_min=0)) for x in derivative_variances]
        derivative_values_lower = [derivative_means[i] - 1.960 * derivative_stds[i] for i in
                                   range(len(derivative_means))]
        derivative_values_upper = [derivative_means[i] + 1.960 * derivative_stds[i] for i in
                                   range(len(derivative_means))]

        smoother_derivatives = self.save_plot(
            plot_times=test_times,
            plot_values=derivative_means,
            plot_values_lower=derivative_values_lower,
            plot_values_upper=derivative_values_upper,
            q=95,
            statistics_type=Statistics.MEAN,
            initial_conditions=initial_condition,
            space=Space.DERIVATIVE
        )
        data['smoother_derivatives'] = smoother_derivatives

        dynamics_stds = [jnp.sqrt(jnp.clip(x, a_min=0)) for x in dynamics_variances]
        dynamics_values_lower = [dynamics_means[i] - 1.960 * dynamics_stds[i] for i in
                                 range(len(dynamics_means))]
        dynamics_values_upper = [dynamics_means[i] + 1.960 * dynamics_stds[i] for i in
                                 range(len(dynamics_means))]

        dynamics_derivatives = self.save_plot(
            plot_times=test_times,
            plot_values=dynamics_means,
            plot_values_lower=dynamics_values_lower,
            plot_values_upper=dynamics_values_upper,
            q=95,
            statistics_type=Statistics.MEAN,
            initial_conditions=initial_condition,
            space=Space.DERIVATIVE
        )
        data['dynamics_derivatives'] = dynamics_derivatives
        return data

    def plot_predicted_trajectory(self,
                                  initial_condition,
                                  test_times,
                                  state_means,
                                  state_variances,
                                  derivative_means,
                                  derivative_variances,
                                  dynamics_means,
                                  dynamics_variances,
                                  all_initial_conditions: Optional[List[jnp.array]] = None):
        num_dimensions = state_means[0].shape[1]
        figure, ax = plt.subplots(3, num_dimensions, figsize=(15, 18))
        ax = ax.reshape(3, num_dimensions)
        figure.suptitle("Prediction for initial condition {}".format(initial_condition), fontsize=20)

        state_stds = [jnp.sqrt(jnp.clip(x, a_min=0)) for x in state_variances]
        values_lower = [state_means[i] - 1.960 * state_stds[i] for i in range(len(state_means))]
        values_upper = [state_means[i] + 1.960 * state_stds[i] for i in range(len(state_means))]

        self._create_plot(
            axs=ax[0, :].reshape(1, -1),
            plot_times=test_times,
            plot_values=state_means,
            plot_values_lower=values_lower,
            plot_values_upper=values_upper,
            q=95,
            statistics_type=Statistics.MEAN,
            initial_conditions=initial_condition,
            space=Space.STATE,
            all_initial_conditions=all_initial_conditions
        )

        derivative_stds = [jnp.sqrt(jnp.clip(x, a_min=0)) for x in derivative_variances]
        derivative_values_lower = [derivative_means[i] - 1.960 * derivative_stds[i] for i in
                                   range(len(derivative_means))]
        derivative_values_upper = [derivative_means[i] + 1.960 * derivative_stds[i] for i in
                                   range(len(derivative_means))]

        self._create_plot(
            axs=ax[1, :].reshape(1, -1),
            plot_times=test_times,
            plot_values=derivative_means,
            plot_values_lower=derivative_values_lower,
            plot_values_upper=derivative_values_upper,
            q=95,
            statistics_type=Statistics.MEAN,
            initial_conditions=initial_condition,
            space=Space.DERIVATIVE
        )

        dynamics_stds = [jnp.sqrt(jnp.clip(x, a_min=0)) for x in dynamics_variances]
        dynamics_values_lower = [dynamics_means[i] - 1.960 * dynamics_stds[i] for i in
                                 range(len(dynamics_means))]
        dynamics_values_upper = [dynamics_means[i] + 1.960 * dynamics_stds[i] for i in
                                 range(len(dynamics_means))]
        self._create_plot(
            axs=ax[2, :].reshape(1, -1),
            plot_times=test_times,
            plot_values=dynamics_means,
            plot_values_lower=dynamics_values_lower,
            plot_values_upper=dynamics_values_upper,
            q=95,
            statistics_type=Statistics.MEAN,
            initial_conditions=initial_condition,
            space=Space.DERIVATIVE
        )
        return figure

    def _create_plot(
            self,
            axs,
            plot_times: List[jnp.array],  # jnp.array of shape (n, )
            plot_values: List[jnp.array],  # jnp.array of shape (n, num_dimension)
            plot_values_lower: List[jnp.array],  # jnp.array of shape (n, num_dimensions)
            plot_values_upper: List[jnp.array],  # jnp.array of shape (n, num_dimensions)
            q: float,
            statistics_type: Statistics,
            initial_conditions: Optional[List[jnp.array]] = None,
            times_train: Optional[List[jnp.array]] = None,  # jnp.array of shape (n, 1)
            observations: Optional[List[jnp.array]] = None,  # jnp.array of shape (n, num_dimensions)
            add_data: bool = False,
            space: Space = Space.STATE,
            all_initial_conditions: Optional[List[jnp.array]] = None
    ):
        num_dimensions = plot_values[0].shape[1]
        num_trajectories = len(plot_times)
        if all_initial_conditions is not None:
            # We assume the time horizon is the same for every trajectory,
            # otherwise we need to do a lot of integration
            all_trajectories = []
            times = jnp.linspace(
                float(jnp.min(plot_times[0])),
                float(jnp.max(plot_times[0])),
                100,
            )
            for ic in all_initial_conditions:
                current_trajectory = self.simulator.simulate_trajectory(
                    initial_condition=ic.reshape(-1),
                    times=times,
                    sigma=None,
                    key=None
                )[0]
                all_trajectories.append(current_trajectory)
        if initial_conditions is None:
            initial_conditions = self.initial_conditions

        for trajectory in range(num_trajectories):
            print('Starting with trajectory {}'.format(trajectory))
            times_for_true_points = jnp.linspace(
                float(jnp.min(plot_times[trajectory])),
                float(jnp.max(plot_times[trajectory])),
                100,
            )
            values_for_true_points = self.simulator.simulate_trajectory(
                initial_condition=initial_conditions[trajectory].reshape(-1),
                times=times_for_true_points,
                sigma=None,
                key=None
            )[0]
            if space == Space.DERIVATIVE:
                values_for_true_points = self.simulator.get_derivatives(
                    values_for_true_points
                )
            for dimension in range(num_dimensions):
                print('Dimension {}'.format(dimension))
                values = plot_values[trajectory][:, dimension]
                values_lower = plot_values_lower[trajectory][:, dimension]
                values_upper = plot_values_upper[trajectory][:, dimension]

                ax = axs[trajectory, dimension]
                self._add_prediction_on_plot(
                    ax=ax,
                    plot_times=plot_times[trajectory],
                    plot_values=values,
                    plot_values_lower=values_lower,
                    plot_values_upper=values_upper,
                    q=q,
                    statistics_type=statistics_type,
                    times_from_train_dataset=times_train[trajectory] if add_data else None,
                    observations=observations[trajectory][:, dimension] if add_data else None,
                    add_data=add_data,
                )
                if all_initial_conditions is not None:
                    self.add_ground_truth_trajectories(ax, times, [traj[:, dimension] for traj in all_trajectories])
                ax.set_xlabel(r"t")
                ax.set_ylabel(
                    r"$x_{}(t)$".format('{' + str(dimension) + '}')
                    if space == Space.STATE
                    else r"$\dot x_{}(t)$".format('{' + str(dimension) + '}')
                )
                ax.plot(
                    times_for_true_points,
                    values_for_true_points[:, dimension],
                    label=r"True $x_{}$".format('{' + str(dimension) + '}')
                    if space == Space.STATE
                    else r"True $\dot x_{}$".format('{' + str(dimension) + '}'),
                    color="black",
                )
                if trajectory == 0 and dimension == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys())
                    # ax.legend()
        # Add description for which dimension and trajectory we are on
        cols = ["Dimension {}".format(col) for col in range(num_dimensions)]
        rows = ["Trajectory {}".format(row) for row in range(num_trajectories)]
        pad = 5
        if num_dimensions >= 2:
            for ax, col in zip(axs[0], cols):
                ax.annotate(
                    col,
                    xy=(0.5, 1),
                    xytext=(0, pad),
                    xycoords="axes fraction",
                    textcoords="offset points",
                    size="large",
                    ha="center",
                    va="baseline",
                )
        if num_trajectories >= 2:
            for ax, row in zip(axs[:, 0], rows):
                ax.annotate(
                    row,
                    xy=(0, 0.5),
                    xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label,
                    textcoords="offset points",
                    size="large",
                    ha="right",
                    va="center",
                )

    @staticmethod
    def add_ground_truth_trajectories(ax, times, values):
        for trajectory in values:
            ax.plot(times, trajectory, color='gray', alpha=.2, label='Ground truth trajectories')

    @staticmethod
    def _add_prediction_on_plot(
            ax,
            plot_times: jnp.array,  # Of shape (n, )
            plot_values: jnp.array,  # Of shape (n, )
            plot_values_lower: jnp.array,  # Of shape (n, )
            plot_values_upper: jnp.array,  # Of shape (n, )
            q: float,
            statistics_type: Statistics,
            times_from_train_dataset: Optional[jnp.array] = None,  # Of shape (n, 1)
            observations: Optional[jnp.array] = None,  # Of shape (n, )
            add_data: bool = False,
    ):
        if add_data:
            ax.plot(
                times_from_train_dataset,
                observations,
                "r.",
                markersize=10,
                label="Observations",
            )
        if statistics_type == Statistics.MEDIAN:
            label = "Between {:.2f} and {:.2f} quantile".format(1 - q, q)
        elif statistics_type == Statistics.MEAN:
            label = "{}% confidence interval".format(q)
        ax.plot(plot_times, plot_values, "b-", label="Prediction")
        ax.fill_between(plot_times, plot_values_lower, plot_values_upper, alpha=0.5, fc="b", ec="None", label=label)

    def plot_learned_vector_field(
            self,
            initial_conditions: List[jnp.array],
            observations: List[jnp.array],
            grid,
            true_vector_field,
            learned_vector_field_mean,
            volume_covariance_learned,
            max_covariance_eigenvalue,
    ):
        fig, axs = plt.subplots(2, 3, figsize=(12, 12))
        # fig.suptitle("Lotka Volterra", fontsize=16)
        x, y = grid
        u_true, v_true, norm_true = true_vector_field
        u_mean_learned, v_mean_learned, norm_mean_learned = learned_vector_field_mean

        true_contour = axs[0, 0].contourf(x, y, norm_true)
        fig.colorbar(true_contour, ax=axs[0, 0])
        axs[0, 0].quiver(x, y, u_true, v_true, cmap="coolwarm")
        axs[0, 0].set_title("True vector field")
        axs[0, 0].set_xlabel(r"$x_0$")
        axs[0, 0].set_ylabel(r"$x_1$")
        self.add_observations_and_initial_conditions(axs[0, 0], observations, initial_conditions)

        learned_contour = axs[0, 1].contourf(x, y, norm_mean_learned)
        fig.colorbar(learned_contour, ax=axs[0, 1])
        axs[0, 1].quiver(x, y, u_mean_learned, v_mean_learned, cmap="coolwarm")
        axs[0, 1].set_title("Learned vector field")
        axs[0, 1].set_xlabel(r"$x_0$")
        self.add_observations_and_initial_conditions(axs[0, 1], observations, initial_conditions)

        norm_diff = jnp.sqrt((u_mean_learned - u_true) ** 2 + (v_mean_learned - v_true) ** 2)
        diff_contour = axs[0, 2].contourf(x, y, norm_diff)
        fig.colorbar(diff_contour, ax=axs[0, 2])
        axs[0, 2].quiver(x, y, u_mean_learned - u_true, v_mean_learned - v_true, cmap="coolwarm")
        axs[0, 2].set_title("Absolute Difference Learned-True vector field")
        axs[0, 2].set_xlabel(r"$x_0$")
        self.add_observations_and_initial_conditions(axs[0, 2], observations, initial_conditions)

        norm_diff = jnp.sqrt((u_mean_learned - u_true) ** 2 + (v_mean_learned - v_true) ** 2)
        norm_diff_relative = norm_diff / norm_true
        diff_contour = axs[1, 0].contourf(x, y, norm_diff_relative)
        fig.colorbar(diff_contour, ax=axs[1, 0])
        axs[1, 0].quiver(x, y, u_mean_learned - u_true, v_mean_learned - v_true, cmap="coolwarm")
        axs[1, 0].set_title("Relative Difference Learned-True vector field")
        axs[1, 0].set_ylabel(r"$x_1$")
        axs[1, 0].set_xlabel(r"$x_0$")

        diff_contour = axs[1, 1].contourf(x, y, volume_covariance_learned)
        fig.colorbar(diff_contour, ax=axs[1, 1])
        axs[1, 1].quiver(x, y, u_mean_learned - u_true, v_mean_learned - v_true, cmap="coolwarm")
        axs[1, 1].set_title("Volume of 95% confidence ellipsoid")
        axs[1, 1].set_xlabel(r"$x_0$")

        diff_contour = axs[1, 2].contourf(x, y, max_covariance_eigenvalue)
        fig.colorbar(diff_contour, ax=axs[1, 2])
        axs[1, 2].quiver(x, y, u_mean_learned - u_true, v_mean_learned - v_true, cmap="coolwarm")
        axs[1, 2].set_title("Largest sqrt eigenvalue of covariance matrix")
        axs[1, 2].set_xlabel(r"$x_0$")
        return fig

    @staticmethod
    def add_observations_and_initial_conditions(ax, observations, initial_conditions):
        for i in range(len(initial_conditions)):
            ax.plot(
                observations[i][:, 0],
                observations[i][:, 1],
                "r.",
                markersize=10,
                color="red",
            )
            ax.plot(
                initial_conditions[i][0],
                initial_conditions[i][1],
                "r.",
                markersize=15,
                color="black",
            )

    def plot_sample_trajectories(self, initial_condition, times, values, lower_values, upper_values, q,
                                 all_initial_conditions: Optional[List[jnp.array]] = None):
        num_dimensions = values.shape[1]

        figure, ax = plt.subplots(1, num_dimensions, figsize=(15, 6))
        ax = ax.reshape(1, num_dimensions)
        figure.suptitle("Bayesian integration", fontsize=20)
        self._create_plot(
            axs=ax[0, :].reshape(1, -1),
            plot_times=[times],
            plot_values=[values],
            plot_values_lower=[lower_values],
            plot_values_upper=[upper_values],
            statistics_type=Statistics.MEDIAN,
            q=q,
            initial_conditions=[initial_condition],
            space=Space.STATE,
            all_initial_conditions=all_initial_conditions
        )
        return figure
