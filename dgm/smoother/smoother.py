from typing import Optional, Any, Tuple, Dict

import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from jax.scipy.linalg import block_diag

from dgm.smoother.deep_approx_kernel import get_deep_kernel_feature_creator
from dgm.smoother.deep_kernel import get_deep_kernel
from dgm.smoother.feature_extractors import get_time_and_states_to_features_extractor
from dgm.smoother.gp_means import get_composite_gp_mean
from dgm.utils.helper_functions import make_positive, diag_product, negative_log_likelihood_normal
from dgm.utils.representatives import KernelType, FeaturesToFeaturesType, TimeAndStatesToFeaturesType

pytree = Any


def get_smoother(
        kernel: KernelType, kernel_kwargs: Dict,
        core_type: TimeAndStatesToFeaturesType, core_kwargs: Dict,
        mean_head_type: FeaturesToFeaturesType, mean_head_kwargs: Dict,
        kernel_core_type: FeaturesToFeaturesType, kernel_core_kwargs: Dict,
        kernel_head_type: FeaturesToFeaturesType, kernel_head_kwargs: Dict,
        n_dim: int, numerical_correction: float = 1e-3,
):
    core_init, _, get_regularization_core = get_time_and_states_to_features_extractor(core_type, core_kwargs)
    mean_init, mean_apply, get_regularization_mean = get_composite_gp_mean(core_type=core_type,
                                                                           core_kwargs=core_kwargs,
                                                                           mean_head_type=mean_head_type,
                                                                           mean_head_kwargs=mean_head_kwargs)
    smoother_init, smoother_apply, get_means_and_covariances_test, \
    filter_smoother_params, get_regularization_smoother = joint_smoother(kernel=kernel,
                                                                         kernel_kwargs=kernel_kwargs,
                                                                         core_type=core_type,
                                                                         core_kwargs=core_kwargs,
                                                                         kernel_core_type=kernel_core_type,
                                                                         kernel_core_kwargs=kernel_core_kwargs,
                                                                         kernel_head_type=kernel_head_type,
                                                                         kernel_head_kwargs=kernel_head_kwargs,
                                                                         numerical_correction=numerical_correction,
                                                                         )
    d_mean_apply = jit(jax.jacrev(mean_apply, 0))
    vec_d_mean_apply = vector_and_matrix_function(d_mean_apply, in_axes=(0, 0, None, None), static_argnums=())[0]

    def multi_d_smoother_init(rng, n_output_dimensions):
        all_params = dict()
        smoother_rng, mean_rng = jax.random.split(rng, 2)
        all_params["smoother"] = smoother_init(smoother_rng, n_output_dimensions)[1]
        all_params["mean"] = mean_init(mean_rng, n_output_dimensions)[1]
        if core_type:
            all_params["core"] = core_init(rng, (-1, n_dim + 1))[1]
        else:
            all_params["core"] = None
        return all_params

    @jax.jit
    def multi_d_smoother_apply(
            observation_times: jnp.array,  # shape n_obs x 1
            matching_times: jnp.array,  # shape n_deriv_obs x 1
            ic_for_observation_times: jnp.array,  # shape n_obs x n_dim
            ic_for_matching_times: jnp.array,  # shape n_deriv_obs x n_dim
            observations: jnp.array,  # shape n_obs x n_dim
            parameters: pytree,  # type jax.pytree
    ) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array, jnp.array]:

        prior_means = mean_apply(observation_times, ic_for_observation_times, parameters["mean"], parameters["core"])
        prior_means_for_derivatives = mean_apply(matching_times, ic_for_matching_times,
                                                 parameters["mean"], parameters["core"])
        prior_derivative_means = jnp.squeeze(vec_d_mean_apply(matching_times.reshape(-1),
                                                              ic_for_matching_times,
                                                              parameters["mean"], parameters["core"]))

        def apply_smoother_in_dimension(dimension):
            return smoother_apply(
                observation_times,
                matching_times,
                ic_for_observation_times,
                ic_for_matching_times,
                observations[:, dimension],
                filter_smoother_params(parameters["smoother"], dimension),
                prior_means[:, dimension],
                prior_means_for_derivatives[:, dimension],
                prior_derivative_means[:, dimension],
                core_parameters=parameters['core']
            )

        apply_smoother_all = jit(vmap(jit(apply_smoother_in_dimension), out_axes=(1, 1, 1, 0)))
        return apply_smoother_all(jnp.arange(n_dim))

    @jax.jit
    def multi_d_smoother_posterior(
            evaluation_times: jnp.array,  # List over trajectories, 1D vectors (n_dim x 1)
            observation_times: jnp.array,  # List over trajectories, 1D vectors (n_dim x 1)
            ic_for_evaluation_times: jnp.array,  # List over trajectories, 1D arrays  (n_obs x n_dim)
            ic_for_observation_times: jnp.array,  # List over trajectories, 1D arrays  (n_obs x n_dim)
            observations: jnp.array,  # List over trajectories, 2D arrays (n_obs x n_dim)
            parameters: pytree,  # type jax.pytree
    ):
        normal_prior_means = mean_apply(observation_times, ic_for_observation_times, parameters["mean"], parameters["core"])
        test_prior_means = mean_apply(evaluation_times, ic_for_evaluation_times, parameters["mean"], parameters["core"])
        derivative_prior_means = jnp.squeeze(
            vec_d_mean_apply(evaluation_times, ic_for_evaluation_times, parameters["mean"], parameters["core"]))

        def apply_means_and_covariances_in_dimension(dimension):
            return get_means_and_covariances_test(
                evaluation_times,
                observation_times,
                ic_for_evaluation_times,
                ic_for_observation_times,
                observations[:, dimension],
                filter_smoother_params(parameters["smoother"], dimension),
                normal_prior_means[:, dimension],
                test_prior_means[:, dimension],
                derivative_prior_means[:, dimension],
                core_parameters=parameters['core'])

        apply_means_and_covariances_all = jit(
            vmap(jit(apply_means_and_covariances_in_dimension), out_axes=(1, 1, 1, 1)))
        return apply_means_and_covariances_all(jnp.arange(n_dim))

    @jax.jit
    def multi_d_regularization(parameters, weights):
        objective = get_regularization_core(parameters['core'], weights)
        objective += get_regularization_mean(parameters['mean'], weights)
        objective += get_regularization_smoother(parameters['smoother'], weights)
        return objective

    return multi_d_smoother_init, multi_d_smoother_apply, multi_d_smoother_posterior, multi_d_regularization


def joint_smoother(kernel: KernelType, kernel_kwargs: Dict,
                   core_type: TimeAndStatesToFeaturesType, core_kwargs: Dict,
                   kernel_core_type: FeaturesToFeaturesType, kernel_core_kwargs: Dict,
                   kernel_head_type: FeaturesToFeaturesType, kernel_head_kwargs: Dict,
                   numerical_correction: Optional[float] = 1e-3,
                   ):
    if kernel == KernelType.RBF_RFF or kernel == KernelType.LINEAR:
        return approximate_joint_smoother(
            kernel,
            kernel_kwargs,
            core_type,
            core_kwargs,
            kernel_core_type,
            kernel_core_kwargs,
            kernel_head_type,
            kernel_head_kwargs,
        )

    dk_m, dif_dk_m, dif_dk_dif_v, dif_dk_dif_m = _get_kernel_functions(kernel, kernel_kwargs,
                                                                       core_type, core_kwargs,
                                                                       kernel_core_type, kernel_core_kwargs,
                                                                       kernel_head_type, kernel_head_kwargs)

    @jax.jit
    def smoother_apply(
            observation_times: jnp.array,
            matching_times: jnp.array,
            ic_for_observation_times: jnp.array,
            ic_for_matching_times: jnp.array,
            observations: jnp.array,
            filtered_params: pytree,
            prior_mean_for_observation_times: jnp.array,
            prior_means_for_matching_times: jnp.array,
            prior_derivative_means_for_matching_times: jnp.array,
            core_parameters: pytree
    ) -> Tuple[jnp.array, jnp.array, jnp.array, float]:

        observations = observations.reshape(-1, 1)
        prior_mean_for_observation_times = prior_mean_for_observation_times.reshape(-1, 1)
        prior_means_for_matching_times = prior_means_for_matching_times.reshape(-1, 1)
        prior_derivative_means_for_matching_times = prior_derivative_means_for_matching_times.reshape(-1, 1)

        num_points_derivatives = matching_times.size
        num_points = observation_times.size

        first_indices, second_indices = jnp.meshgrid(jnp.arange(num_points), jnp.arange(num_points))
        covariance_matrix = dk_m(observation_times[first_indices], ic_for_observation_times[first_indices, :],
                                 observation_times[second_indices], ic_for_observation_times[second_indices, :],
                                 filtered_params["kernel"], core_parameters)

        noise_variance = make_positive(filtered_params["noise_std"]) ** 2
        noisy_covariance_matrix = covariance_matrix + noise_variance * jnp.eye(num_points)

        cholesky_tuple = jax.scipy.linalg.cho_factor(noisy_covariance_matrix)
        cholesky_eigenvals = jnp.diag(cholesky_tuple[0])
        log_eigenvals = jnp.log(cholesky_eigenvals)
        log_determinant = 2 * jnp.sum(log_eigenvals)

        denoised_obs = jax.scipy.linalg.cho_solve(cholesky_tuple, observations - prior_mean_for_observation_times)
        obs_term = (observations - prior_mean_for_observation_times).reshape(-1) @ denoised_obs.reshape(-1)
        first_indices_cross, second_indices_cross = jnp.meshgrid(jnp.arange(num_points_derivatives),
                                                                 jnp.arange(num_points))
        smoothed_mean = prior_means_for_matching_times + dk_m(
            matching_times[first_indices_cross],
            ic_for_matching_times[first_indices_cross, :],
            observation_times[second_indices_cross],
            ic_for_observation_times[second_indices_cross, :],
            filtered_params["kernel"],
            core_parameters) @ denoised_obs

        d_k = dif_dk_m(
            matching_times[first_indices_cross], ic_for_matching_times[first_indices_cross, :],
            observation_times[second_indices_cross], ic_for_observation_times[second_indices_cross, :],
            filtered_params["kernel"], core_parameters
        )
        k_d = d_k.T

        derivative_covariance = jax.scipy.linalg.cho_solve(cholesky_tuple, k_d)
        derivative_covariance_second_term = diag_product(d_k, derivative_covariance)
        derivative_covariance_first_term = dif_dk_dif_v(matching_times, ic_for_matching_times,
                                                        matching_times, ic_for_matching_times,
                                                        filtered_params["kernel"], core_parameters)
        vectorized_derivative_covariance = derivative_covariance_first_term - derivative_covariance_second_term

        derivative_mean = prior_derivative_means_for_matching_times + d_k @ denoised_obs
        derivative_covariance = vectorized_derivative_covariance
        return smoothed_mean.reshape(-1), derivative_mean.reshape(-1), derivative_covariance, log_determinant + obs_term

    kernel_init, _, filter_kernel_params, get_kernel_regularization = get_deep_kernel(kernel, kernel_kwargs,
                                                                                      core_type,
                                                                                      core_kwargs,
                                                                                      kernel_core_type,
                                                                                      kernel_core_kwargs,
                                                                                      kernel_head_type,
                                                                                      kernel_head_kwargs)

    def filter_joint_smoother_params(all_params, dim_id):
        return {
            "noise_std": all_params["noise_std"][dim_id],
            "kernel": filter_kernel_params(all_params["kernel"], dim_id)
        }

    def smoother_init(rng, num_dim):
        parameters = dict()
        parameters['noise_std'] = -2 * jnp.ones(num_dim)

        n_features = num_dim + 1
        structure, kernel_params = kernel_init(rng, n_features, num_dim)
        parameters["kernel"] = kernel_params
        return structure, parameters

    @jax.jit
    def get_smoother_regularization(parameters, weights):
        num_dim = len(parameters.keys()) - 1
        objective = get_kernel_regularization(parameters['kernel'], weights)
        for dim_id in range(num_dim):
            objective += weights["obs_noise"] * negative_log_likelihood_normal(
                parameters["noise_std"][dim_id],
                mu=-2.25,  # TODO: make more flexible, i.e. via dict call from weights
                sigma=2
            )
        return objective

    @jax.jit
    def smoother_posterior(
            evaluation_times: jnp.array,  # List over trajectories, 1D vectors (n_dim x 1)
            observation_times: jnp.array,  # List over trajectories, 1D vectors (n_dim x 1)
            ic_for_evaluation_times: jnp.array,  # List over trajectories, 1D arrays  (1 x n_dim)
            ic_for_observation_times: jnp.array,  # List over trajectories, 1D arrays  (1 x n_dim)
            observations: jnp.array,  # List over trajectories, 2D arrays (n_obs x n_dim)
            filtered_params: pytree,  # type jax.pytree
            prior_means_for_observation_times: jnp.array,  # List over trajectories, 2D arrays (n_obs x n_dim)
            prior_means_for_evaluation_times: jnp.array,  # List over trajectories, 2D arrays (n_obs x n_dim)
            prior_derivative_means_for_evaluation_times: jnp.array,
            core_parameters: pytree,
    ):
        joint_times = jnp.concatenate([observation_times, evaluation_times], axis=0)
        num_joint_points = len(joint_times)
        num_observations = len(observation_times)

        joint_initial_conditions = jnp.concatenate([ic_for_observation_times, ic_for_evaluation_times], axis=0)

        first_indices_joint, second_indices_joint = jnp.meshgrid(jnp.arange(num_joint_points),
                                                                 jnp.arange(num_joint_points))
        first_indices_train, second_indices_train = jnp.meshgrid(jnp.arange(num_observations),
                                                                 jnp.arange(num_observations))

        sigma = filtered_params["noise_std"]
        joint_covariance = dk_m(joint_times[first_indices_joint], joint_initial_conditions[first_indices_joint, :],
                                joint_times[second_indices_joint], joint_initial_conditions[second_indices_joint, :],
                                filtered_params["kernel"], core_parameters)
        joint_diff_covariance = dif_dk_m(joint_times[first_indices_joint],
                                         joint_initial_conditions[first_indices_joint, :],
                                         joint_times[second_indices_joint],
                                         joint_initial_conditions[second_indices_joint, :],
                                         filtered_params["kernel"], core_parameters)
        test_diff_covariance_diff = dif_dk_dif_v(evaluation_times, ic_for_evaluation_times,
                                                 evaluation_times, ic_for_evaluation_times,
                                                 filtered_params["kernel"], core_parameters)
        observation_covariance = dk_m(observation_times[first_indices_train],
                                      ic_for_observation_times[first_indices_train, :],
                                      observation_times[second_indices_train],
                                      ic_for_observation_times[second_indices_train, :],
                                      filtered_params["kernel"], core_parameters)

        noise_variance = make_positive(sigma) ** 2
        observation_covariance = observation_covariance + noise_variance * jnp.eye(num_observations)

        observation_covariance_cholesky = jax.scipy.linalg.cho_factor(observation_covariance)
        covariance_test_train = joint_covariance[num_observations:, :num_observations]
        covariance_test_test = joint_covariance[num_observations:, num_observations:]
        diff_covariance_test_train_diff = joint_diff_covariance[num_observations:, :num_observations]

        noisy_covariance_times_observations = jax.scipy.linalg.cho_solve(
            observation_covariance_cholesky, observations.reshape([-1, 1]) - prior_means_for_observation_times.reshape(-1, 1)
        )

        # Compute mean of states
        state_means = prior_means_for_evaluation_times.reshape(-1, 1) + covariance_test_train @ noisy_covariance_times_observations

        # Compute mean of derivatives
        derivative_means = prior_derivative_means_for_evaluation_times.reshape(-1, 1) + \
                           diff_covariance_test_train_diff @ noisy_covariance_times_observations

        # Compute variances of states
        second_term = jax.scipy.linalg.cho_solve(observation_covariance_cholesky, covariance_test_train.T)
        second_term = covariance_test_train @ second_term
        state_variances = jnp.diag(covariance_test_test - second_term)

        # Compute variances of derivatives
        second_term = jax.scipy.linalg.cho_solve(observation_covariance_cholesky, diff_covariance_test_train_diff.T)
        second_term = diag_product(diff_covariance_test_train_diff, second_term)
        derivative_variances = test_diff_covariance_diff - second_term

        return state_means.reshape(-1), state_variances, derivative_means.reshape(-1), derivative_variances

    return smoother_init, smoother_apply, smoother_posterior, filter_joint_smoother_params, \
           get_smoother_regularization


def _get_kernel_functions(kernel: KernelType, kernel_kwargs,
                          core_type: TimeAndStatesToFeaturesType, core_kwargs: Dict,
                          kernel_core_type: FeaturesToFeaturesType, kernel_core_kwargs: Dict,
                          kernel_head_type: FeaturesToFeaturesType, kernel_head_kwargs: Dict):
    kernel_apply = get_deep_kernel(kernel, kernel_kwargs, core_type, core_kwargs,
                                   kernel_core_type, kernel_core_kwargs, kernel_head_type, kernel_head_kwargs)[1]

    _dif_dk = grad(kernel_apply, 0)
    _dif_dk_dif = grad(grad(kernel_apply, 0), 2)

    dk_m = vector_and_matrix_function(kernel_apply, static_argnums=())[1]
    dif_dk_m = vector_and_matrix_function(_dif_dk, static_argnums=())[1]
    dif_dk_dif_v, dif_dk_dif_m = vector_and_matrix_function(_dif_dk_dif, static_argnums=())

    return dk_m, dif_dk_m, dif_dk_dif_v, dif_dk_dif_m


def vector_and_matrix_function(fun, in_axes=(0, 0, 0, 0, None, None), static_argnums=()):
    fun_v = jit(vmap(fun, in_axes=in_axes, out_axes=0), static_argnums=static_argnums)
    fun_m = jit(vmap(fun_v, in_axes=in_axes, out_axes=1), static_argnums=static_argnums)
    return fun_v, fun_m


def _get_feature_map_functions(kernel, kernel_kwargs,
                               neural_core_type, neural_core_kwargs,
                               kernel_core_type, kernel_core_kwargs,
                               kernel_head_type, kernel_head_kwargs):
    _, feature_creator_apply, _, _ = \
        get_deep_kernel_feature_creator(
            kernel_feature_creator_type=kernel,
            kernel_feature_creator_kwargs=kernel_kwargs,
            core_type=neural_core_type,
            core_kwargs=neural_core_kwargs,
            kernel_core_type=kernel_core_type,
            kernel_core_kwargs=kernel_core_kwargs,
            kernel_head_type=kernel_head_type,
            kernel_head_kwargs=kernel_head_kwargs
        )

    get_phi_transposed = jit(vmap(feature_creator_apply, in_axes=(0, 0, None, None)), static_argnums=())

    d_kernel = jax.jacrev(feature_creator_apply, argnums=0)
    get_d_phi_transposed = jit(vmap(jit(d_kernel), in_axes=(0, 0, None, None)), static_argnums=())

    return get_phi_transposed, get_d_phi_transposed


def _get_diagonal_elements_of_matrix_multiplication_calculator():
    def dot_product(u, v):
        return u.reshape(1, -1) @ v.reshape(-1, 1)

    return jit(vmap(dot_product, in_axes=(0, 1)))


def approximate_joint_smoother(kernel: KernelType, kernel_kwargs: Dict,
                               neural_core_type: TimeAndStatesToFeaturesType, neural_core_kwargs: Dict,
                               kernel_core_type: FeaturesToFeaturesType, kernel_core_kwargs: Dict,
                               kernel_head_type: FeaturesToFeaturesType, kernel_head_kwargs: Dict):
    approx_kernel_init, _, filter_approx_kernel_params, get_approx_kernel_regularization = \
        get_deep_kernel_feature_creator(
            kernel_feature_creator_type=kernel, kernel_feature_creator_kwargs=kernel_kwargs,
            core_type=neural_core_type, core_kwargs=neural_core_kwargs,
            kernel_core_type=kernel_core_type, kernel_core_kwargs=kernel_core_kwargs,
            kernel_head_type=kernel_head_type, kernel_head_kwargs=kernel_head_kwargs)

    get_phi_transposed, get_d_phi_transposed = _get_feature_map_functions(
        kernel, kernel_kwargs,
        neural_core_type, neural_core_kwargs,
        kernel_core_type, kernel_core_kwargs,
        kernel_head_type, kernel_head_kwargs
    )

    diagonal_element_calculator = _get_diagonal_elements_of_matrix_multiplication_calculator()

    @jax.jit
    def smoother_apply(
            observation_times: jnp.array,
            matching_times: jnp.array,
            ic_for_observation_times: jnp.array,
            ic_for_matching_times: jnp.array,
            observations: jnp.array,
            filtered_params,
            prior_means_for_observation_times: jnp.array,
            prior_means_for_matching_times: jnp.array,
            prior_derivative_means_for_matching_times: jnp.array,
            core_parameters
    ):
        """ get all feature matrices """
        phi_train_times = get_phi_transposed(observation_times,
                                             ic_for_observation_times,
                                             filtered_params["kernel"],
                                             core_parameters).T
        phi_train_times = phi_train_times / jnp.sqrt(phi_train_times.shape[0])
        phi_deriv_times = get_phi_transposed(matching_times,
                                             ic_for_matching_times,
                                             filtered_params["kernel"],
                                             core_parameters).T
        phi_deriv_times = phi_deriv_times / jnp.sqrt(phi_deriv_times.shape[0])
        diff_phi_deriv_times = get_d_phi_transposed(matching_times.reshape(-1),
                                                    ic_for_matching_times,
                                                    filtered_params["kernel"],
                                                    core_parameters).T
        diff_phi_deriv_times = diff_phi_deriv_times / jnp.sqrt(diff_phi_deriv_times.shape[0])

        n_features, n_train_times = phi_train_times.shape

        """ get all observation related terms """
        observations = observations.reshape(-1, 1)

        sigma = filtered_params["noise_std"]
        noise_variance = make_positive(sigma) ** 2

        phi_times_phi_transposed = phi_train_times @ phi_train_times.T
        small_invertible_matrix = phi_times_phi_transposed + noise_variance * jnp.eye(n_features)
        small_matrix_cholesky_tuple = jax.scipy.linalg.cho_factor(small_invertible_matrix)

        denoised_obs = phi_train_times @ (observations - prior_means_for_observation_times.reshape([-1, 1]))
        denoised_obs = jax.scipy.linalg.cho_solve(small_matrix_cholesky_tuple, denoised_obs)
        denoised_obs = phi_train_times.T @ denoised_obs
        denoised_obs = observations - prior_means_for_observation_times.reshape([-1, 1]) - denoised_obs
        denoised_obs = denoised_obs / noise_variance

        obs_term = (observations.reshape(-1) - prior_means_for_observation_times) @ denoised_obs.reshape(-1)

        phi_times_denoised_obs = phi_train_times @ denoised_obs

        smoothed_mean = prior_means_for_matching_times.reshape([-1, 1]) + \
                        phi_deriv_times.T @ phi_times_denoised_obs
        derivative_mean = prior_derivative_means_for_matching_times.reshape(-1, 1) + \
                          diff_phi_deriv_times.T @ phi_times_denoised_obs

        """ Get log determinant """
        cholesky_eigenvals = jnp.diag(small_matrix_cholesky_tuple[0])
        log_eigenvals = jnp.log(cholesky_eigenvals)
        # We multiply with 2 since we compute logdet of cholesky factor which is "sqrt" of matrix
        log_determinant = 2 * jnp.sum(log_eigenvals)
        log_determinant = log_determinant + (n_train_times - n_features) * jnp.log(noise_variance)

        """ Get derivative covariance """
        first_term = diagonal_element_calculator(diff_phi_deriv_times.T, diff_phi_deriv_times)

        first_k_by_k_matrix = phi_times_phi_transposed
        second_k_by_k_matrix = jax.scipy.linalg.cho_solve(small_matrix_cholesky_tuple, phi_times_phi_transposed)
        second_k_by_k_matrix = phi_times_phi_transposed @ second_k_by_k_matrix
        complete_k_by_k_matrix = (first_k_by_k_matrix - second_k_by_k_matrix) / noise_variance

        second_term = diagonal_element_calculator(
            diff_phi_deriv_times.T,
            complete_k_by_k_matrix @ diff_phi_deriv_times
        )

        derivative_covariance = first_term - second_term
        derivative_covariance = derivative_covariance.reshape(-1)

        return smoothed_mean.reshape(-1), derivative_mean.reshape(-1), derivative_covariance, log_determinant + obs_term

    def filter_joint_smoother_params(all_params, dim_id):
        return {
            "noise_std": all_params["noise_std"][dim_id],
            "kernel": filter_approx_kernel_params(all_params["kernel"], dim_id)
        }

    def smoother_init(rng, num_dim):
        parameters = dict()
        parameters['noise_std'] = -2 * jnp.ones(num_dim)

        n_features = num_dim + 1
        _, kernel_params = approx_kernel_init(rng, n_features, num_dim)
        parameters["kernel"] = kernel_params
        return (), parameters

    @jax.jit
    def get_smoother_regularization(parameters, weights):
        num_dim = len(parameters.keys()) - 1
        objective = get_approx_kernel_regularization(parameters['kernel'], weights)
        for dim_id in range(num_dim):
            objective += weights["obs_noise"] * negative_log_likelihood_normal(
                parameters["noise_std"][dim_id],
                mu=-2.25,  # TODO: make more flexible, i.e. via dict call from weights
                sigma=2
            )
        return objective

    @jax.jit
    def smoother_posterior(
            evaluation_times: jnp.array,  # List over trajectories, 1D vectors (n_dim x 1)
            observation_times: jnp.array,  # List over trajectories, 1D vectors (n_dim x 1)
            ic_for_evaluation_times: jnp.array,  # List over trajectories, 1D arrays  (1 x n_dim)
            ic_for_observation_times: jnp.array,  # List over trajectories, 1D arrays  (1 x n_dim)
            observations: jnp.array,  # List over trajectories, 2D arrays (n_obs x n_dim)
            filtered_params: pytree,  # type jax.pytree
            prior_means_for_observation_times: jnp.array,  # List over trajectories, 2D arrays (n_obs x n_dim)
            prior_means_for_evaluation_times: jnp.array,  # List over trajectories, 2D arrays (n_obs x n_dim)
            prior_derivative_means_for_evaluation_times: jnp.array,
            core_parameters: pytree,
    ):
        # Placeholder for means and variance for different trajectories for the dimension dim_id
        phi_train_times = get_phi_transposed(observation_times, ic_for_observation_times,
                                             filtered_params["kernel"], core_parameters).T
        phi_train_times = phi_train_times / jnp.sqrt(phi_train_times.shape[0])

        """ get all feature matrices """
        phi_test_times = get_phi_transposed(evaluation_times, ic_for_evaluation_times, filtered_params["kernel"],
                                            core_parameters).T
        phi_test_times = phi_test_times / jnp.sqrt(phi_test_times.shape[0])

        test_phi_deriv_times = get_d_phi_transposed(evaluation_times.reshape(-1), ic_for_evaluation_times,
                                                    filtered_params["kernel"], core_parameters).T
        test_phi_deriv_times = test_phi_deriv_times / jnp.sqrt(test_phi_deriv_times.shape[0])

        """ get all observation related terms """

        sigma = filtered_params["noise_std"]
        noise_variance = make_positive(sigma) ** 2

        phi_times_phi_transposed = phi_train_times @ phi_train_times.T
        small_invertible_matrix = phi_times_phi_transposed + noise_variance * jnp.eye(
            phi_times_phi_transposed.shape[0])
        small_matrix_cholesky_tuple = jax.scipy.linalg.cho_factor(small_invertible_matrix)

        denoised_obs = phi_train_times @ (observations.reshape(-1, 1) - prior_means_for_observation_times.reshape(-1, 1))
        denoised_obs = jax.scipy.linalg.cho_solve(small_matrix_cholesky_tuple, denoised_obs)
        denoised_obs = phi_train_times.T @ denoised_obs
        denoised_obs = observations.reshape(-1, 1) - prior_means_for_observation_times.reshape(-1, 1) - denoised_obs
        denoised_obs = denoised_obs / noise_variance

        phi_times_denoised_obs = phi_train_times @ denoised_obs

        # Compute mean of states
        state_means = prior_means_for_evaluation_times.reshape(-1, 1) + phi_test_times.T @ phi_times_denoised_obs

        # Compute mean of derivatives
        derivative_means = prior_derivative_means_for_evaluation_times.reshape(-1, 1) + test_phi_deriv_times.T @ phi_times_denoised_obs

        """ Get covariances """
        first_term_states = diagonal_element_calculator(phi_test_times.T, phi_test_times)
        first_term_derivs = diagonal_element_calculator(test_phi_deriv_times.T, test_phi_deriv_times)

        first_k_by_k_matrix = phi_times_phi_transposed
        second_k_by_k_matrix = jax.scipy.linalg.cho_solve(small_matrix_cholesky_tuple, phi_times_phi_transposed)
        second_k_by_k_matrix = phi_times_phi_transposed @ second_k_by_k_matrix
        complete_k_by_k_matrix = (first_k_by_k_matrix - second_k_by_k_matrix) / noise_variance

        second_term_states = diagonal_element_calculator(phi_test_times.T, complete_k_by_k_matrix @ phi_test_times)
        second_term_derivs = diagonal_element_calculator(test_phi_deriv_times.T,
                                                         complete_k_by_k_matrix @ test_phi_deriv_times)

        state_variances = first_term_states - second_term_states
        state_variances = state_variances.reshape(-1, 1)

        derivative_variances = first_term_derivs - second_term_derivs
        derivative_variances = derivative_variances.reshape(-1, 1)

        return state_means.reshape(-1), state_variances.reshape(-1), derivative_means.reshape(
            -1), derivative_variances.reshape(-1)

    return smoother_init, smoother_apply, smoother_posterior, \
           filter_joint_smoother_params, get_smoother_regularization
