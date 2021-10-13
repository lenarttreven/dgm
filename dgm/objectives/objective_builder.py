from typing import Callable, Any

import jax.numpy as jnp
import jax

pytree = Any
apply_fun = Callable[[pytree, jnp.array], jnp.array]


def get_objective_builder(
        apply_smoother: Callable,
        apply_dynamics: apply_fun,
        get_dynamics_regularization,
        get_smoother_regularization,
):
    @jax.jit
    def build_objective(
            parameters: pytree,
            times: jnp.array,  # shape n_obs x 1
            times_for_derivatives: jnp.array,  # shape n_deriv_obs x 1
            initial_conditions: jnp.array,  # shape n_obs x n_dim
            initial_conditions_for_derivatives: jnp.array,  # shape n_deriv_obs x n_dim
            observations: jnp.array,  # shape n_obs x n_dim
            betas: jnp.array,
            weights
    ) -> float:
        # Computer regularization
        objective = get_dynamics_regularization(parameters['dynamics'], weights)
        objective += get_smoother_regularization(parameters['smoother'], weights)

        # Compute smoother terms
        (
            posterior_means,
            posterior_smoother_derivative_means,
            posterior_smoother_derivatives_covariances,
            data_fit
        ) = apply_smoother(
            times, times_for_derivatives, initial_conditions, initial_conditions_for_derivatives, observations,
            parameters["smoother"]
        )
        # Compute dynamics terms
        posterior_dynamics_derivative_means, posterior_dynamics_derivatives_covariances = apply_dynamics(
            parameters["dynamics"], posterior_means)

        # Compute data fit term (marginal log likelihood in the case of full GP)

        mll_terms = jnp.sum(data_fit)
        objective += mll_terms

        # Compute fairness factor
        num_states = initial_conditions.shape[0]
        num_derivatives = initial_conditions_for_derivatives.shape[0]
        fairness_factor = num_states / num_derivatives

        # Compute Wasserstein distance
        wass_regularization = jnp.sum(
            betas * jnp.sum((posterior_smoother_derivative_means - posterior_dynamics_derivative_means) ** 2, axis=0))
        wass_regularization += jnp.sum(
            betas * jnp.sum(posterior_smoother_derivatives_covariances + posterior_dynamics_derivatives_covariances,
                            axis=0))
        wass_regularization -= 2 * jnp.sum(betas * jnp.sum(
            jnp.sqrt(posterior_smoother_derivatives_covariances * posterior_dynamics_derivatives_covariances), axis=0))
        wass_regularization = 2 * wass_regularization
        return objective + fairness_factor * wass_regularization

    return build_objective
