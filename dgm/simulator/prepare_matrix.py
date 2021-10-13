import jax
import jax.numpy as jnp
import jax.scipy as jsp


def create_marginally_stable_matrix(n, key, period_bound=jnp.pi / 2):
    a = jax.random.uniform(key=key, shape=(n, n))
    skew = 0.5 * (a - a.T)
    max_eigen_value = jnp.max(jnp.abs(jnp.linalg.eigvals(skew)))
    return skew / max_eigen_value * period_bound


def create_stable_matrix(n, key):
    diagonals = jax.random.uniform(key=key, shape=(n,), minval=-0.5, maxval=-0.1)
    transition_matrix = jax.random.uniform(key=key, shape=(n, n))
    transition_matrix, _, _ = jnp.linalg.svd(transition_matrix)
    return transition_matrix @ jnp.diag(diagonals) @ transition_matrix.T


def create_unstable_matrix(n, key):
    diagonals = jax.random.uniform(key=key, shape=(n,), minval=0, maxval=5)
    transition_matrix = jax.random.uniform(key=key, shape=(n, n))
    transition_matrix, _, _ = jnp.linalg.svd(transition_matrix)
    return transition_matrix @ jnp.diag(diagonals) @ transition_matrix.T


def create_matrix(triple, key):
    dim = sum(triple)
    key, *subkeys = jax.random.split(key, 5)
    stable_part = create_stable_matrix(triple[0], subkeys[0])
    marginally_stable_part = create_marginally_stable_matrix(triple[1], subkeys[1])
    unstable_part = create_unstable_matrix(triple[2], subkeys[2])
    whole_matrix = jsp.linalg.block_diag(stable_part, marginally_stable_part, unstable_part)
    transition_matrix = jax.random.uniform(key=subkeys[3], shape=(dim, dim))
    transition_matrix, _, _ = jnp.linalg.svd(transition_matrix)
    return transition_matrix @ whole_matrix @ transition_matrix.T


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    b = create_marginally_stable_matrix(8, subkey)
