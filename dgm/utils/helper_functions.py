import collections
from re import search
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import wandb
from jax import jit, tree_leaves
from jax.lib import xla_bridge
from jax.tree_util import tree_flatten

pytree = Any


@jit
def make_positive(x):
    return jnp.log(1 + jnp.exp(x))


def create_dict_from_params(params):
    variables_dict = dict()
    leaves = tree_leaves(params)
    for index, component in enumerate(leaves):
        key = '_component_' + str(index)
        value = np.array(component)
        variables_dict[key] = value
    return variables_dict


def unroll_dictionary(dictionary, parent_key="", sep="_"):
    unrolled_dictionary = dict()
    for key, value in dictionary.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, collections.MutableMapping):
            unrolled_dictionary.update(unroll_dictionary(value, new_key, sep=sep))
        else:
            if xla_bridge.get_backend().platform == 'cpu':
                if value is None:
                    continue
                elif type(value) != jax.interpreters.xla._DeviceArray:
                    network_dict = create_dict_from_params(value)
                    for key_network, value_network in network_dict.items():
                        unrolled_dictionary[new_key + key_network] = wandb.Histogram(value_network)
                else:
                    if search("lengthscales", new_key) or search("std", new_key):
                        value = make_positive(value)
                    value = np.array(value)
                    if value.size > 1:
                        value = wandb.Histogram(value)
                    unrolled_dictionary[new_key] = value
            else:
                if value is None:
                    continue
                elif type(value) != jax.interpreters.xla._DeviceArray and type(
                        value) != jaxlib.xla_extension.DeviceArray:
                    network_dict = create_dict_from_params(value)
                    for key_network, value_network in network_dict.items():
                        unrolled_dictionary[new_key + key_network] = wandb.Histogram(value_network)
                else:
                    if search("lengthscales", new_key) or search("std", new_key):
                        value = make_positive(value)
                    value = np.array(value)
                    if value.size > 1:
                        value = wandb.Histogram(value)
                    unrolled_dictionary[new_key] = value
    return unrolled_dictionary


def stack_over_dimension(unstacked_list):
    num_dimensions = len(unstacked_list)
    num_trajectories = len(unstacked_list[0])
    stacked_list = [
        jnp.concatenate([unstacked_list[j][i] for j in range(num_dimensions)], axis=1)
        for i in range(num_trajectories)
    ]
    return stacked_list


def log_space(start, end, num_points, base=2):
    return jnp.array(
        [start + (end - start) * (1 - base ** k) / (1 - base ** num_points) for k in range(num_points + 1)])


def diag_product(a: jnp.array, b: jnp.array):
    # Return jnp.diag(a @ b) without computing the whole matrix a @ b
    return jnp.sum(a * b.T, axis=1)


def replace_str(str):
    new_str = ''
    for char in str:
        if char == '[':
            char = '('
        elif char == ']':
            char = ')'
        new_str += char
    return new_str


@jax.jit
def squared_l2_norm(tree: pytree) -> jnp.array:
    """Compute the squared l2 norm of a pytree of arrays. Useful for weight decay."""
    leaves, _ = tree_flatten(tree)
    return sum(jnp.vdot(x, x) for x in leaves)


@jax.jit
def negative_log_likelihood_normal(x: jnp.array, mu: float, sigma: float):
    return x.size * jnp.log(2 * jnp.pi * (sigma ** 2)) / 2 + jnp.sum(((x - mu) ** 2) / (2 * (sigma ** 2)))


if __name__ == "__main__":
    test = {"a": 1, "c": {"a": 2, "b": {"x": 5, "y": 10}}, "d": [1, 2, 3]}
    print(unroll_dictionary(test))

    seed = 0
    rng = jax.random.PRNGKey(seed)
    rng, subkey = jax.random.split(rng)
    test_a = jax.random.normal(key=subkey, shape=(4, 4), dtype=jnp.float32)
    rng, subkey = jax.random.split(rng)
    test_b = jax.random.normal(key=subkey, shape=(4, 4), dtype=jnp.float32)

    print(diag_product(test_a, test_b))
    print(jnp.diag(test_a @ test_b))
