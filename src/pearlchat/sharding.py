"""Device mesh setup, tensor sharding, and visual partitioning utilities."""

import os
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from pearlchat.config import ShardingConfig


def create_device_mesh(
    mesh_shape: Tuple[int, int] = (4, 2),
    axis_names: Tuple[str, str] = ("data", "model"),
) -> Mesh:
    """Create a 2D JAX device mesh across data and model axes."""
    devices = jax.devices()
    total_required = mesh_shape[0] * mesh_shape[1]

    if len(devices) < total_required:
        # Fall back to 1D or adapt shape if fewer devices available
        if len(devices) >= 2 and len(devices) % 2 == 0:
            mesh_shape = (len(devices) // 2, 2)
        else:
            mesh_shape = (len(devices), 1)

    devices_subset = devices[: mesh_shape[0] * mesh_shape[1]]
    device_array = np.array(devices_subset).reshape(mesh_shape)
    return Mesh(device_array, axis_names)


def get_data_parallel_sharding(
    mesh: Mesh,
    data_axis_name: str = "data",
) -> NamedSharding:
    """Return NamedSharding for batch data parallel distribution."""
    return NamedSharding(mesh, P(data_axis_name, None))


def get_model_parallel_sharding(
    mesh: Mesh,
    model_axis_name: str = "model",
) -> NamedSharding:
    """Return NamedSharding for weight column model parallel distribution."""
    return NamedSharding(mesh, P(None, model_axis_name))


def shard_array(
    array: jax.Array,
    sharding: NamedSharding,
) -> jax.Array:
    """Put array on devices according to named sharding specification."""
    return jax.device_put(array, sharding)


def print_sharding_diagram(array: jax.Array, title: str = "Array sharding") -> None:
    """Print ASCII visual diagram of array sharding across devices."""
    print(f"\n{title}:")
    print(f"Shape: {array.shape} | Sharding: {array.sharding}")
    jax.debug.visualize_array_sharding(array)
