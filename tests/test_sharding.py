"""Tests for JAX device mesh and array sharding reassembly."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pearlchat.sharding import (
    create_device_mesh,
    get_data_parallel_sharding,
    get_model_parallel_sharding,
    shard_array,
)


def test_device_mesh_creation():
    mesh = create_device_mesh()
    assert mesh is not None
    assert "data" in mesh.axis_names
    assert "model" in mesh.axis_names


def test_data_parallel_reassembly():
    mesh = create_device_mesh()
    sharding = get_data_parallel_sharding(mesh)

    original = jnp.arange(32 * 8, dtype=jnp.float32).reshape((32, 8))
    sharded = shard_array(original, sharding)

    assert sharded.shape == original.shape
    reassembled = np.array(sharded)
    np.testing.assert_allclose(reassembled, np.array(original))


def test_model_parallel_reassembly():
    mesh = create_device_mesh()
    sharding = get_model_parallel_sharding(mesh)

    original = jnp.ones((16, 32), dtype=jnp.float32)
    sharded = shard_array(original, sharding)

    assert sharded.shape == original.shape
    reassembled = np.array(sharded)
    np.testing.assert_allclose(reassembled, np.array(original))
