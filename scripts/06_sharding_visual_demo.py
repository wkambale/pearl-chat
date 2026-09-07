"""Visual demonstration of JAX device mesh, data parallelism, and model parallelism."""

import os
# Configure simulated 8 logical CPU devices before JAX initialization
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from pearlchat.sharding import (
    create_device_mesh,
    get_data_parallel_sharding,
    get_model_parallel_sharding,
    print_sharding_diagram,
    shard_array,
)


def run_sharding_demo() -> None:
    """Demonstrate batch data parallelism and weight tensor parallelism."""
    print("------------------------------------------------------------")
    print("Pearl-Chat JAX sharding demonstration")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Discovered devices: {len(jax.devices())}")
    for idx, dev in enumerate(jax.devices()):
        print(f"  Device {idx}: {dev}")
    print("------------------------------------------------------------")

    mesh = create_device_mesh(mesh_shape=(4, 2), axis_names=("data", "model"))
    print(f"Device mesh initialized: {mesh}\n")

    # 1. Data-parallel batch sharding
    print("1. Data-parallel batch partitioning:")
    print("   Splitting batch dimension across 'data' axis (4 ways).")
    batch_size = 16
    seq_len = 8
    dummy_batch = jnp.arange(batch_size * seq_len, dtype=jnp.float32).reshape(
        (batch_size, seq_len)
    )

    data_sharding = get_data_parallel_sharding(mesh, data_axis_name="data")
    sharded_batch = shard_array(dummy_batch, data_sharding)
    print_sharding_diagram(sharded_batch, title="Data-parallel input batch")

    # 2. Tensor-parallel weight matrix sharding
    print("\n2. Tensor-parallel weight matrix partitioning:")
    print("   Splitting projection weight column dimension across 'model' axis (2 ways).")
    embed_dim = 128
    vocab_size = 256
    weight_matrix = jnp.ones((embed_dim, vocab_size), dtype=jnp.float32)

    model_sharding = get_model_parallel_sharding(mesh, model_axis_name="model")
    sharded_weights = shard_array(weight_matrix, model_sharding)
    print_sharding_diagram(sharded_weights, title="Tensor-parallel projection weights")

    # 3. Distributed matrix multiplication
    print("\n3. Distributed computation across the mesh:")
    print("   Multiplying sharded inputs by sharded weights.")

    @jax.jit
    def distributed_forward(x: jax.Array, w: jax.Array) -> jax.Array:
        # x is partitioned along batch (data axis)
        # w is partitioned along columns (model axis)
        # result is 2D partitioned along (data, model)
        return jnp.matmul(x, w)

    out_sharding = NamedSharding(mesh, P("data", "model"))
    x_for_matmul = jnp.ones((batch_size, embed_dim), dtype=jnp.float32)
    x_sharded_matmul = shard_array(x_for_matmul, data_sharding)

    distributed_out = distributed_forward(x_sharded_matmul, sharded_weights)
    print_sharding_diagram(
        distributed_out,
        title="Output tensor sharded across both data and model axes",
    )

    print("\nSharding demonstration completed successfully.")


if __name__ == "__main__":
    run_sharding_demo()
