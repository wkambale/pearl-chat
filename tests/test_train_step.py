"""Tests for training step, loss reduction on synthetic data, and checkpoint reproduction."""

import tempfile
from pathlib import Path
import jax
import jax.numpy as jnp
from flax import nnx
import pytest

from pearlchat.config import ModelConfig, TrainConfig
from pearlchat.checkpointing import CheckpointManager
from pearlchat.model import PearlChatModel
from pearlchat.train import compute_loss, create_optimizer, train_step


def test_loss_decreases_over_steps():
    """Verify loss strictly decreases over 50 steps on fixed synthetic data."""
    config = ModelConfig(
        vocab_size=32,
        context_length=8,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        feed_forward_dim=32,
        dropout_rate=0.0,
    )
    rngs = nnx.Rngs(params=42, dropout=42)
    model = PearlChatModel(config, rngs)

    train_config = TrainConfig(
        learning_rate=1e-2,
        total_steps=50,
        warmup_steps=5,
        weight_decay=0.0,
        grad_clip_norm=1.0,
    )
    optimizer, _ = create_optimizer(model, train_config)

    fixed_batch = {
        "inputs": jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32),
        "targets": jnp.array([[2, 3, 4, 5, 6, 7, 8, 9]], dtype=jnp.int32),
    }

    initial_loss = float(compute_loss(model, fixed_batch, deterministic=True))

    for _ in range(50):
        _ = train_step(model, optimizer, fixed_batch)

    final_loss = float(compute_loss(model, fixed_batch, deterministic=True))

    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"
    assert final_loss < initial_loss * 0.5, f"Expected at least 50% drop, got {initial_loss} -> {final_loss}"


def test_checkpoint_save_and_restore_reproducibility():
    """Verify restored checkpoint reproduces identical loss."""
    config = ModelConfig(
        vocab_size=32,
        context_length=8,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        feed_forward_dim=32,
        dropout_rate=0.0,
    )
    rngs = nnx.Rngs(params=42, dropout=42)
    model1 = PearlChatModel(config, rngs)

    fixed_batch = {
        "inputs": jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32),
        "targets": jnp.array([[2, 3, 4, 5, 6, 7, 8, 9]], dtype=jnp.int32),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir)
        manager.save_checkpoint(model1, step=10, config=config)

        # Create a second fresh model with different weights
        rngs2 = nnx.Rngs(params=999, dropout=999)
        model2 = PearlChatModel(config, rngs2)

        # Confirm different initial loss
        loss1 = float(compute_loss(model1, fixed_batch, deterministic=True))
        loss2_before = float(compute_loss(model2, fixed_batch, deterministic=True))
        assert loss1 != loss2_before

        # Restore model2 from model1's checkpoint
        manager.restore_latest_checkpoint(model2)
        loss2_after = float(compute_loss(model2, fixed_batch, deterministic=True))

        assert pytest.approx(loss1, rel=1e-5) == loss2_after
