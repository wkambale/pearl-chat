"""Tests for PearlChatModel shapes, parameter counting, and causal masking."""

import jax
import jax.numpy as jnp
from flax import nnx
import pytest

from pearlchat.config import ModelConfig
from pearlchat.model import PearlChatModel


@pytest.fixture
def test_model():
    config = ModelConfig(
        vocab_size=256,
        context_length=32,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        feed_forward_dim=128,
        dropout_rate=0.0,
    )
    rngs = nnx.Rngs(params=42, dropout=43)
    return PearlChatModel(config=config, rngs=rngs)


def test_forward_pass_shape(test_model):
    batch_size = 3
    seq_len = 16
    token_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    logits, _ = test_model(token_ids, deterministic=True)
    expected_shape = (batch_size, seq_len, test_model.config.vocab_size)
    assert logits.shape == expected_shape


def test_parameter_count(test_model):
    param_count = test_model.count_parameters()
    assert param_count > 0
    assert isinstance(param_count, int)


def test_causal_mask_attention_weights(test_model):
    batch_size = 2
    seq_len = 8
    token_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    _, weights_list = test_model(
        token_ids,
        deterministic=True,
        return_attention_weights=True,
    )
    assert weights_list is not None
    assert len(weights_list) == test_model.config.num_layers

    for weights in weights_list:
        # weights shape: (batch_size, num_heads, seq_len, seq_len)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                future_attn = weights[:, :, i, j]
                np_future = jnp.max(jnp.abs(future_attn))
                assert float(np_future) == pytest.approx(0.0, abs=1e-6)
