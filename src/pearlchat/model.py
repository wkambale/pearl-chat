"""GPT-2 style decoder-only transformer implemented in Flax NNX."""

from typing import Optional
import jax
import jax.numpy as jnp
from flax import nnx

from pearlchat.config import ModelConfig


class CausalSelfAttention(nnx.Module):
    """Multi-head causal self-attention module."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nnx.Linear(embed_dim, embed_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(embed_dim, embed_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(embed_dim, embed_dim, use_bias=False, rngs=rngs)
        self.out_proj = nnx.Linear(embed_dim, embed_dim, use_bias=False, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        deterministic: bool = True,
        return_attention_weights: bool = False,
    ) -> tuple[jax.Array, Optional[jax.Array]]:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).reshape((batch_size, seq_len, self.num_heads, self.head_dim)).swapaxes(1, 2)
        k = self.k_proj(x).reshape((batch_size, seq_len, self.num_heads, self.head_dim)).swapaxes(1, 2)
        v = self.v_proj(x).reshape((batch_size, seq_len, self.num_heads, self.head_dim)).swapaxes(1, 2)

        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.einsum("bhid,bhjd->bhij", q, k) * scale

        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        scores = jnp.where(causal_mask, scores, -1e9)
        weights = jax.nn.softmax(scores, axis=-1)
        weights_dropped = self.dropout(weights, deterministic=deterministic)

        context = jnp.einsum("bhij,bhjd->bhid", weights_dropped, v)
        context = context.swapaxes(1, 2).reshape((batch_size, seq_len, self.embed_dim))
        out = self.out_proj(context)

        attn_weights = weights if return_attention_weights else None
        return out, attn_weights


class MLP(nnx.Module):
    """Feed-forward multi-layer perceptron with GELU activation."""

    def __init__(
        self,
        embed_dim: int,
        feed_forward_dim: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        self.dense_1 = nnx.Linear(embed_dim, feed_forward_dim, rngs=rngs)
        self.dense_2 = nnx.Linear(feed_forward_dim, embed_dim, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        h = self.dense_1(x)
        h = nnx.gelu(h)
        h = self.dense_2(h)
        return self.dropout(h, deterministic=deterministic)


class TransformerBlock(nnx.Module):
    """Single transformer decoder block with pre-layer normalization."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feed_forward_dim: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        self.ln_1 = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
        self.attn = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )
        self.ln_2 = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
        self.mlp = MLP(
            embed_dim=embed_dim,
            feed_forward_dim=feed_forward_dim,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        deterministic: bool = True,
        return_attention_weights: bool = False,
    ) -> tuple[jax.Array, Optional[jax.Array]]:
        norm_1 = self.ln_1(x)
        attn_out, weights = self.attn(
            norm_1,
            deterministic=deterministic,
            return_attention_weights=return_attention_weights,
        )
        x = x + attn_out

        norm_2 = self.ln_2(x)
        mlp_out = self.mlp(norm_2, deterministic=deterministic)
        x = x + mlp_out
        return x, weights


class PearlChatModel(nnx.Module):
    """Decoder-only language model built in Flax NNX."""

    def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
        self.config = config
        self.token_embedding = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.embed_dim,
            rngs=rngs,
        )
        self.position_embedding = nnx.Embed(
            num_embeddings=config.context_length,
            features=config.embed_dim,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)

        self.blocks = nnx.List([
            TransformerBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                feed_forward_dim=config.feed_forward_dim,
                dropout_rate=config.dropout_rate,
                rngs=rngs,
            )
            for _ in range(config.num_layers)
        ])

        self.ln_f = nnx.LayerNorm(num_features=config.embed_dim, rngs=rngs)

    def __call__(
        self,
        token_ids: jax.Array,
        deterministic: bool = True,
        return_attention_weights: bool = False,
    ) -> tuple[jax.Array, Optional[list[jax.Array]]]:
        """Forward pass taking token ids of shape (batch, seq_len)."""
        batch_size, seq_len = token_ids.shape
        if seq_len > self.config.context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds context length {self.config.context_length}."
            )

        positions = jnp.arange(0, seq_len, dtype=jnp.int32)
        positions = jnp.broadcast_to(positions, (batch_size, seq_len))

        tok_emb = self.token_embedding(token_ids)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(tok_emb + pos_emb, deterministic=deterministic)

        collected_weights = [] if return_attention_weights else None
        for block in self.blocks:
            x, weights = block(
                x,
                deterministic=deterministic,
                return_attention_weights=return_attention_weights,
            )
            if return_attention_weights:
                collected_weights.append(weights)

        x = self.ln_f(x)
        # Weight-tied projection to vocabulary logits
        embedding_weights = self.token_embedding.embedding[...]
        logits = jnp.matmul(x, embedding_weights.T)

        return logits, collected_weights

    def count_parameters(self) -> int:
        """Count total trainable parameters in the model."""
        state = nnx.state(self)
        total = 0
        for leaf in jax.tree_util.tree_leaves(state):
            if isinstance(leaf, jax.Array):
                total += leaf.size
        return total
