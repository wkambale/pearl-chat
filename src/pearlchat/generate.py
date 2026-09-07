"""Autoregressive text generation and sampling for Pearl-Chat."""

from typing import Iterator, List, Optional
import jax
import jax.numpy as jnp

from pearlchat.model import PearlChatModel
from pearlchat.tokenizer import LugandaTokenizer


def sample_next_token(
    logits: jax.Array,
    key: jax.Array,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
) -> int:
    """Apply temperature, top-k, top-p filtering and sample a single token id."""
    if temperature <= 0.0:
        return int(jnp.argmax(logits))

    scaled_logits = logits / max(temperature, 1e-4)

    # Top-k filtering
    if top_k > 0 and top_k < scaled_logits.shape[-1]:
        top_k_vals, _ = jax.lax.top_k(scaled_logits, top_k)
        cutoff = top_k_vals[-1]
        scaled_logits = jnp.where(scaled_logits < cutoff, -1e9, scaled_logits)

    # Top-p (nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_indices = jnp.argsort(scaled_logits)[::-1]
        sorted_logits = scaled_logits[sorted_indices]
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1))

        # Remove tokens with cumulative probability above threshold (keep at least 1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove = jnp.roll(sorted_indices_to_remove, 1)
        sorted_indices_to_remove = sorted_indices_to_remove.at[0].set(False)

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        scaled_logits = scaled_logits.at[indices_to_remove].set(-1e9)

    token_id = jax.random.categorical(key, scaled_logits)
    return int(token_id)


def generate_stream(
    model: PearlChatModel,
    tokenizer: LugandaTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    seed: int = 42,
) -> Iterator[str]:
    """Stream generated tokens one by one for interactive chat UI."""
    prompt_tokens = tokenizer.encode(prompt)
    if not prompt_tokens:
        prompt_tokens = [tokenizer.eos_id]

    current_ids = list(prompt_tokens)
    key = jax.random.key(seed)
    context_length = model.config.context_length

    for _ in range(max_new_tokens):
        key, subkey = jax.random.split(key)
        input_window = current_ids[-context_length:]
        input_tensor = jnp.array([input_window], dtype=jnp.int32)

        logits, _ = model(input_tensor, deterministic=True)
        next_token_logits = logits[0, -1, :]

        next_id = sample_next_token(
            next_token_logits,
            key=subkey,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        if next_id == tokenizer.eos_id:
            break

        current_ids.append(next_id)
        yield tokenizer.decode([next_id])


def generate_text(
    model: PearlChatModel,
    tokenizer: LugandaTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    seed: int = 42,
) -> str:
    """Generate complete text response to a prompt."""
    tokens = list(
        generate_stream(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
    )
    return "".join(tokens)
