"""Web application and inference server for Pearl-Chat."""

import argparse
import json
import mimetypes
import os
from pathlib import Path
import threading
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Tuple, Union
from http.server import HTTPServer, SimpleHTTPRequestHandler

import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import ByteLevel


@dataclass
class ModelConfig:
    """Hyperparameters for the Pearl-Chat transformer model."""

    vocab_size: int = 8192
    context_length: int = 256
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 8
    feed_forward_dim: int = 1024
    dropout_rate: float = 0.1


class LugandaTokenizer:
    """Byte-level byte-pair encoding tokenizer for Luganda."""

    def __init__(self, tokenizer: Optional[ByteLevelBPETokenizer] = None):
        self.tokenizer = tokenizer
        self._eos_token = "<|endoftext|>"
        self._pad_token = "<|pad|>"
        self._unk_token = "<|unk|>"

    @classmethod
    def load(cls, tokenizer_dir: Union[str, Path]) -> "LugandaTokenizer":
        vocab_file = os.path.join(tokenizer_dir, "vocab.json")
        merges_file = os.path.join(tokenizer_dir, "merges.txt")
        if not os.path.exists(vocab_file) or not os.path.exists(merges_file):
            raise FileNotFoundError(
                f"Missing vocab.json or merges.txt in directory: {tokenizer_dir}"
            )
        raw_tokenizer = ByteLevelBPETokenizer(
            vocab=vocab_file,
            merges=merges_file,
            lowercase=False,
        )
        raw_tokenizer.post_processor = ByteLevel(trim_offsets=False)
        return cls(tokenizer=raw_tokenizer)

    def encode(self, text: str) -> List[int]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    @property
    def eos_id(self) -> int:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")
        token_id = self.tokenizer.token_to_id(self._eos_token)
        return token_id if token_id is not None else 0

    @property
    def pad_id(self) -> int:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")
        token_id = self.tokenizer.token_to_id(self._pad_token)
        return token_id if token_id is not None else self.eos_id

    @property
    def unk_id(self) -> int:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")
        token_id = self.tokenizer.token_to_id(self._unk_token)
        return token_id if token_id is not None else 2

    @property
    def vocab_size(self) -> int:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")
        return self.tokenizer.get_vocab_size()


class CausalSelfAttention(nnx.Module):
    """Multi-head causal self-attention module."""

    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float, rngs: nnx.Rngs):
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
        return out, (weights if return_attention_weights else None)


class MLP(nnx.Module):
    """Feed-forward multi-layer perceptron with GELU activation."""

    def __init__(self, embed_dim: int, feed_forward_dim: int, dropout_rate: float, rngs: nnx.Rngs):
        self.dense_1 = nnx.Linear(embed_dim, feed_forward_dim, rngs=rngs)
        self.dense_2 = nnx.Linear(feed_forward_dim, embed_dim, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        h = nnx.gelu(self.dense_1(x))
        return self.dropout(self.dense_2(h), deterministic=deterministic)


class TransformerBlock(nnx.Module):
    """Single transformer decoder block with pre-layer normalization."""

    def __init__(self, embed_dim: int, num_heads: int, feed_forward_dim: int, dropout_rate: float, rngs: nnx.Rngs):
        self.ln_1 = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout_rate, rngs)
        self.ln_2 = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
        self.mlp = MLP(embed_dim, feed_forward_dim, dropout_rate, rngs)

    def __call__(
        self,
        x: jax.Array,
        deterministic: bool = True,
        return_attention_weights: bool = False,
    ) -> tuple[jax.Array, Optional[jax.Array]]:
        attn_out, weights = self.attn(
            self.ln_1(x),
            deterministic=deterministic,
            return_attention_weights=return_attention_weights,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x), deterministic=deterministic)
        return x, weights


class PearlChatModel(nnx.Module):
    """Decoder-only language model built in Flax NNX."""

    def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
        self.config = config
        self.token_embedding = nnx.Embed(num_embeddings=config.vocab_size, features=config.embed_dim, rngs=rngs)
        self.position_embedding = nnx.Embed(num_embeddings=config.context_length, features=config.embed_dim, rngs=rngs)
        self.dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.blocks = nnx.List([
            TransformerBlock(config.embed_dim, config.num_heads, config.feed_forward_dim, config.dropout_rate, rngs)
            for _ in range(config.num_layers)
        ])
        self.ln_f = nnx.LayerNorm(num_features=config.embed_dim, rngs=rngs)

    def __call__(
        self,
        token_ids: jax.Array,
        deterministic: bool = True,
        return_attention_weights: bool = False,
    ) -> tuple[jax.Array, Optional[list[jax.Array]]]:
        batch_size, seq_len = token_ids.shape
        positions = jnp.broadcast_to(jnp.arange(0, seq_len, dtype=jnp.int32), (batch_size, seq_len))
        x = self.dropout(self.token_embedding(token_ids) + self.position_embedding(positions), deterministic=deterministic)
        collected_weights = [] if return_attention_weights else None
        for block in self.blocks:
            x, weights = block(x, deterministic=deterministic, return_attention_weights=return_attention_weights)
            if return_attention_weights:
                collected_weights.append(weights)
        x = self.ln_f(x)
        logits = jnp.matmul(x, self.token_embedding.embedding[...].T)
        return logits, collected_weights

    def count_parameters(self) -> int:
        state = nnx.state(self)
        return sum(leaf.size for leaf in jax.tree_util.tree_leaves(state) if isinstance(leaf, jax.Array))


def _restore_int_keys(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {
            int(k) if isinstance(k, str) and k.isdigit() else k: _restore_int_keys(v)
            for k, v in tree.items()
        }
    return tree


class CheckpointManager:
    """Manages restoring Flax NNX model states using Orbax."""

    def __init__(self, checkpoint_dir: Union[str, Path]):
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpointer = ocp.StandardCheckpointer()

    def restore_latest_checkpoint(self, model: PearlChatModel) -> Tuple[PearlChatModel, int]:
        latest_file = self.checkpoint_dir / "latest_step.txt"
        if latest_file.exists():
            step = int(latest_file.read_text(encoding="utf-8").strip())
            step_dir = self.checkpoint_dir / f"step_{step}"
        else:
            step_dirs = [d for d in self.checkpoint_dir.glob("step_*") if d.is_dir()]
            if not step_dirs:
                if (self.checkpoint_dir / "model_state").exists():
                    step_dir = self.checkpoint_dir
                    step = 0
                else:
                    raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
            else:
                step_dirs.sort(key=lambda d: int(d.name.split("_")[-1]))
                step_dir = step_dirs[-1]
                step = int(step_dir.name.split("_")[-1])

        state_path = step_dir / "model_state"
        target_dict = nnx.to_pure_dict(nnx.state(model))
        try:
            restored_state = self.checkpointer.restore(state_path, target=target_dict)
        except Exception:
            restored_state = self.checkpointer.restore(state_path)
        nnx.update(model, _restore_int_keys(restored_state))
        return model, step


def sample_next_token(
    logits: jax.Array,
    key: jax.Array,
    temperature: float = 0.65,
    top_k: int = 40,
    top_p: float = 0.9,
    generated_ids: Optional[List[int]] = None,
    repetition_penalty: float = 1.0,
) -> int:
    if temperature <= 0.0:
        return int(jnp.argmax(logits))
    scaled = jnp.array(logits, dtype=jnp.float32)
    if generated_ids and repetition_penalty > 1.0:
        for tid in set(generated_ids):
            if scaled[tid] > 0:
                scaled = scaled.at[tid].set(scaled[tid] / repetition_penalty)
            else:
                scaled = scaled.at[tid].set(scaled[tid] * repetition_penalty)
    scaled_logits = scaled / max(temperature, 1e-4)
    if 0 < top_k < scaled_logits.shape[-1]:
        top_k_vals, _ = jax.lax.top_k(scaled_logits, top_k)
        scaled_logits = jnp.where(scaled_logits < top_k_vals[-1], -1e9, scaled_logits)
    if 0.0 < top_p < 1.0:
        sorted_indices = jnp.argsort(scaled_logits)[::-1]
        sorted_logits = scaled_logits[sorted_indices]
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1))
        sorted_indices_to_remove = jnp.roll(cumulative_probs > top_p, 1).at[0].set(False)
        scaled_logits = scaled_logits.at[sorted_indices[sorted_indices_to_remove]].set(-1e9)
    return int(jax.random.categorical(key, scaled_logits))


def generate_stream(
    model: PearlChatModel,
    tokenizer: LugandaTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.65,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.25,
    seed: int = 42,
    min_new_tokens: int = 4,
) -> Iterator[str]:
    normalized_prompt = prompt.rstrip(" ") if prompt.rstrip(" ").endswith("Omuyambi:") else prompt
    prompt_tokens = tokenizer.encode(normalized_prompt) or [tokenizer.eos_id]
    current_ids = list(prompt_tokens)
    generated = []
    key = jax.random.key(seed)
    context_length = model.config.context_length
    previous_text = ""

    for step_idx in range(max_new_tokens):
        key, subkey = jax.random.split(key)
        input_tensor = jnp.array([current_ids[-context_length:]], dtype=jnp.int32)
        logits, _ = model(input_tensor, deterministic=True)
        next_token_logits = logits[0, -1, :]
        if step_idx < min_new_tokens:
            next_token_logits = next_token_logits.at[tokenizer.eos_id].set(-1e9)

        next_id = sample_next_token(
            next_token_logits,
            key=subkey,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generated_ids=generated,
            repetition_penalty=repetition_penalty,
        )
        if next_id == tokenizer.eos_id:
            break
        current_ids.append(next_id)
        generated.append(next_id)

        full_text = tokenizer.decode(generated)
        token_str = full_text[len(previous_text):]
        previous_text = full_text
        if "<|endoftext|>" not in token_str and token_str:
            yield token_str


class ModelHolder:
    """Thread-safe container for active model and tokenizer."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.model: Optional[PearlChatModel] = None
        self.tokenizer: Optional[LugandaTokenizer] = None
        self.source: str = "kambale/pearl-chat"
        self.status: str = "Uninitialized"

    def load(
        self,
        source: str = "checkpoints/pearl-chat",
        tokenizer_dir: str = "data/tokenizer",
    ) -> None:
        """Load model and tokenizer from local path or Hugging Face repository."""
        with self.lock:
            self.status = "Loading"
            target_path = Path(source)

            # Prefer verified local export if default HF repo ID is passed
            if source == "kambale/pearl-chat" and Path("checkpoints/pearl-chat/model_state").exists():
                target_path = Path("checkpoints/pearl-chat").resolve()
            elif not target_path.exists():
                try:
                    from huggingface_hub import snapshot_download
                    downloaded_dir = snapshot_download(repo_id=source)
                    target_path = Path(downloaded_dir)
                except Exception as err:
                    if Path("checkpoints/pearl-chat/model_state").exists():
                        target_path = Path("checkpoints/pearl-chat").resolve()
                    elif Path("checkpoints/pearlchat-warm-start").exists():
                        target_path = Path("checkpoints/pearlchat-warm-start").resolve()
                    else:
                        self.status = f"Error: {err}"
                        raise

            # Prioritize tokenizer packaged inside target_path to prevent tokenizer/checkpoint mismatch
            if (target_path / "vocab.json").exists():
                tok_path = target_path
            elif tokenizer_dir and Path(tokenizer_dir).exists() and (Path(tokenizer_dir) / "vocab.json").exists():
                tok_path = Path(tokenizer_dir)
            else:
                raise FileNotFoundError(
                    f"Tokenizer files not found in {target_path} or {tokenizer_dir}."
                )

            self.tokenizer = LugandaTokenizer.load(tok_path)

            # Check for metadata/config (defaulting to ~8.5M 8-layer architecture)
            context_length = 256
            embed_dim = 256
            num_heads = 8
            num_layers = 8
            feed_forward_dim = 1024

            meta_file = target_path / "metadata.json"
            config_file = target_path / "config.json"
            if config_file.exists():
                with open(config_file, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                    context_length = cfg_data.get("context_length", context_length)
                    embed_dim = cfg_data.get("embed_dim", embed_dim)
                    num_heads = cfg_data.get("num_heads", num_heads)
                    num_layers = cfg_data.get("num_layers", num_layers)
                    feed_forward_dim = cfg_data.get("feed_forward_dim", feed_forward_dim)
            elif meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                    context_length = meta_data.get("context_length", context_length)
                    embed_dim = meta_data.get("embed_dim", embed_dim)
                    num_heads = meta_data.get("num_heads", num_heads)
                    num_layers = meta_data.get("num_layers", num_layers)
                    feed_forward_dim = meta_data.get("feed_forward_dim", feed_forward_dim)

            model_config = ModelConfig(
                vocab_size=self.tokenizer.vocab_size,
                context_length=context_length,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                feed_forward_dim=feed_forward_dim,
                dropout_rate=0.0,
            )

            rngs = nnx.Rngs(params=42, dropout=42)
            model = PearlChatModel(config=model_config, rngs=rngs)

            if target_path.exists():
                manager = CheckpointManager(checkpoint_dir=target_path)
                try:
                    model, _ = manager.restore_latest_checkpoint(model)
                except Exception:
                    pass

            self.model = model
            self.source = str(target_path)
            self.status = "Ready"


_GLOBAL_HOLDER = ModelHolder()


class PearlChatRequestHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for web app static assets and inference API."""

    def __init__(self, *args, **kwargs):
        # Serve static assets from ui directory
        ui_dir = Path(__file__).parent.resolve()
        super().__init__(*args, directory=str(ui_dir), **kwargs)

    def end_headers(self) -> None:
        """Inject cache-control headers on static file responses."""
        if self.command == "GET" and not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self) -> None:
        """Handle GET requests for static assets or info endpoint."""
        if self.path == "/api/info":
            self._handle_api_info()
        elif self.path in ("/", ""):
            self.path = "/index.html"
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self) -> None:
        """Handle POST requests for inference or model loading."""
        if self.path == "/api/chat":
            self._handle_api_chat()
        elif self.path == "/api/load":
            self._handle_api_load()
        else:
            self.send_error(404, "Endpoint not found")

    def _handle_api_info(self) -> None:
        """Return model metadata and execution status."""
        with _GLOBAL_HOLDER.lock:
            model = _GLOBAL_HOLDER.model
            tokenizer = _GLOBAL_HOLDER.tokenizer

            info = {
                "status": _GLOBAL_HOLDER.status,
                "source": _GLOBAL_HOLDER.source,
                "backend": f"JAX {jax.default_backend().upper()}",
                "parameters": model.count_parameters() if model else 0,
                "vocab_size": tokenizer.vocab_size if tokenizer else 0,
                "context_length": model.config.context_length if model else 128,
            }

        response_bytes = json.dumps(info).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response_bytes)

    def _handle_api_load(self) -> None:
        """Load model from specified source path or repository."""
        content_len = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_len)
        try:
            req_data = json.loads(post_data.decode("utf-8"))
            source = req_data.get("source", "kambale/pearl-chat")
            _GLOBAL_HOLDER.load(source=source)
            result = {"status": "ok", "message": f"Loaded model from {source}"}
            code = 200
        except Exception as err:
            result = {"status": "error", "error": str(err)}
            code = 500

        response_bytes = json.dumps(result).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response_bytes)

    def _handle_api_chat(self) -> None:
        """Stream generated tokens as Server-Sent Events."""
        content_len = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_len)
        try:
            req = json.loads(post_data.decode("utf-8"))
        except Exception:
            self.send_error(400, "Invalid JSON payload")
            return

        prompt = req.get("prompt", "").strip()
        chat_prompt = prompt if "Omuntu:" in prompt else f"Omuntu: {prompt}\nOmuyambi: "
        temperature = float(req.get("temperature", 0.65))
        top_p = float(req.get("top_p", 0.9))
        top_k = int(req.get("top_k", 40))
        max_tokens = int(req.get("max_tokens", 48))

        with _GLOBAL_HOLDER.lock:
            model = _GLOBAL_HOLDER.model
            tokenizer = _GLOBAL_HOLDER.tokenizer

        if model is None or tokenizer is None:
            self.send_error(503, "Model is not loaded")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        accumulated_text = ""
        try:
            for token in generate_stream(
                model=model,
                tokenizer=tokenizer,
                prompt=chat_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=1.25,
                min_new_tokens=4,
            ):
                candidate = accumulated_text + token
                if "Omuntu:" in candidate or "User:" in candidate or "Omuyambi:" in candidate:
                    break

                accumulated_text += token
                payload = f"data: {json.dumps({'token': token})}\n\n"
                self.wfile.write(payload.encode("utf-8"))
                self.wfile.flush()

            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.close_connection = True


def load_model_and_tokenizer(
    checkpoint_dir: str = "kambale/pearl-chat",
    tokenizer_dir: str = "data/tokenizer",
) -> Tuple[PearlChatModel, LugandaTokenizer]:
    """Compatibility loader for tests and scripts."""
    try:
        _GLOBAL_HOLDER.load(source=checkpoint_dir, tokenizer_dir=tokenizer_dir)
    except Exception:
        if checkpoint_dir != "checkpoints/pearlchat-warm-start" and Path("checkpoints/pearlchat-warm-start").exists():
            _GLOBAL_HOLDER.load(source="checkpoints/pearlchat-warm-start", tokenizer_dir=tokenizer_dir)
        else:
            raise
    return _GLOBAL_HOLDER.model, _GLOBAL_HOLDER.tokenizer


def build_gradio_app(checkpoint_dir: str = "kambale/pearl-chat"):
    """Gradio fallback interface."""
    import gradio as gr
    model, tokenizer = load_model_and_tokenizer(checkpoint_dir=checkpoint_dir)

    def respond(message, chat_history, temperature, max_tokens):
        if not message.strip():
            yield "", chat_history
            return
        chat_history = list(chat_history) if chat_history else []
        is_tuple_format = len(chat_history) > 0 and isinstance(chat_history[0], (list, tuple))
        if is_tuple_format:
            chat_history.append((message, ""))
        else:
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": ""})
        yield "", chat_history

        accumulated = ""
        chat_prompt = message if "Omuntu:" in message else f"Omuntu: {message}\nOmuyambi: "
        for token in generate_stream(
            model=model,
            tokenizer=tokenizer,
            prompt=chat_prompt,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
        ):
            accumulated += token
            if is_tuple_format:
                chat_history[-1] = (message, accumulated)
            else:
                chat_history[-1]["content"] = accumulated
            yield "", chat_history

    with gr.Blocks(title="Pearl-Chat") as demo:
        gr.Markdown(
            "# Pearl-Chat\n"
            "Native Luganda language model implemented in pure JAX using Flax NNX."
        )
        chatbot = gr.Chatbot(label="Conversation", height=450)
        with gr.Row():
            msg = gr.Textbox(
                label="Prompt",
                placeholder="Enter prompt in Luganda (for example: Oli otya?)...",
                scale=8,
            )
            submit_btn = gr.Button("Send", scale=1)
        with gr.Accordion("Generation parameters", open=False):
            temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
            max_tokens = gr.Slider(16, 256, value=64, step=16, label="Maximum new tokens")

        submit_btn.click(respond, [msg, chatbot, temperature, max_tokens], [msg, chatbot])
        msg.submit(respond, [msg, chatbot, temperature, max_tokens], [msg, chatbot])
    return demo


def run_web_app(
    host: str = "127.0.0.1",
    port: int = 7860,
    checkpoint_dir: str = "kambale/pearl-chat",
    tokenizer_dir: str = "data/tokenizer",
) -> None:
    """Launch Pearl-Chat local web application."""
    print(f"Initializing Pearl-Chat with checkpoint from {checkpoint_dir}...")
    _GLOBAL_HOLDER.load(source=checkpoint_dir, tokenizer_dir=tokenizer_dir)

    server = HTTPServer((host, port), PearlChatRequestHandler)
    print(f"Pearl-Chat inference web console is active at http://{host}:{port}/")
    print("Press Ctrl+C to terminate.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web application.")
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pearl-Chat web application")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind")
    parser.add_argument(
        "--model",
        default="kambale/pearl-chat",
        help="Hugging Face repository ID (default: kambale/pearl-chat)",
    )
    parser.add_argument(
        "--gradio",
        action="store_true",
        help="Launch legacy Gradio UI instead of web console",
    )
    args = parser.parse_args()

    if args.gradio:
        demo = build_gradio_app(checkpoint_dir=args.model)
        demo.launch(server_name=args.host, server_port=args.port, share=False)
    else:
        run_web_app(
            host=args.host,
            port=args.port,
            checkpoint_dir=args.model,
        )


if __name__ == "__main__":
    main()
