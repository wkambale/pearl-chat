"""Training loop and state management with Flax NNX and Optax."""

import csv
from pathlib import Path
import time
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from pearlchat.config import ModelConfig, TrainConfig
from pearlchat.checkpointing import CheckpointManager
from pearlchat.model import PearlChatModel


def compute_loss(
    model: PearlChatModel,
    batch: dict[str, jax.Array],
    deterministic: bool = False,
) -> jax.Array:
    """Compute cross-entropy loss over sequence tokens."""
    inputs = batch["inputs"]
    targets = batch["targets"]
    logits, _ = model(inputs, deterministic=deterministic)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return jnp.mean(loss)


@nnx.jit
def train_step(
    model: PearlChatModel,
    optimizer: nnx.Optimizer,
    batch: dict[str, jax.Array],
) -> jax.Array:
    """Single JIT-compiled optimization step."""
    def loss_closure(m: PearlChatModel) -> jax.Array:
        return compute_loss(m, batch, deterministic=False)

    loss, grads = nnx.value_and_grad(loss_closure)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(
    model: PearlChatModel,
    batch: dict[str, jax.Array],
) -> jax.Array:
    """Single evaluation step without gradients."""
    return compute_loss(model, batch, deterministic=True)


def create_optimizer(
    model: PearlChatModel,
    config: TrainConfig,
) -> Tuple[nnx.Optimizer, Callable[[int], float]]:
    """Create learning rate schedule and Optax optimizer chain."""
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.total_steps,
        end_value=config.min_learning_rate,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay),
    )

    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    return optimizer, schedule


class TrainLogger:
    """CSV and console logger for training metrics."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "train_log.csv"

        fieldnames = ["step", "train_loss", "val_loss", "learning_rate", "tokens_per_sec"]
        with open(self.log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log(
        self,
        step: int,
        train_loss: float,
        learning_rate: float,
        tokens_per_sec: float,
        val_loss: Optional[float] = None,
    ) -> None:
        """Write metrics to console and CSV."""
        val_str = f" | val loss: {val_loss:.4f}" if val_loss is not None else ""
        print(
            f"step {step:4d} | train loss: {train_loss:.4f}{val_str} | "
            f"lr: {learning_rate:.6f} | tokens/sec: {tokens_per_sec:.1f}"
        )

        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                round(train_loss, 4),
                round(val_loss, 4) if val_loss is not None else "",
                f"{learning_rate:.6e}",
                round(tokens_per_sec, 1),
            ])


def run_training_loop(
    model: PearlChatModel,
    train_config: TrainConfig,
    train_loader: Any,
    val_loader: Optional[Any] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
    start_step: int = 0,
    target_steps: Optional[int] = None,
) -> float:
    """Execute training loop with logging and checkpointing."""
    optimizer, schedule = create_optimizer(model, train_config)
    logger = TrainLogger(Path(train_config.log_dir))

    max_steps = target_steps if target_steps is not None else train_config.total_steps
    train_iter = iter(train_loader)

    latest_loss = 0.0
    start_time = time.time()
    tokens_since_last_log = 0

    print(f"Beginning training from step {start_step} to step {max_steps}...")

    for step in range(start_step + 1, max_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch_inputs = jnp.asarray(batch["inputs"])
        batch_targets = jnp.asarray(batch["targets"])
        jax_batch = {"inputs": batch_inputs, "targets": batch_targets}

        loss = train_step(model, optimizer, jax_batch)
        latest_loss = float(loss)

        batch_tokens = int(batch_inputs.shape[0] * batch_inputs.shape[1])
        tokens_since_last_log += batch_tokens

        # Evaluation
        val_loss = None
        if val_loader is not None and (
            step % train_config.eval_every == 0 or step == max_steps
        ):
            val_losses = []
            val_iter = iter(val_loader)
            for _ in range(train_config.eval_steps):
                try:
                    v_batch = next(val_iter)
                except StopIteration:
                    break
                v_jax_batch = {
                    "inputs": jnp.asarray(v_batch["inputs"]),
                    "targets": jnp.asarray(v_batch["targets"]),
                }
                v_loss = eval_step(model, v_jax_batch)
                val_losses.append(float(v_loss))
            if val_losses:
                val_loss = sum(val_losses) / len(val_losses)

        # Logging
        if step % 5 == 0 or step == max_steps or step == start_step + 1:
            elapsed = max(time.time() - start_time, 1e-4)
            tokens_per_sec = tokens_since_last_log / elapsed
            lr = float(schedule(step))
            logger.log(
                step=step,
                train_loss=latest_loss,
                learning_rate=lr,
                tokens_per_sec=tokens_per_sec,
                val_loss=val_loss,
            )
            tokens_since_last_log = 0
            start_time = time.time()

        # Checkpointing
        if checkpoint_manager is not None and (
            step % train_config.checkpoint_every == 0 or step == max_steps
        ):
            checkpoint_manager.save_checkpoint(
                model=model,
                step=step,
                config=model.config,
                extra_metadata={"loss": latest_loss},
            )

    print(f"Training completed at step {max_steps} with final loss {latest_loss:.4f}.")
    return latest_loss
