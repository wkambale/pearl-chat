"""Main training script for Pearl-Chat model."""

import argparse
from pathlib import Path

from flax import nnx
import jax

from pearlchat.checkpointing import CheckpointManager
from pearlchat.config import DataConfig, ModelConfig, TrainConfig
from pearlchat.data_pipeline import LugandaTextDataSource, create_data_loader
from pearlchat.model import PearlChatModel
from pearlchat.tokenizer import LugandaTokenizer
from pearlchat.train import run_training_loop


def train_pearlchat(
    mode: str = "full",
    total_steps: int = 500,
    live_steps: int = 30,
    batch_size: int = 16,
    learning_rate: float = 5e-4,
    checkpoint_dir: str = "checkpoints/pearlchat-warm-start",
    data_dir: str = "data/processed",
    tokenizer_dir: str = "data/tokenizer",
) -> None:
    """Run Pearl-Chat training in full or live workshop mode."""
    print(f"JAX backend: {jax.default_backend()} | Devices: {jax.devices()}")

    tokenizer_path = Path(tokenizer_dir)
    if not (tokenizer_path / "vocab.json").exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. Run scripts/02_train_tokenizer.py first."
        )
    tokenizer = LugandaTokenizer.load(tokenizer_path)
    print(f"Loaded tokenizer with vocabulary size: {tokenizer.vocab_size}")

    train_file = Path(data_dir) / "train.txt"
    val_file = Path(data_dir) / "val.txt"
    if not train_file.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_file}. Run scripts/03_prepare_dataset.py first."
        )

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=128,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        feed_forward_dim=512,
        dropout_rate=0.1,
    )

    train_config = TrainConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        total_steps=total_steps,
        checkpoint_dir=checkpoint_dir,
        live_steps=live_steps,
    )

    print("Building Grain data loaders...")
    train_source = LugandaTextDataSource.from_file(
        train_file,
        tokenizer=tokenizer,
        context_length=model_config.context_length,
    )
    train_loader = create_data_loader(
        train_source,
        batch_size=train_config.batch_size,
        seed=train_config.seed,
        shuffle=True,
    )

    val_loader = None
    if val_file.exists():
        val_source = LugandaTextDataSource.from_file(
            val_file,
            tokenizer=tokenizer,
            context_length=model_config.context_length,
        )
        if len(val_source) > 0:
            val_loader = create_data_loader(
                val_source,
                batch_size=train_config.batch_size,
                seed=train_config.seed + 1,
                shuffle=False,
            )

    rngs = nnx.Rngs(params=train_config.seed, dropout=train_config.seed + 1)
    model = PearlChatModel(config=model_config, rngs=rngs)
    param_count = model.count_parameters()
    print(f"Initialized Pearl-Chat model with {param_count:,} parameters.")

    checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

    if mode == "live":
        print(f"Restoring warm-start checkpoint from {checkpoint_dir}...")
        model, start_step = checkpoint_manager.restore_latest_checkpoint(model)
        print(f"Resumed from step {start_step}. Continuing live training for {live_steps} steps...")
        target_steps = start_step + live_steps
        train_config.total_steps = target_steps
        run_training_loop(
            model=model,
            train_config=train_config,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_manager=checkpoint_manager,
            start_step=start_step,
            target_steps=target_steps,
        )
    else:
        print(f"Starting full training from random initialization for {total_steps} steps...")
        run_training_loop(
            model=model,
            train_config=train_config,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_manager=checkpoint_manager,
            start_step=0,
            target_steps=total_steps,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Pearl-Chat language model.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "live"],
        default="full",
        help="Training mode: 'full' for pre-training or 'live' for workshop continuation",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Total training steps for full mode",
    )
    parser.add_argument(
        "--live-steps",
        type=int,
        default=25,
        help="Number of steps for live workshop demo mode",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/pearlchat-warm-start",
        help="Directory to save or restore model checkpoints",
    )
    args = parser.parse_args()

    train_pearlchat(
        mode=args.mode,
        total_steps=args.steps,
        live_steps=args.live_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
