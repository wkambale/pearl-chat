"""Command line interface for Pearl-Chat model inference."""

import argparse
from pathlib import Path
import sys
from flax import nnx

from pearlchat.checkpointing import CheckpointManager
from pearlchat.config import ModelConfig
from pearlchat.generate import generate_stream
from pearlchat.model import PearlChatModel
from pearlchat.tokenizer import LugandaTokenizer


def run_cli_chat(
    checkpoint_dir: str = "checkpoints/pearlchat-warm-start",
    tokenizer_dir: str = "data/tokenizer",
    prompt: str = "Oli otya?",
    max_new_tokens: int = 64,
    temperature: float = 0.7,
) -> None:
    """Run interactive or single-prompt generation in the terminal."""
    tokenizer_path = Path(tokenizer_dir)
    if not (tokenizer_path / "vocab.json").exists():
        raise FileNotFoundError(
            f"Tokenizer files not found in {tokenizer_path}. Run scripts/02_train_tokenizer.py first."
        )

    tokenizer = LugandaTokenizer.load(tokenizer_path)
    print(f"Loaded Luganda tokenizer (vocab size: {tokenizer.vocab_size})")

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=128,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        feed_forward_dim=512,
        dropout_rate=0.0,
    )
    rngs = nnx.Rngs(params=42, dropout=42)
    model = PearlChatModel(config=model_config, rngs=rngs)

    checkpoint_path = Path(checkpoint_dir)
    if checkpoint_path.exists():
        manager = CheckpointManager(checkpoint_dir=checkpoint_path)
        try:
            model, step = manager.restore_latest_checkpoint(model)
            print(f"Restored model weights from step {step}.")
        except Exception as exc:
            print(f"Notice: Running with initialized weights ({exc}).")
    else:
        print("Notice: No checkpoint directory found, running with initialized weights.")

    print("\nPearl-Chat terminal interface")
    print("------------------------------------------------------------")
    print(f"Prompt: {prompt}")
    print("Response: ", end="", flush=True)

    for token_str in generate_stream(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    ):
        print(token_str, end="", flush=True)

    print("\n------------------------------------------------------------\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pearl-Chat terminal interface.")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/pearlchat-warm-start",
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default="data/tokenizer",
        help="Path to tokenizer directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Oli otya?",
        help="Initial prompt in Luganda",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    args = parser.parse_args()

    run_cli_chat(
        checkpoint_dir=args.checkpoint_dir,
        tokenizer_dir=args.tokenizer_dir,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temp,
    )


if __name__ == "__main__":
    main()
