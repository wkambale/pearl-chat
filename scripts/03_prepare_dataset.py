"""Prepare train and validation splits from the cleaned Luganda corpus."""

import argparse
import random
from pathlib import Path


def split_corpus(
    input_file: Path,
    train_file: Path,
    val_file: Path,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[int, int]:
    """Split cleaned corpus into deterministic train and validation files."""
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    rng = random.Random(seed)
    rng.shuffle(lines)

    val_count = max(1, int(len(lines) * val_ratio))
    val_lines = lines[:val_count]
    train_lines = lines[val_count:]

    train_file.parent.mkdir(parents=True, exist_ok=True)

    with open(train_file, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line + "\n")

    return len(train_lines), len(val_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset splits.")
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/processed/luganda_clean.txt",
        help="Path to cleaned corpus",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/processed/train.txt",
        help="Path for training split",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="data/processed/val.txt",
        help="Path for validation split",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Ratio of data reserved for validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Cleaned corpus not found at {input_path}. Run scripts/02_train_tokenizer.py first."
        )

    train_path = Path(args.train_file)
    val_path = Path(args.val_file)

    train_count, val_count = split_corpus(
        input_path, train_path, val_path, args.val_ratio, args.seed
    )
    print(f"Prepared dataset splits: {train_count} train lines, {val_count} val lines.")


if __name__ == "__main__":
    main()
