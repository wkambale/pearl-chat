"""Clean Luganda raw corpus and train byte-level BPE tokenizer."""

import argparse
import html
import re
from pathlib import Path
from typing import List, Set

from pearlchat.tokenizer import LugandaTokenizer


ENGLISH_STOPWORDS = {
    "the", "and", "is", "are", "was", "were", "this", "that", "with",
    "from", "have", "has", "had", "they", "their", "which", "about",
}


def clean_line(line: str) -> str:
    """Strip HTML, markdown links, and normalize whitespace."""
    text = html.unescape(line)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_luganda(text: str) -> bool:
    """Filter out non-Luganda or contaminated lines."""
    words = text.lower().split()
    if len(words) < 3:
        return False
    # Check English stopword contamination
    english_matches = sum(1 for w in words if w in ENGLISH_STOPWORDS)
    if english_matches / len(words) > 0.25:
        return False
    return True


def preprocess_corpus(input_files: List[Path], output_file: Path) -> int:
    """Clean and deduplicate lines from input files."""
    seen_lines: Set[str] = set()
    total_written = 0

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as out_f:
        for file_path in input_files:
            if not file_path.exists():
                continue
            with open(file_path, "r", encoding="utf-8", errors="ignore") as in_f:
                for raw_line in in_f:
                    line = clean_line(raw_line)
                    if not line:
                        continue
                    if line in seen_lines:
                        continue
                    if not is_valid_luganda(line):
                        continue
                    seen_lines.add(line)
                    out_f.write(line + "\n")
                    total_written += 1

    return total_written


def run_tokenizer_comparison(tokenizer: LugandaTokenizer) -> None:
    """Print comparison between native tokenizer and generic byte tokenizer."""
    sample_sentence = (
        "Abayizi bagenda ku masomero okufuna amagezi n'obukugu mu bulamu bwabwe."
    )
    native_tokens = tokenizer.encode(sample_sentence)

    print("\nslide-ready tokenizer comparison:")
    print("------------------------------------------------------------")
    print(f"Sample sentence: {sample_sentence}")
    print(f"Character count: {len(sample_sentence)}")
    print(f"Native Luganda BPE token count: {len(native_tokens)}")
    print(f"Tokens: {[tokenizer.decode([t]) for t in native_tokens[:8]]}...")
    print("------------------------------------------------------------\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Luganda BPE tokenizer.")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Path to raw dataset directory",
    )
    parser.add_argument(
        "--processed-file",
        type=str,
        default="data/processed/luganda_clean.txt",
        help="Path to output cleaned corpus",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/tokenizer",
        help="Directory to save tokenizer files",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8192,
        help="Vocabulary size for the tokenizer",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    input_files = list(raw_path.glob("*.txt"))
    if not input_files:
        raise FileNotFoundError(
            f"No text files found in {raw_path}. Run scripts/01_fetch_data.py first."
        )

    processed_path = Path(args.processed_file)
    print(f"Cleaning raw text files from {raw_path}...")
    cleaned_lines = preprocess_corpus(input_files, processed_path)
    print(f"Cleaned corpus written to {processed_path} ({cleaned_lines} unique lines)")

    print(f"Training byte-level BPE tokenizer (vocab size {args.vocab_size})...")
    tokenizer = LugandaTokenizer.train_from_files(
        files=[str(processed_path)],
        vocab_size=args.vocab_size,
        min_frequency=2,
    )

    out_dir = Path(args.output_dir)
    tokenizer.save(out_dir)
    print(f"Tokenizer saved to {out_dir} (vocab_size={tokenizer.vocab_size})")

    run_tokenizer_comparison(tokenizer)


if __name__ == "__main__":
    main()
