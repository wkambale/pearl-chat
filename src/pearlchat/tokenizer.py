"""Byte-level BPE tokenizer for Luganda text."""

import os
from pathlib import Path
from typing import List, Optional, Union

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import ByteLevel


SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|pad|>",
    "<|unk|>",
]


class LugandaTokenizer:
    """Byte-level byte-pair encoding tokenizer for Luganda."""

    def __init__(self, tokenizer: Optional[ByteLevelBPETokenizer] = None):
        self.tokenizer = tokenizer
        self._eos_token = "<|endoftext|>"
        self._pad_token = "<|pad|>"
        self._unk_token = "<|unk|>"

    @classmethod
    def train_from_files(
        cls,
        files: List[str],
        vocab_size: int = 8192,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
    ) -> "LugandaTokenizer":
        """Train a byte-level BPE tokenizer on text files."""
        if special_tokens is None:
            special_tokens = SPECIAL_TOKENS

        raw_tokenizer = ByteLevelBPETokenizer(lowercase=False)
        raw_tokenizer.train(
            files=files,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
        )
        raw_tokenizer.post_processor = ByteLevel(trim_offsets=False)
        return cls(tokenizer=raw_tokenizer)

    def save(self, output_dir: Union[str, Path]) -> None:
        """Save vocab.json and merges.txt to directory."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been initialized or trained.")
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_model(str(out_path))

    @classmethod
    def load(cls, tokenizer_dir: Union[str, Path]) -> "LugandaTokenizer":
        """Load tokenizer from directory containing vocab.json and merges.txt."""
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
        """Encode text to token ids."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decode token ids back to text."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    @property
    def eos_id(self) -> int:
        """End of sequence token id."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")
        token_id = self.tokenizer.token_to_id(self._eos_token)
        return token_id if token_id is not None else 0

    @property
    def pad_id(self) -> int:
        """Padding token id."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")
        token_id = self.tokenizer.token_to_id(self._pad_token)
        return token_id if token_id is not None else self.eos_id

    @property
    def unk_id(self) -> int:
        """Unknown token id."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")
        token_id = self.tokenizer.token_to_id(self._unk_token)
        return token_id if token_id is not None else 2

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")
        return self.tokenizer.get_vocab_size()
