"""Data pipeline using Grain for deterministic, resumable batch loading."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import grain.python as grain
import numpy as np

from pearlchat.tokenizer import LugandaTokenizer


class LugandaTextDataSource(grain.RandomAccessDataSource):
    """Grain data source that chunks tokenized Luganda text into fixed sequences."""

    def __init__(
        self,
        texts: List[str],
        tokenizer: LugandaTokenizer,
        context_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.chunks = self._pack_and_chunk_texts(texts)

    def _pack_and_chunk_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Tokenize texts, concatenate with EOS, and split into fixed length chunks."""
        token_stream: List[int] = []
        eos_id = self.tokenizer.eos_id

        for text in texts:
            clean = text.strip()
            if not clean:
                continue
            tokens = self.tokenizer.encode(clean)
            if tokens:
                token_stream.extend(tokens)
                token_stream.append(eos_id)

        seq_len = self.context_length + 1
        num_chunks = len(token_stream) // seq_len

        chunks = []
        for i in range(num_chunks):
            start_idx = i * seq_len
            end_idx = start_idx + seq_len
            chunk = np.array(token_stream[start_idx:end_idx], dtype=np.int32)
            chunks.append(chunk)

        return chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, record_key: int) -> Dict[str, np.ndarray]:
        chunk = self.chunks[record_key]
        inputs = chunk[:-1]
        targets = chunk[1:]
        return {
            "inputs": inputs,
            "targets": targets,
        }

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        tokenizer: LugandaTokenizer,
        context_length: int = 256,
    ) -> "LugandaTextDataSource":
        """Load text from a file and build data source."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
        return cls(texts=lines, tokenizer=tokenizer, context_length=context_length)


def create_data_loader(
    data_source: grain.RandomAccessDataSource,
    batch_size: int = 16,
    seed: int = 42,
    shuffle: bool = True,
    num_epochs: Optional[int] = None,
    drop_remainder: bool = True,
) -> grain.DataLoader:
    """Create a Grain DataLoader with IndexSampler and batching."""
    sampler = grain.IndexSampler(
        num_records=len(data_source),
        shuffle=shuffle,
        seed=seed,
        shard_options=grain.NoSharding(),
        num_epochs=num_epochs,
    )

    operations = [
        grain.Batch(batch_size=batch_size, drop_remainder=drop_remainder),
    ]

    return grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=operations,
    )
