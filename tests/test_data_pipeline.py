"""Tests for Grain data pipeline determinism, shape, dtype, and checkpointing."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from pearlchat.tokenizer import LugandaTokenizer
from pearlchat.data_pipeline import LugandaTextDataSource, create_data_loader


SAMPLE_TEXTS = [
    "Oli otya muganda wange, olwa leero tulina emirimu mingi gye tulina okukola.",
    "Uganda nsi nungi nnyo eri wakati mu buvanjuba bwa ssemazinga wa Afirika.",
    "Abalimi mu bitundu by'ebyalo bakola nnyo okulima emwanyi n'amatooke agagulibwa mu katale.",
    "Abaana b'amasomero basanyukira nnyo okuyiga kompyuta n'amagezi ag'ebweru.",
    "Kampala kye kibuga ekisinga obunene era kye kikulu mu ggwanga lyaffe erya Uganda.",
]


@pytest.fixture
def test_tokenizer():
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = Path(tmpdir) / "corpus.txt"
        corpus_path.write_text("\n".join(SAMPLE_TEXTS), encoding="utf-8")
        tokenizer = LugandaTokenizer.train_from_files(
            files=[str(corpus_path)],
            vocab_size=256 + 30,
            min_frequency=1,
        )
        yield tokenizer


def test_data_source_shapes_and_dtypes(test_tokenizer):
    context_length = 16
    data_source = LugandaTextDataSource(
        texts=SAMPLE_TEXTS,
        tokenizer=test_tokenizer,
        context_length=context_length,
    )
    assert len(data_source) > 0
    sample = data_source[0]
    assert "inputs" in sample and "targets" in sample
    assert sample["inputs"].shape == (context_length,)
    assert sample["targets"].shape == (context_length,)
    assert sample["inputs"].dtype == np.int32
    assert sample["targets"].dtype == np.int32


def test_data_loader_batching(test_tokenizer):
    context_length = 8
    batch_size = 2
    data_source = LugandaTextDataSource(
        texts=SAMPLE_TEXTS * 5,
        tokenizer=test_tokenizer,
        context_length=context_length,
    )
    loader = create_data_loader(
        data_source=data_source,
        batch_size=batch_size,
        seed=42,
        shuffle=False,
        num_epochs=1,
    )
    batch = next(iter(loader))
    assert batch["inputs"].shape == (batch_size, context_length)
    assert batch["targets"].shape == (batch_size, context_length)


def test_pipeline_checkpoint_resumption(test_tokenizer):
    context_length = 8
    batch_size = 2
    data_source = LugandaTextDataSource(
        texts=SAMPLE_TEXTS * 6,
        tokenizer=test_tokenizer,
        context_length=context_length,
    )
    loader = create_data_loader(
        data_source=data_source,
        batch_size=batch_size,
        seed=123,
        shuffle=True,
        num_epochs=1,
    )

    # Full run
    full_batches = list(loader)
    assert len(full_batches) >= 3

    # Interrupted run with checkpointing
    it1 = iter(loader)
    _ = next(it1)
    state = it1.get_state()
    batch_2_uninterrupted = next(it1)

    # Resume on a new iterator
    it2 = iter(loader)
    it2.set_state(state)
    batch_2_resumed = next(it2)

    np.testing.assert_array_equal(
        batch_2_uninterrupted["inputs"],
        batch_2_resumed["inputs"],
        err_msg="Resumed inputs do not match uninterrupted run.",
    )
    np.testing.assert_array_equal(
        batch_2_uninterrupted["targets"],
        batch_2_resumed["targets"],
        err_msg="Resumed targets do not match uninterrupted run.",
    )
