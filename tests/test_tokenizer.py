"""Unit tests for LugandaTokenizer."""

import tempfile
from pathlib import Path
import pytest
from pearlchat.tokenizer import LugandaTokenizer


SAMPLE_CORPUS = """Oli otya? Gyendi, webale nnyo.
Uganda nsi nungi eri mu buvanjuba bwa Afirika.
Kampala kye kibuga ekikulu ekya Uganda.
Abayizi bagenda ku masomero okufuna amagezi n'obukugu.
"""


@pytest.fixture
def trained_tokenizer():
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = Path(tmpdir) / "sample.txt"
        corpus_path.write_text(SAMPLE_CORPUS, encoding="utf-8")
        tokenizer = LugandaTokenizer.train_from_files(
            files=[str(corpus_path)],
            vocab_size=256 + 20,
            min_frequency=1,
        )
        tokenizer_dir = Path(tmpdir) / "tokenizer"
        tokenizer.save(tokenizer_dir)
        yield LugandaTokenizer.load(tokenizer_dir)


def test_round_trip_encoding(trained_tokenizer):
    test_sentence = "Uganda nsi nungi"
    token_ids = trained_tokenizer.encode(test_sentence)
    assert len(token_ids) > 0
    decoded_text = trained_tokenizer.decode(token_ids)
    assert decoded_text == test_sentence


def test_special_tokens(trained_tokenizer):
    assert trained_tokenizer.eos_id is not None
    assert trained_tokenizer.pad_id is not None
    assert trained_tokenizer.vocab_size >= 256


def test_common_words_preservation(trained_tokenizer):
    words = ["Oli", "otya", "Uganda", "Kampala"]
    for word in words:
        tokens = trained_tokenizer.encode(word)
        decoded = trained_tokenizer.decode(tokens)
        assert decoded == word
