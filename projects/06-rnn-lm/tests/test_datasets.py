"""Encoder round-trip and sequence sampling tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_PROJECT_DIR = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rnn_datasets", _PROJECT_DIR / "datasets.py")
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
sys.modules["rnn_datasets"] = mod
_spec.loader.exec_module(mod)

TextEncoder = mod.TextEncoder
sample_sequence = mod.sample_sequence
CORPUS = mod.CORPUS


class TestCorpus:
    def test_corpus_is_non_empty_shakespeare(self) -> None:
        # Assert
        assert isinstance(CORPUS, str)
        assert len(CORPUS) > 1000, "corpus should be at least ~1KB"


class TestTextEncoder:
    def test_vocab_covers_every_character_in_corpus(self) -> None:
        # Arrange
        text = "hello world\nthe quick brown fox"
        enc = TextEncoder(text)

        # Act / Assert
        assert enc.vocab_size == len(set(text))

    def test_encode_decode_roundtrip(self) -> None:
        # Arrange
        text = "hello world"
        enc = TextEncoder(text)

        # Act
        ids = enc.encode(text)
        recovered = enc.decode(ids)

        # Assert
        assert ids.dtype.kind in ("i", "u")
        assert ids.shape == (len(text),)
        assert recovered == text

    def test_encode_rejects_unseen_character(self) -> None:
        # Arrange
        enc = TextEncoder("abc")

        # Act / Assert
        try:
            enc.encode("z")
            raise AssertionError("expected error on unseen char")
        except (AssertionError, KeyError, ValueError):
            pass


class TestSampleSequence:
    def test_returns_input_and_target_shifted_by_one(self) -> None:
        # Arrange
        ids = np.arange(20, dtype=np.int64)

        # Act
        inputs, targets = sample_sequence(ids, seq_len=8, seed=0)

        # Assert — targets are inputs shifted by one position
        assert inputs.shape == (8,)
        assert targets.shape == (8,)
        assert np.array_equal(targets[:-1], inputs[1:])
