"""Behavioral tests for the PyTorch LSTM/GRU character LM."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_PROJECT_DIR = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("lstm_pt", _PROJECT_DIR / "lstm_pytorch.py")
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
sys.modules["lstm_pt"] = mod
_spec.loader.exec_module(mod)

CharLM = mod.CharLM


class TestArchitecture:
    def test_lstm_has_parameters(self) -> None:
        model = CharLM(vocab_size=32, hidden_size=64, cell_type="lstm", seed=0)
        assert model.n_parameters() > 0

    def test_gru_has_parameters(self) -> None:
        model = CharLM(vocab_size=32, hidden_size=64, cell_type="gru", seed=0)
        assert model.n_parameters() > 0

    def test_lstm_has_more_params_than_gru_same_hidden(self) -> None:
        # LSTM has 4 gates, GRU has 3 (or 2, depending on formulation).
        # PyTorch's LSTM: 4*H*(H+I+2), GRU: 3*H*(H+I+2). LSTM should be larger.
        lstm = CharLM(vocab_size=32, hidden_size=64, cell_type="lstm", seed=0)
        gru = CharLM(vocab_size=32, hidden_size=64, cell_type="gru", seed=0)
        assert lstm.n_parameters() > gru.n_parameters()


class TestTraining:
    @pytest.mark.parametrize("cell_type", ["lstm", "gru"])
    def test_loss_decreases_across_steps(self, cell_type: str) -> None:
        # Arrange
        torch.manual_seed(0)
        model = CharLM(
            vocab_size=8, hidden_size=32, cell_type=cell_type, learning_rate=0.01, seed=0
        )
        rng = np.random.default_rng(0)
        ids = torch.from_numpy(rng.integers(0, 8, size=200, dtype=np.int64))
        seq_len = 16

        first_losses: list[float] = []
        last_losses: list[float] = []

        # Act — 100 steps
        for step in range(100):
            start = step % (len(ids) - seq_len - 1)
            inputs = ids[start : start + seq_len].unsqueeze(0)  # (1, T)
            targets = ids[start + 1 : start + 1 + seq_len].unsqueeze(0)
            loss = model.fit_step(inputs, targets)
            if step < 5:
                first_losses.append(loss)
            if step >= 95:
                last_losses.append(loss)

        # Assert
        assert np.mean(last_losses) < np.mean(first_losses), (
            f"[{cell_type}] loss should decrease: "
            f"{np.mean(first_losses):.3f} -> {np.mean(last_losses):.3f}"
        )


class TestGeneration:
    def test_sample_returns_valid_ids_of_requested_length(self) -> None:
        # Arrange
        model = CharLM(vocab_size=5, hidden_size=16, cell_type="lstm", seed=0)

        # Act
        ids = model.sample(seed_id=0, length=25)

        # Assert
        assert ids.shape == (25,)
        assert ids.dtype.kind in ("i", "u")
        assert ids.min() >= 0
        assert ids.max() < 5

    def test_temperature_affects_sampling_diversity(self) -> None:
        """High temperature = more diverse; temperature=0.0001 ~= argmax (repetitive)."""
        # Arrange
        model = CharLM(vocab_size=10, hidden_size=32, cell_type="lstm", seed=0)

        # Act
        greedy = model.sample(seed_id=0, length=50, temperature=0.0001)
        diverse = model.sample(seed_id=0, length=50, temperature=2.0)

        # Assert — diverse sample should use more unique tokens
        assert len(set(diverse.tolist())) >= len(set(greedy.tolist()))


class TestPreconditions:
    def test_rejects_unknown_cell_type(self) -> None:
        with pytest.raises((AssertionError, ValueError)):
            CharLM(vocab_size=32, hidden_size=64, cell_type="transformer", seed=0)

    def test_rejects_non_positive_hidden_size(self) -> None:
        with pytest.raises(AssertionError):
            CharLM(vocab_size=32, hidden_size=0, cell_type="lstm", seed=0)
