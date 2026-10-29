"""Black-box tests for the NumPy RNN + hand-derived BPTT.

The load-bearing test here is `test_analytical_gradient_matches_numerical` —
BPTT is error-prone to derive; if analytical gradients don't match finite
differences to ~1e-3, backprop through time is wrong and any training
progress is coincidence.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_PROJECT_DIR = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rnn_numpy", _PROJECT_DIR / "rnn_numpy.py")
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
sys.modules["rnn_numpy"] = mod
_spec.loader.exec_module(mod)

VanillaRNN = mod.VanillaRNN


class TestForwardShapes:
    def test_hidden_state_and_logits_have_expected_shape(self) -> None:
        # Arrange
        rnn = VanillaRNN(vocab_size=10, hidden_size=8, seed=0)
        inputs = np.array([0, 1, 2, 3], dtype=np.int64)
        h_prev = np.zeros((8,), dtype=np.float64)

        # Act
        xs, hs, ys, ps = rnn._forward(inputs, h_prev)

        # Assert
        assert xs.shape == (4, 10), "one-hot inputs (T, V)"
        assert hs.shape == (4, 8), "hidden states (T, H)"
        assert ys.shape == (4, 10), "output logits (T, V)"
        assert ps.shape == (4, 10), "softmax probs (T, V)"
        np.testing.assert_allclose(ps.sum(axis=1), 1.0, rtol=1e-6)


class TestGradientCorrectness:
    """Load-bearing: BPTT gradients must match finite-difference numerical gradients."""

    def test_analytical_gradient_matches_numerical(self) -> None:
        # Arrange
        rnn = VanillaRNN(vocab_size=5, hidden_size=4, seed=0)
        inputs = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        targets = np.array([1, 2, 3, 4, 0], dtype=np.int64)
        h_prev = np.zeros((4,), dtype=np.float64)

        # Act
        grads, _ = rnn._compute_gradients(inputs, targets, h_prev)

        # Assert — check every parameter matrix/vector
        eps = 1e-5
        max_checks_per_param = 5
        for name in ("W_xh", "W_hh", "W_hy", "b_h", "b_y"):
            param = getattr(rnn, name)
            g = grads[name]
            it = np.nditer(param, flags=["multi_index"])
            for checked, _ in enumerate(it):
                if checked >= max_checks_per_param:
                    break
                idx = it.multi_index
                original = param[idx]
                param[idx] = original + eps
                loss_plus = rnn._loss(inputs, targets, h_prev)
                param[idx] = original - eps
                loss_minus = rnn._loss(inputs, targets, h_prev)
                param[idx] = original

                numerical = (loss_plus - loss_minus) / (2 * eps)
                np.testing.assert_allclose(
                    g[idx],
                    numerical,
                    rtol=1e-3,
                    atol=1e-5,
                    err_msg=f"grad mismatch at {name}{idx}",
                )


class TestTraining:
    def test_loss_decreases_across_steps(self) -> None:
        # Arrange
        rnn = VanillaRNN(vocab_size=8, hidden_size=32, learning_rate=0.1, seed=0)
        rng = np.random.default_rng(0)
        ids = rng.integers(0, 8, size=200, dtype=np.int64)
        seq_len = 16

        first_losses: list[float] = []
        last_losses: list[float] = []
        h = np.zeros(32, dtype=np.float64)

        # Act — 200 steps
        for step in range(200):
            start = step % (len(ids) - seq_len - 1)
            inputs = ids[start : start + seq_len]
            targets = ids[start + 1 : start + 1 + seq_len]
            loss, h = rnn.fit_step(inputs, targets, h)
            if step < 5:
                first_losses.append(loss)
            if step >= 195:
                last_losses.append(loss)

        # Assert
        assert np.mean(last_losses) < np.mean(first_losses), (
            f"loss should decrease: {np.mean(first_losses):.3f} -> {np.mean(last_losses):.3f}"
        )


class TestGeneration:
    def test_sample_returns_valid_ids_of_requested_length(self) -> None:
        # Arrange
        rnn = VanillaRNN(vocab_size=5, hidden_size=8, seed=0)

        # Act
        ids = rnn.sample(seed_id=0, length=25)

        # Assert
        assert ids.shape == (25,)
        assert ids.dtype.kind in ("i", "u")
        assert ids.min() >= 0
        assert ids.max() < 5
