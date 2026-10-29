"""Vanilla RNN with hand-derived Backprop Through Time (BPTT), pure NumPy.

At each time step t:

    h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)      # hidden state
    y_t = W_hy @ h_t + b_y                              # logits
    p_t = softmax(y_t)                                  # next-char distribution

Loss = sum_t cross_entropy(p_t, target_t)

BPTT unrolls the graph for T time steps. The gradient of L w.r.t. W_hh
involves a chain of T Jacobians of tanh, which is exactly why vanilla RNNs
have the vanishing / exploding gradient problem: if the largest eigenvalue
of W_hh times sech^2(z_t) is < 1, the product shrinks toward zero and the
network can't learn dependencies across many time steps. LSTMs fix this by
introducing a straight-line gradient path through the cell state, gated by
sigmoids the network learns to open and close.

We use gradient clipping (norm-based, threshold 5.0) as a cheap safeguard
against the exploding side. There is no cheap safeguard against vanishing;
that's what the LSTM in `lstm_pytorch.py` is for.
"""

from __future__ import annotations

import numpy as np


class VanillaRNN:
    """Character-level vanilla RNN. Hand-derived BPTT, Adagrad updates.

    Attributes:
        vocab_size: V — number of possible input/output tokens.
        hidden_size: H — dimension of hidden state.
        W_xh: (H, V) input-to-hidden weight.
        W_hh: (H, H) hidden-to-hidden weight.
        W_hy: (V, H) hidden-to-output weight.
        b_h, b_y: bias vectors.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        learning_rate: float = 0.1,
        seed: int = 42,
    ) -> None:
        assert vocab_size > 0, f"vocab_size must be positive, got {vocab_size}"
        assert hidden_size > 0, f"hidden_size must be positive, got {hidden_size}"
        assert learning_rate > 0, f"learning_rate must be positive, got {learning_rate}"

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        rng = np.random.default_rng(seed)
        scale = 0.01
        self.W_xh = rng.normal(0.0, scale, size=(hidden_size, vocab_size))
        self.W_hh = rng.normal(0.0, scale, size=(hidden_size, hidden_size))
        self.W_hy = rng.normal(0.0, scale, size=(vocab_size, hidden_size))
        self.b_h = np.zeros(hidden_size, dtype=np.float64)
        self.b_y = np.zeros(vocab_size, dtype=np.float64)

        # Adagrad memory (per-parameter squared-gradient accumulator).
        self._m_W_xh = np.zeros_like(self.W_xh)
        self._m_W_hh = np.zeros_like(self.W_hh)
        self._m_W_hy = np.zeros_like(self.W_hy)
        self._m_b_h = np.zeros_like(self.b_h)
        self._m_b_y = np.zeros_like(self.b_y)

    def _forward(
        self, inputs: np.ndarray, h_prev: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Roll the RNN forward. Returns (one-hot xs, hidden states, logits, probs)."""
        assert inputs.ndim == 1
        T = inputs.shape[0]
        xs = np.zeros((T, self.vocab_size), dtype=np.float64)
        hs = np.zeros((T, self.hidden_size), dtype=np.float64)
        ys = np.zeros((T, self.vocab_size), dtype=np.float64)
        ps = np.zeros((T, self.vocab_size), dtype=np.float64)

        h = h_prev
        for t in range(T):
            xs[t, int(inputs[t])] = 1.0
            h = np.tanh(self.W_xh @ xs[t] + self.W_hh @ h + self.b_h)
            hs[t] = h
            ys[t] = self.W_hy @ h + self.b_y
            ys_shifted = ys[t] - ys[t].max()
            exp_y = np.exp(ys_shifted)
            ps[t] = exp_y / exp_y.sum()
        return xs, hs, ys, ps

    def _loss(self, inputs: np.ndarray, targets: np.ndarray, h_prev: np.ndarray) -> float:
        assert inputs.shape == targets.shape, f"inputs {inputs.shape} vs targets {targets.shape}"
        _, _, _, ps = self._forward(inputs, h_prev)
        eps = 1e-12
        picked = ps[np.arange(len(targets)), targets]
        return float(-np.log(np.clip(picked, eps, 1.0)).sum())

    def _compute_gradients(
        self, inputs: np.ndarray, targets: np.ndarray, h_prev: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """BPTT: return (grads dict, final hidden state)."""
        assert inputs.shape == targets.shape
        T = inputs.shape[0]
        xs, hs, _, ps = self._forward(inputs, h_prev)

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        dh_next = np.zeros_like(h_prev)

        for t in reversed(range(T)):
            dy = ps[t].copy()
            dy[int(targets[t])] -= 1.0  # softmax + CE grad

            dW_hy += np.outer(dy, hs[t])
            db_y += dy

            dh = self.W_hy.T @ dy + dh_next
            dh_raw = (1.0 - hs[t] * hs[t]) * dh  # tanh derivative

            db_h += dh_raw
            dW_xh += np.outer(dh_raw, xs[t])
            h_prev_step = hs[t - 1] if t > 0 else h_prev
            dW_hh += np.outer(dh_raw, h_prev_step)
            dh_next = self.W_hh.T @ dh_raw

        grads = {
            "W_xh": dW_xh,
            "W_hh": dW_hh,
            "W_hy": dW_hy,
            "b_h": db_h,
            "b_y": db_y,
        }
        return grads, hs[-1].copy()

    def fit_step(
        self, inputs: np.ndarray, targets: np.ndarray, h_prev: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """One BPTT step + Adagrad update. Returns (loss, final hidden state)."""
        loss = self._loss(inputs, targets, h_prev)
        grads, h_last = self._compute_gradients(inputs, targets, h_prev)

        # Gradient clipping (exploding-gradient guard).
        clip = 5.0
        for g in grads.values():
            np.clip(g, -clip, clip, out=g)

        # Adagrad: element-wise adaptive learning rate.
        mem_map = {
            "W_xh": self._m_W_xh,
            "W_hh": self._m_W_hh,
            "W_hy": self._m_W_hy,
            "b_h": self._m_b_h,
            "b_y": self._m_b_y,
        }
        for name, g in grads.items():
            mem = mem_map[name]
            mem += g * g
            param = getattr(self, name)
            param -= self.learning_rate * g / (np.sqrt(mem) + 1e-8)
        return loss, h_last

    def sample(self, seed_id: int, length: int, h_prev: np.ndarray | None = None) -> np.ndarray:
        """Generate `length` character ids by sampling from the model."""
        assert 0 <= seed_id < self.vocab_size, (
            f"seed_id {seed_id} out of range [0, {self.vocab_size})"
        )
        assert length > 0, f"length must be positive, got {length}"

        h = h_prev if h_prev is not None else np.zeros(self.hidden_size, dtype=np.float64)
        x = np.zeros(self.vocab_size, dtype=np.float64)
        x[seed_id] = 1.0
        ids = np.zeros(length, dtype=np.int64)
        rng = np.random.default_rng(0)
        for t in range(length):
            h = np.tanh(self.W_xh @ x + self.W_hh @ h + self.b_h)
            y = self.W_hy @ h + self.b_y
            y_shifted = y - y.max()
            p = np.exp(y_shifted)
            p /= p.sum()
            next_id = int(rng.choice(self.vocab_size, p=p))
            ids[t] = next_id
            x = np.zeros(self.vocab_size, dtype=np.float64)
            x[next_id] = 1.0
        return ids
