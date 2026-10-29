"""LSTM (or GRU) character language model, PyTorch.

Same task as `rnn_numpy.py` — predict the next character given the previous
one and a running hidden state — but with two upgrades:

1. **Learned embeddings** (`nn.Embedding`) instead of one-hot inputs. Each
   character id maps to a dense vector; the network learns which characters
   should sit near each other in embedding space.
2. **Gated recurrent cells** (`nn.LSTM` or `nn.GRU`) instead of vanilla tanh
   recurrence. Gates give the network explicit knobs to control what to
   remember (forget gate close to 1 = keep) and what to add (input gate).
   The resulting straight-line gradient path through the cell state is what
   makes LSTMs learn long-range dependencies where vanilla RNNs can't.

Compare parameter counts: for the same `hidden_size`, LSTM has ~4x the
recurrent-cell parameters of a vanilla RNN and ~1.33x the parameters of a
GRU (which uses 3 gates instead of 4). The extra parameters buy that
straight-line gradient path.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class CharLM:
    """Learned-embedding + LSTM/GRU + linear head.

    Attributes:
        vocab_size: V — number of possible input/output tokens.
        hidden_size: H — dimension of recurrent hidden state.
        cell_type: "lstm" or "gru".
        n_layers: Number of stacked recurrent layers.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 128,
        cell_type: str = "lstm",
        n_layers: int = 1,
        learning_rate: float = 0.003,
        seed: int = 42,
    ) -> None:
        assert vocab_size > 0, f"vocab_size must be positive, got {vocab_size}"
        assert hidden_size > 0, f"hidden_size must be positive, got {hidden_size}"
        assert n_layers > 0, f"n_layers must be positive, got {n_layers}"
        assert learning_rate > 0, f"learning_rate must be positive, got {learning_rate}"
        assert cell_type in ("lstm", "gru"), f"cell_type must be 'lstm' or 'gru', got {cell_type!r}"

        torch.manual_seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.n_layers = n_layers

        cell_cls = nn.LSTM if cell_type == "lstm" else nn.GRU
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = cell_cls(hidden_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

        self._parameters = (
            list(self.embedding.parameters())
            + list(self.rnn.parameters())
            + list(self.head.parameters())
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self._parameters, lr=learning_rate)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self._parameters)

    def _forward(
        self, inputs: torch.Tensor, hidden: object | None = None
    ) -> tuple[torch.Tensor, object]:
        """Forward pass. inputs: (batch, T). Returns logits (batch, T, V) and new hidden."""
        assert inputs.dim() == 2, f"expected (batch, T), got {tuple(inputs.shape)}"
        emb = self.embedding(inputs)  # (batch, T, H)
        out, hidden = self.rnn(emb, hidden)  # (batch, T, H), new hidden state
        logits = self.head(out)  # (batch, T, V)
        return logits, hidden

    def fit_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """One optimizer step. inputs/targets: (batch, T). Returns scalar loss."""
        assert inputs.shape == targets.shape, (
            f"inputs {tuple(inputs.shape)} vs targets {tuple(targets.shape)}"
        )
        for m in (self.embedding, self.rnn, self.head):
            m.train()
        self.optimizer.zero_grad()
        logits, _ = self._forward(inputs)
        # CrossEntropyLoss wants (N, C), so flatten batch and time
        loss = self.loss_fn(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._parameters, max_norm=5.0)
        self.optimizer.step()
        return float(loss.item())

    def sample(self, seed_id: int, length: int, temperature: float = 1.0) -> np.ndarray:
        """Autoregressive sampling, temperature-scaled softmax."""
        assert 0 <= seed_id < self.vocab_size, (
            f"seed_id {seed_id} out of range [0, {self.vocab_size})"
        )
        assert length > 0, f"length must be positive, got {length}"
        assert temperature > 0, f"temperature must be positive, got {temperature}"

        for m in (self.embedding, self.rnn, self.head):
            m.eval()
        ids = np.zeros(length, dtype=np.int64)
        current = torch.tensor([[seed_id]], dtype=torch.long)
        hidden: object | None = None
        with torch.no_grad():
            for t in range(length):
                logits, hidden = self._forward(current, hidden)
                scaled = logits[0, -1] / temperature
                probs = torch.softmax(scaled, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
                ids[t] = next_id
                current = torch.tensor([[next_id]], dtype=torch.long)
        return ids
