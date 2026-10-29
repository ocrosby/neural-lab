"""Demo: train the NumPy RNN and PyTorch LSTM on the same Shakespeare snippet.

Prints running loss and, at the end, a sample of generated text from each
model. The LSTM should reach lower loss with less noise; the RNN's samples
often plateau at high loss and produce very repetitive character patterns —
the vanishing-gradient story made visible.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).parent


def _load(name: str, path: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


datasets = _load("rnn_datasets", _HERE / "datasets.py")
rnn_numpy = _load("rnn_numpy", _HERE / "rnn_numpy.py")
lstm_pytorch = _load("lstm_pt", _HERE / "lstm_pytorch.py")

TextEncoder = datasets.TextEncoder
CORPUS = datasets.CORPUS
VanillaRNN = rnn_numpy.VanillaRNN
CharLM = lstm_pytorch.CharLM


def main() -> None:
    sys.path.insert(0, str(_HERE.parents[1]))
    from shared.seeds import set_seed

    set_seed(0)

    enc = TextEncoder(CORPUS)
    ids = enc.encode(CORPUS)
    print(f"corpus: {len(CORPUS)} chars, vocab_size={enc.vocab_size}")

    hidden_size = 64
    seq_len = 32
    n_steps = 400

    print("\n=== NumPy vanilla RNN (hand-derived BPTT) ===")
    rnn = VanillaRNN(vocab_size=enc.vocab_size, hidden_size=hidden_size, learning_rate=0.1, seed=0)
    h = np.zeros(hidden_size, dtype=np.float64)
    for step in range(n_steps):
        start = (step * seq_len) % (len(ids) - seq_len - 1)
        inputs = ids[start : start + seq_len]
        targets = ids[start + 1 : start + 1 + seq_len]
        loss, h = rnn.fit_step(inputs, targets, h)
        if step % 50 == 0 or step == n_steps - 1:
            avg_bits = loss / seq_len / np.log(2)
            print(f"  step {step:4d}: loss/token={loss / seq_len:.3f}  (~{avg_bits:.2f} bits/char)")

    print("\nSample from NumPy RNN (seed='F', 200 chars):")
    seed_id = int(enc.encode("F")[0])
    sample_ids = rnn.sample(seed_id=seed_id, length=200)
    print("  " + repr(enc.decode(sample_ids))[1:-1])

    print("\n=== PyTorch LSTM ===")
    torch.manual_seed(0)
    lstm = CharLM(
        vocab_size=enc.vocab_size,
        hidden_size=hidden_size,
        cell_type="lstm",
        learning_rate=0.003,
        seed=0,
    )
    print(f"parameters: {lstm.n_parameters():,}")
    ids_t = torch.from_numpy(ids)
    for step in range(n_steps):
        start = (step * seq_len) % (len(ids) - seq_len - 1)
        inputs = ids_t[start : start + seq_len].unsqueeze(0)
        targets = ids_t[start + 1 : start + 1 + seq_len].unsqueeze(0)
        loss = lstm.fit_step(inputs, targets)
        if step % 50 == 0 or step == n_steps - 1:
            avg_bits = loss / np.log(2)
            print(f"  step {step:4d}: loss/token={loss:.3f}  (~{avg_bits:.2f} bits/char)")

    print("\nSample from LSTM (seed='F', temperature=0.8, 200 chars):")
    sample_ids = lstm.sample(seed_id=seed_id, length=200, temperature=0.8)
    print("  " + repr(enc.decode(sample_ids))[1:-1])


if __name__ == "__main__":
    main()
