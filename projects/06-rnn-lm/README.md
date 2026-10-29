# 06 — RNN language model

Character-level language model on an embedded Shakespeare snippet. Two implementations:

- `rnn_numpy.py` — vanilla RNN with **hand-derived Backprop Through Time** (BPTT) in pure NumPy. Trained with Adagrad + gradient clipping.
- `lstm_pytorch.py` — LSTM (or GRU) with `nn.Embedding + nn.LSTM + nn.Linear` in PyTorch. Same task, dramatically better perplexity.

Per the plan for project 06: RNN in NumPy for depth, LSTM in PyTorch to see the payoff.

## What this teaches

- **Hidden state as memory.** The RNN carries a fixed-size vector `h_t` forward across time. All information about the past that the network gets to remember has to fit into that vector.
- **Backpropagation Through Time (BPTT).** Unroll the RNN for `T` steps, compute per-step cross-entropy, sum the losses, and backprop through the unrolled graph. The gradient w.r.t. `W_hh` involves a chain of `T` Jacobians — that's where the vanishing/exploding gradient problems live.
- **Why LSTMs beat RNNs.** LSTMs introduce a *cell state* with a straight-line gradient path, gated by learned sigmoids. The network can learn to hold information for hundreds of time steps by keeping the forget gate near 1. A vanilla RNN has no such mechanism.
- **Gradient clipping.** Norm-based clipping (threshold 5.0) is the standard exploding-gradient safeguard. No cheap safeguard exists against vanishing — that's what LSTM/GRU are for.
- **Temperature sampling.** Divide logits by temperature before softmax. `T → 0` is argmax (repetitive), `T → ∞` is uniform (garbage), `T ≈ 0.7-1.0` is the interesting range.

## The reality check

Trained on the same 1.6KB Shakespeare snippet for 400 steps at hidden_size=64:

| Model | Final loss (bits/char) | Sample |
|---|---|---|
| **NumPy vanilla RNN** | ~4.0 | `ic phthtnre ticf Cinpeztote atC out pce hs p tn t d ls` |
| **PyTorch LSTM** | ~1.6 | `inst his to the rasond Ciet Citizen:\ngec-ess hilst` |

The LSTM sample has recognizable Shakespeare structure — `Citizen:` tags, tag-newline-line patterns, word-length rhythms. The RNN sample doesn't. **This gap is not a bug in the RNN implementation** — the load-bearing gradient-check test verifies BPTT is mathematically correct. The gap is the vanishing-gradient problem made visible.

## Architecture

**Vanilla RNN** (`rnn_numpy.py`):

```
x_t (one-hot, V)
   │
   ▼
h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)     # (H,)
   │
   ▼
y_t = W_hy @ h_t + b_y                             # (V,) logits
   │
   ▼
softmax(y_t)
```

Parameters (for V=48, H=64): `W_xh` (H×V) + `W_hh` (H×H) + `W_hy` (V×H) + biases = ~10K.

**LSTM** (`lstm_pytorch.py`):

```
x_t (int id) --[Embedding(V, H)]-->  emb_t (H,)
   │
   ▼
[nn.LSTM(H, H)]  --> h_t (H,)   with 4 gates: input, forget, output, candidate
   │
   ▼
[nn.Linear(H, V)] --> logits (V,)
```

Parameters (for V=48, H=64): `Embedding` (V×H = 3072) + `LSTM` (~33K, from 4 gates × (H×H + H×H + H + H)) + head (H×V + V = 3120) = ~39K. About 4× the RNN parameter count, and ~3-5× lower loss.

## Run

```bash
uv run python projects/06-rnn-lm/demo.py
```

Trains both models on the embedded Shakespeare corpus and generates a 200-character sample from each. Runs in ~30s on CPU.

## Test

```bash
uv run pytest projects/06-rnn-lm/tests/
```

Fast (< 2s total). Three test files:
- `test_datasets.py` — encoder round-trip, sequence sampling shapes
- `test_rnn_numpy.py` — **the load-bearing test is `test_analytical_gradient_matches_numerical`** — BPTT gradients checked against finite differences to `rtol=1e-3`. If this passes, BPTT is right.
- `test_lstm_pytorch.py` — architecture, training, temperature-controlled sampling, LSTM vs. GRU parameter counts (LSTM has 4/3 the params of GRU)

## What surprised me

_(fill this in after you finish the project)_

Ideas to explore before moving on:
- **Full tiny-shakespeare.** Karpathy's classic corpus is 1MB (vs the 1.6KB embedded here). Point the demo at `~/.cache/tiny-shakespeare.txt` and train the LSTM for 20k steps. You'll get samples that read like Shakespeare — same architecture, more data.
- **Gradient norm across time.** Instrument the NumPy RNN to log `||dh_t/dh_0||` for each backward step. Plot it. You'll see the norms decay exponentially — that's the vanishing gradient problem in one plot.
- **GRU vs LSTM.** Rerun the demo with `cell_type="gru"`. GRU has ~75% the parameters of LSTM and typically gets slightly worse loss on hard tasks — but converges faster in wall-clock time on small models.
- **Beam search decode.** Replace `torch.multinomial` sampling with beam search (`k=5`). Compare sample quality at the same temperature.
- **Read Karpathy's [char-rnn blog post](https://karpathy.github.io/2015/05/21/rnn-effectiveness/).** He built exactly this system in Lua/Torch in 2015 — the code is different, the intuition is identical.
- **The transformer looms.** Project 07 replaces the recurrent scan (`for t in range(T): h_t = f(h_{t-1}, x_t)`) with **parallel attention over all previous tokens**. That change alone — no more sequential recurrence — is what unlocked training LMs at trillion-token scale. This project is the last time you'll see a for-loop over time steps in the ladder.

## Layout

```
06-rnn-lm/
├── rnn_numpy.py       # vanilla RNN + hand-derived BPTT (~200 lines)
├── lstm_pytorch.py    # nn.LSTM / nn.GRU + embedding + linear head (~120 lines)
├── datasets.py        # embedded Shakespeare + TextEncoder
├── demo.py            # trains both, prints samples
└── tests/
    ├── test_datasets.py
    ├── test_rnn_numpy.py       # includes BPTT gradient check
    └── test_lstm_pytorch.py    # includes LSTM vs GRU comparison
```
