# neural-lab

Progressive neural network projects, from a single perceptron to your own LLM.

**Long-term goal**: train your own LLM. This repo is the ladder that gets you there — starting with pencil-and-paper math and ending at a small trained transformer, alignment, and inference optimization.

## Project ladder

Fifteen projects. Each lives in `projects/NN-slug/` with code, tests, and a per-project README. Do them in order — each project assumes the previous.

### Foundations (build the intuition)

- [x] **01** — [`perceptron`](projects/01-perceptron/) — Rosenblatt's 1958 perceptron in NumPy. Solves linearly-separable data, fails on XOR.
- [x] **02** — [`mlp-from-scratch`](projects/02-mlp-from-scratch/) — Multi-layer perceptron with hand-derived backprop. Pure NumPy. Solves XOR + multiclass Gaussian blobs. Includes analytical-vs-numerical gradient check.
- [x] **03** — [`micrograd`](projects/03-micrograd/) — Scalar autograd engine (~150 lines) + Neuron/Layer/MLP on top. PyTorch-comparison tests as ground truth; trains XOR through the engine.

### Frameworks (stop reinventing)

- [x] **04** — [`pytorch-mlp`](projects/04-pytorch-mlp/) — Same MLP as project 02, rebuilt in PyTorch. 145 lines vs 248. Same behavioral contract, near-identical tests.
- [x] **05** — [`cnn-vision`](projects/05-cnn-vision/) — LeNet-style CNN. 98.9% on real MNIST in 3 epochs. Fast tests on synthetic data + `DataLoader` batching pattern.

### Sequences (time is a dimension)

- [x] **06** — [`rnn-lm`](projects/06-rnn-lm/) — Character-level LM on Shakespeare. NumPy RNN with hand-derived BPTT (gradient check) + PyTorch LSTM/GRU. Vanishing gradients made visible.
- [ ] **07a** — `transformer-numpy` — Forward-only transformer in pure NumPy. Read the paper from the inside out.
- [ ] **07b** — `transformer-pytorch` — Same architecture, trainable, in PyTorch.

### LLM path

- [ ] **08** — `tokenization` — Build BPE from scratch. Compare against `tiktoken`, `sentencepiece`.
- [ ] **09** — `nanoGPT` — Train a 10M–125M param GPT on TinyStories. **First "own LLM."**
- [ ] **10** — `data-pipeline` — Dedup, quality filter, streaming, sequence packing.
- [ ] **11** — `scaling-laws` — Mini Chinchilla experiment. Fit the scaling law yourself.
- [ ] **12** — `distributed-training` — DDP → FSDP → tensor parallel.
- [ ] **13** — `finetune-and-align` — SFT → DPO/RLHF. Turn a base model into an assistant.
- [ ] **14** — `inference-optimization` — KV cache, speculative decoding, quantization.
- [ ] **15** — `evaluation` — MMLU, HellaSwag, HumanEval, GSM8K.

## Setup

```bash
uv sync
uv run pytest              # every project's tests
uv run ruff check .        # lint
```

Python 3.12+, `uv` for env management. PyTorch is the default framework from project 04 onward, with deliberate NumPy detours at 06 (RNN half) and 07a (transformer).

## Layout

```
neural-lab/
├── pyproject.toml            # workspace root, shared deps
├── shared/                   # utilities every project uses (seeds, plotting, datasets)
├── projects/
│   ├── 01-perceptron/
│   ├── 02-mlp-from-scratch/
│   └── ...
└── docs/notes/               # per-project written-up learnings
```

## Conventions

- Every project has `README.md`, code, and `tests/`. Tests are black-box (see `~/.claude/rules/black-box-testing.md`) — assert on outcomes, not internal weights.
- Reproducibility: every entry point calls `shared.seeds.set_seed(42)` first.
- Notebooks are for exploration; `.py` files are the artifact.
- No data in git — `data/`, `checkpoints/`, `runs/` are all gitignored.

## Learning resources

- **Karpathy's Zero-to-Hero** — youtube.com/@AndrejKarpathy — projects 1–9 map onto this course
- **`karpathy/nanoGPT`** — reference for project 09
- **`karpathy/llm.c`** — GPT-2 in raw C/CUDA; read after project 12
- **"Attention Is All You Need"** (Vaswani et al. 2017) — before project 07
- **Chinchilla paper** (Hoffmann et al. 2022) — before project 11
- **HuggingFace `trl`** — reference for project 13
- **`vllm-project/vllm`** — reference for project 14
- **HuggingFace `lm-evaluation-harness`** — reference for project 15

## Compute reality

| Stage | Hardware | Cost |
|---|---|---|
| Projects 1–8 | Laptop or free Colab | $0 |
| Project 9 (nanoGPT) | RTX 3090/4090 or Colab Pro+ | $0–$50 |
| Projects 10–11 | Rented A100 (RunPod/Lambda) | ~$1–$3/hr |
| Project 12+ | Rented 8×A100 or 8×H100 node | ~$10–$30/hr |
| Level-3 pretraining (1B–3B) | 8×H100 for days/weeks | $10k–$100k |
