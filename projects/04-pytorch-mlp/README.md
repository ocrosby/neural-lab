# 04 — PyTorch MLP

The same MLP as project 02, rebuilt in PyTorch. Same API, same behavior, same tests — but with the manual forward pass, backprop, initialization, optimizer, and loss function replaced by framework primitives.

## What this teaches

- **`nn.Module` and `nn.Sequential`** — the composition pattern PyTorch uses for every model, from a 5-line MLP to a 400B-parameter Llama
- **`torch.optim`** — Adam replaces the hand-written `p -= lr * dW` loop. Adaptive learning rates, momentum, and bias correction come for free
- **`nn.CrossEntropyLoss`** — fuses log-softmax + NLL for numerical stability, and gives you the `p − y_onehot` gradient shortcut (which project 02 hand-derived) internally
- **`.backward()`** — the DAG walk from project 03, but through C++ tensors on CPU/MPS/CUDA rather than Python scalars
- **How much boilerplate the framework absorbs** — see line count below

## Same problem, less code

| File | Lines |
|---|---|
| `projects/02-mlp-from-scratch/mlp.py` | 248 |
| `projects/04-pytorch-mlp/mlp.py`      | 145 |
| **Difference**                        | **−103** |

That difference is: manual forward pass, manual backward pass, He init, one-hot, softmax, cross-entropy, full-batch SGD, gradient-check API. All replaced by `nn.Linear`, `nn.ReLU`, `nn.CrossEntropyLoss`, and `torch.optim.Adam`.

The behavior is identical. The tests are near-identical. That's the value of the framework.

## Run

```bash
uv run python projects/04-pytorch-mlp/demo.py
```

Expected output:

```
=== XOR ===
parameters: 42
accuracy = 100.00%
predictions = [0 1 1 0], true = [0 1 1 0]

=== 3-class Gaussian blobs ===
parameters: 99
train accuracy = 100.00%
test accuracy  = 100.00%

=== 5-class Gaussian blobs ===
parameters: 261
train accuracy = 100.00%
test accuracy  = 100.00%
```

## Test

```bash
uv run pytest projects/04-pytorch-mlp/tests/
```

17 tests covering the same behavioral contract as project 02, plus PyTorch-specific tests: `n_parameters()` matches manual math, `state_dict` save/load preserves predictions.

## What surprised me

_(fill this in after you finish the project)_

Ideas to explore before moving on:
- **Compare training curves.** Instrument project 02 and this one to print loss every 50 epochs. Full-batch SGD (project 02) vs. Adam (this project): plot both. Adam's adaptive LR is why it converges faster on the same problem.
- **Run on GPU.** On Apple Silicon: `model.net.to("mps")` and `X_t = X_t.to("mps")`. On CUDA: replace `"mps"` with `"cuda"`. For toy data, GPU is *slower* — dispatch overhead dominates. Try it and observe.
- **Swap the optimizer.** Try `torch.optim.SGD`, `torch.optim.RMSprop`, `torch.optim.AdamW`. Which converges fastest on 5-class Gaussian blobs?
- **Add `DataLoader`.** Full-batch is fine for toy data. Replace with `torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)`. Now `.fit()` iterates minibatches. This is the pattern for project 05 (MNIST doesn't fit in memory in full-batch).
- **Real MNIST.** `torchvision.datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())`. Flatten 28×28 to 784, use `MLP([784, 128, 10])`. Should hit ~98% test accuracy in a few minutes on CPU.
- **Read the source.** `torch.optim.Adam` is ~150 lines. `nn.Linear.forward` is one line. `nn.CrossEntropyLoss` is a thin wrapper over `F.cross_entropy`. All of it is Python — go look.

## Layout

```
04-pytorch-mlp/
├── mlp.py          # 145 lines — nn.Sequential + Adam + CE loss
├── demo.py         # XOR + multiclass
└── tests/
    └── test_mlp.py # same contract as project 02
```
