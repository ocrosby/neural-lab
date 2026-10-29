# 02 — MLP from scratch

Multi-layer perceptron with hand-derived backpropagation. Pure NumPy, no autograd.

## What this teaches

- **Why depth needs non-linearity** — without an activation between layers, stacked linear transforms collapse into a single linear transform (matrix associativity). ReLU breaks the collapse.
- **The chain rule as backprop** — the gradient of the loss w.r.t. every weight is a product of local gradients, propagated backwards from the output. Derive it once by hand and PyTorch's `.backward()` stops being magic.
- **The softmax + cross-entropy shortcut** — the gradient of cross-entropy w.r.t. the output pre-activation simplifies to `p − y_onehot`. This is why every framework fuses the two into one op; the intermediate softmax values don't need to be differentiated through.
- **Gradient checking** — the load-bearing sanity check for any hand-derived backprop. Finite-difference numerical gradients must match analytical gradients to within `~1e-5`. If they don't, the math is wrong and any training success is coincidence.
- **He initialization** — `W ~ N(0, sqrt(2/n_in))` for ReLU. Wrong init and deep networks either explode or collapse to zero.

## Architecture

```
input  ──►  Linear ──► ReLU ──► Linear ──► ReLU ──►  ...  ──► Linear ──► Softmax  ──►  probabilities
                                                                                       │
                                                                                       ▼
                                                                              cross-entropy loss
```

Hidden layers: linear + ReLU. Output layer: linear + softmax. Loss: cross-entropy. Optimizer: full-batch gradient descent (the datasets here are tiny; mini-batch waits for MNIST / project 04).

## Run

```bash
uv run python projects/02-mlp-from-scratch/mlp.py
```

Expected output:

```
=== XOR (perceptron cannot; MLP can) ===
accuracy = 100.00%
predictions = [0 1 1 0], true = [0 1 1 0]

=== 3-class Gaussian blobs ===
train accuracy = 100.00%
test accuracy  = 100.00%

=== 5-class Gaussian blobs ===
train accuracy = 100.00%
test accuracy  = 100.00%
```

## Test

```bash
uv run pytest projects/02-mlp-from-scratch/tests/
```

The `TestGradientCorrectness` test is the critical one — it checks analytical backprop against numerical gradients. Keep it green.

## What surprised me

_(fill this in after you finish the project)_

Ideas to explore before moving on:
- **Depth vs width**: does `[2, 8, 8, 2]` beat `[2, 16, 2]` on XOR? On something harder?
- **Learning rate**: what happens with `lr=1.0`? `lr=0.001`? Plot the loss curve.
- **Activation choice**: swap ReLU for `tanh` or `sigmoid`. Retrain XOR. Any difference in convergence speed? Why does ReLU tend to win?
- **Weight init**: try `rng.normal(0, 0.01)` instead of He init. Does deep training still work?
- **Real MNIST**: `torchvision.datasets.MNIST` is one import away. Reshape 28×28 to 784, feed to `MLP([784, 128, 10])`. See how far you can push accuracy with pure NumPy (should hit ~97% comfortably).
- **The failure mode**: what if you *remove* the ReLU activation and just stack `Linear → Linear → Linear`? Retrain XOR and watch it fail — this proves the collapse-to-linear argument.
