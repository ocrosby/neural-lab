# 01 — Perceptron

Rosenblatt's 1958 single-layer perceptron. Pure NumPy, no framework.

## What this teaches

- **Weights, bias, activation** — the smallest possible parametric model: `y_hat = sign(w · x + b)`
- **The update rule** — Rosenblatt's convergence rule: on misclassification, move the weights toward the correct answer by `η · y · x`
- **Linear separability** — a hyperplane cuts the input space into two half-spaces; if the data lives cleanly on either side, the perceptron finds a separator
- **Why XOR fails** — no single hyperplane separates `{(0,0), (1,1)}` from `{(0,1), (1,0)}`. This motivates the multi-layer perceptron in project 02.

## Run

```bash
uv run python projects/01-perceptron/perceptron.py
```

Expected output:

```
=== Linearly-separable blobs ===
converged in 2 epoch(s), accuracy = 100.00%
weights = [2.47... 2.17...], bias = -1.0000

=== XOR (should fail) ===
stopped after 1000 epoch(s), accuracy = 50.00%
```

## Test

```bash
uv run pytest projects/01-perceptron/tests/
```

## What surprised me

_(fill this in after you finish the project)_

Ideas to explore before moving on:
- Change `learning_rate`. Does it affect the *final* accuracy, or just the number of epochs?
- What happens if the blobs overlap (increase `scale=0.5` to `scale=2.0`)?
- Can you visualize the decision boundary? Add a matplotlib scatter plot with the line `w · x + b = 0`.
- Rosenblatt's proof: the number of updates is bounded by `(R/γ)²` where `R` is the max input norm and `γ` is the margin. Can you observe this?
