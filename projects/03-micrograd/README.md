# 03 — micrograd

A scalar-valued autograd engine in ~150 lines, plus an `nn.py` layer that builds Neurons/Layers/MLPs on top. Following Karpathy's [micrograd](https://github.com/karpathy/micrograd).

**This is the single most important project on the ladder.** Once you understand what's happening inside `Value._backward`, PyTorch stops being magic — every `nn.Linear`, every `.backward()`, every autograd wizard trick reduces to the same DAG walk this engine does, just with tensors instead of scalars and CUDA kernels instead of Python.

## What this teaches

- **Reverse-mode automatic differentiation** — every op attaches a closure that knows how to compute gradients w.r.t. its inputs given the gradient w.r.t. its output. On `.backward()`, we walk the DAG in reverse topological order and run each closure.
- **The chain rule as one line of code** — `self.grad += (local_derivative) * out.grad` is the whole thing. Multiplication by the local derivative propagates the gradient; `+=` accumulates across paths for shared variables.
- **The softmax+CE, tanh, ReLU derivatives** — each activation contributes its own local derivative; they're just calculus.
- **Why PyTorch's API looks the way it does** — `.backward()`, `.grad`, `zero_grad()`, `Module.parameters()` all appear here in tiny form.

## The key insight

Every operation on a `Value` creates a new `Value` that remembers:
1. Its data (the forward-pass result)
2. Its children (which `Value`s produced it)
3. A `_backward` closure that knows how to update children's `.grad` given its own `.grad`

Example, from `Value.__mul__`:

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")

    def _backward():
        self.grad += other.data * out.grad     # d(a*b)/da = b
        other.grad += self.data * out.grad     # d(a*b)/db = a

    out._backward = _backward
    return out
```

That's it. Every op has this shape. Compose them, call `.backward()` on the final `Value`, and gradients flow.

## Run

```bash
uv run python projects/03-micrograd/demo.py
```

Expected output:

```
=== Simple scalar autograd example ===
c = a*b + a^2 - b/2  where a=2.0, b=3.0
c.data = 8.5000
dc/da  = 7.0000  (analytical: b + 2a = 3 + 4 = 7)
dc/db  = 1.5000  (analytical: a - 0.5 = 1.5)

=== Train an MLP on XOR through the engine ===
parameters: 337
step    0: loss = 45.543...
...
step  499: loss = 0.000...

final predictions vs targets: all four correct
```

## Test

```bash
uv run pytest projects/03-micrograd/tests/
```

Two test files:
- `test_engine.py` — the `Value` class, including 6 tests that build the same expression in micrograd AND PyTorch, then compare gradients. **PyTorch is the reference; disagreement is a bug.**
- `test_nn.py` — Module/Neuron/Layer/MLP built on `Value`, plus end-to-end XOR training.

## What surprised me

_(fill this in after you finish the project)_

Ideas to explore before moving on:
- **Visualize the graph**. Add a `draw_dot(root)` function that uses `graphviz` to render the DAG. Look at your loss expression's graph — the shape teaches you a lot.
- **Support `Value ** Value`**. Currently only scalar exponents work. Adding `Value ** Value` needs the product-rule-of-exponents: `d/dx[x^y] = y * x^(y-1)`, `d/dy[x^y] = x^y * log(x)`.
- **Add a broadcasting `Tensor`**. Scale `Value` up to n-dim arrays. Now you're 90% of the way to a mini-PyTorch (the other 10% is CUDA).
- **Read Karpathy's Zero-to-Hero video 1** ("The spelled-out intro to neural networks and backpropagation"). If you built this project first, watching him build it will feel like a review — that's the goal.
- **Compare speed to project 02's numpy MLP**. Same XOR problem, both trained to convergence. Micrograd is ~100× slower — Python object overhead per scalar. This is exactly why PyTorch is tensor-based, not scalar-based.

## Layout

```
03-micrograd/
├── engine.py           # Value class (~150 lines) — the autograd engine
├── nn.py               # Module / Neuron / Layer / MLP built on Value
├── demo.py             # Runnable demo: scalar autograd + XOR training
└── tests/
    ├── test_engine.py  # Value tests, including PyTorch-comparison ground truth
    └── test_nn.py      # NN primitive tests, including end-to-end XOR
```
