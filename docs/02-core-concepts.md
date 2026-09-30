# Core concepts and definitions

This document defines the building blocks referenced throughout the rest of this reference. Each definition is intentionally precise rather than metaphorical.

## Neuron (unit)

The basic computational element. A single neuron takes a vector of inputs `x = [x1, x2, ..., xn]`, a matching vector of learned weights `w = [w1, w2, ..., wn]`, and a learned scalar bias `b`. It computes:

```
z = (w1*x1 + w2*x2 + ... + wn*xn) + b
a = activation(z)
```

`z` is called the **pre-activation** (or logit); `a` is the neuron's **output** (or activation).

## Weights and biases

- **Weight**: a learned scalar that scales the contribution of one input to a neuron. Weights are the network's primary learned parameters — they encode what the network has "learned."
- **Bias**: a learned scalar added after the weighted sum, independent of the input. It shifts the activation function, letting a neuron fire (activate strongly) even when all inputs are zero.

Together, weights and biases are usually just called the network's **parameters**.

## Layer

A group of neurons that all receive the same input and whose outputs are computed in parallel.

- **Input layer**: not really neurons — just the raw feature vector fed into the network.
- **Hidden layer**: any layer between the input and the output. "Deep learning" refers to networks with multiple hidden layers.
- **Output layer**: produces the network's final prediction; its size and activation function match the task (e.g., one unit with a sigmoid for binary classification, `k` units with softmax for `k`-class classification).

## Activation function

A nonlinear function applied to a neuron's pre-activation `z`. Without nonlinearity, stacking layers would collapse into a single linear transformation, no matter how many layers you add — nonlinearity is what lets networks approximate complex functions. Common choices:

| Function | Formula | Typical use |
|---|---|---|
| Sigmoid | `1 / (1 + e^-z)` | Legacy choice; output layer of binary classifiers |
| Tanh | `(e^z - e^-z) / (e^z + e^-z)` | Legacy hidden-layer choice; zero-centered output |
| ReLU | `max(0, z)` | Default hidden-layer choice in most modern networks — cheap and avoids saturating for `z > 0` |
| Softmax | `e^zi / Σ e^zj` (vector-valued) | Output layer for multi-class classification; converts logits into a probability distribution |

## Forward pass

The process of computing a prediction: input flows through each layer in sequence, each layer's output becoming the next layer's input, until the output layer produces a result. "Feedforward" describes any network whose connections only go forward, with no cycles.

## Loss function (cost function)

A function that measures the discrepancy between the network's prediction and the true target for a given example. Training is defined as **minimizing the average loss over the training data**. Common choices are covered in [`03-training-and-learning.md`](03-training-and-learning.md).

## Parameters vs. hyperparameters

- **Parameters**: values the training algorithm learns automatically — weights and biases.
- **Hyperparameters**: values a human sets before training and that the algorithm does not learn — number of layers, neurons per layer, learning rate, batch size, choice of activation function, etc.

## Model capacity

Informally, how complex a function a network can represent. Capacity generally grows with the number of parameters (more layers, more neurons per layer). Too little capacity causes **underfitting** (the network cannot represent the true pattern); too much capacity relative to the amount of training data risks **overfitting** (the network memorizes training examples instead of generalizing) — see [`03-training-and-learning.md`](03-training-and-learning.md) for how this is diagnosed and mitigated.

## Tensor

A generalization of scalars, vectors, and matrices to any number of dimensions — a tensor is simply an n-dimensional array of numbers. Neural network frameworks represent every input, weight, and intermediate value as a tensor, because it is the one data structure that supports both the linear-algebra operations (matrix multiplication) and the batched, GPU-parallel computation neural networks depend on.
