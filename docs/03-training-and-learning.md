# How neural networks learn

Training is the process of finding weight and bias values that make the network's predictions match reality as closely as possible, measured over a training dataset. This document defines that process end to end.

## The training loop, at a glance

1. **Forward pass** — feed a batch of examples through the network to produce predictions.
2. **Compute loss** — compare predictions to the true labels/targets using a loss function.
3. **Backward pass (backpropagation)** — compute how much each weight in the network contributed to the loss.
4. **Update weights** — nudge every weight slightly in the direction that reduces the loss (gradient descent).
5. Repeat for many batches and many passes over the dataset (**epochs**) until the loss stops improving meaningfully.

## Loss functions

The loss function defines what "wrong" means for a given task. Choosing it correctly matters as much as choosing the architecture.

| Task | Common loss | Notes |
|---|---|---|
| Regression (predict a number) | Mean squared error (MSE) | Penalizes large errors quadratically |
| Binary classification | Binary cross-entropy | Pairs with a sigmoid output |
| Multi-class classification | Categorical cross-entropy | Pairs with a softmax output |

## Gradient descent

Gradient descent is the optimization algorithm that updates parameters to reduce loss. For a parameter `w`, the update rule is:

```
w_new = w_old - learning_rate * (∂loss / ∂w)
```

- `∂loss / ∂w` (the **gradient**) says which direction increases the loss; moving the opposite direction decreases it.
- **Learning rate** is a hyperparameter controlling step size. Too large causes the loss to diverge or oscillate; too small makes training impractically slow.

### Variants

- **Batch gradient descent**: compute the gradient over the entire training set before each update. Accurate but slow and memory-heavy for large datasets.
- **Stochastic gradient descent (SGD)**: compute the gradient from a single example at a time. Fast per step but noisy.
- **Mini-batch gradient descent**: compute the gradient over a small batch (e.g., 32–256 examples). The standard default — balances the accuracy of batch gradient descent with the speed of SGD.
- **Adaptive optimizers** (Adam, RMSProp, etc.): adjust the effective learning rate per parameter based on the history of past gradients, typically converging faster and more reliably than plain SGD.

## Backpropagation

Backpropagation is the algorithm that computes `∂loss / ∂w` for every weight in the network efficiently, using the chain rule of calculus, working backward from the output layer to the input layer:

1. Compute the loss at the output layer.
2. Compute how much the loss changes with respect to the output layer's pre-activations.
3. Propagate that quantity backward one layer at a time, at each step combining it with the local derivative of that layer's activation function and weights, to obtain the gradient for that layer's parameters.

The key insight is *reuse*: the gradient computed for one layer is reused to compute the gradient for the layer before it, rather than recomputing everything from scratch — this is what makes training deep networks computationally tractable.

## Epochs, batches, and iterations

- **Epoch**: one full pass over the entire training dataset.
- **Batch**: a subset of the training data processed together in one forward/backward pass.
- **Iteration (step)**: one weight update, corresponding to one batch. `iterations per epoch = dataset size / batch size`.

## Overfitting, underfitting, and generalization

- **Underfitting**: the model is too simple (or undertrained) to capture the pattern in the training data — high error on both training and unseen data.
- **Overfitting**: the model has effectively memorized the training data, including its noise, and performs much worse on unseen data than on training data.
- **Generalization**: how well the model performs on data it did not train on — the actual goal of training, since a model that only performs well on training data is not useful.

### Common mitigations for overfitting

| Technique | Mechanism |
|---|---|
| More training data | Directly reduces the chance of memorizing noise |
| Regularization (L1/L2) | Penalizes large weights in the loss function, favoring simpler solutions |
| Dropout | Randomly disables a fraction of neurons during each training step, preventing over-reliance on any single unit |
| Early stopping | Stops training once performance on a held-out validation set stops improving, even if training loss keeps falling |
| Data augmentation | Synthesizes variations of training examples (e.g., rotated/cropped images) to expose the model to more variety |

## Train / validation / test split

- **Training set**: examples the model directly learns from (used in the forward/backward pass).
- **Validation set**: held-out examples used to tune hyperparameters and detect overfitting during development — the model never learns from these directly.
- **Test set**: a final held-out set used exactly once, after all tuning is complete, to report an unbiased estimate of real-world performance.

See [`02-core-concepts.md`](02-core-concepts.md) for definitions of the terms this document builds on, and [`04-architectures-overview.md`](04-architectures-overview.md) for how these mechanics apply to specific architecture families.
