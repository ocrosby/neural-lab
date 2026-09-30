# Introduction to neural networks

## What a neural network is

A neural network is a computational model that learns a mapping from inputs to outputs by adjusting internal numeric parameters (weights and biases) in response to example data, rather than by following rules a programmer writes by hand. The model is organized as layers of simple processing units ("neurons"), loosely inspired by biological neurons, that pass signals forward through weighted connections.

Formally, a neural network approximates a function `f(x) → y`. Instead of specifying `f` directly, you specify:

- an **architecture** (how many layers, how many neurons per layer, how they connect),
- an **objective** (a loss function measuring how wrong the current output is), and
- a **learning procedure** (an algorithm that adjusts weights to reduce that loss).

The network then discovers its own approximation of `f` from data. This is what distinguishes it from classical, explicitly-programmed software.

## Why neural networks matter

Many real-world problems — recognizing objects in images, transcribing speech, translating language, predicting numeric outcomes from many weakly-correlated variables — are difficult to solve with hand-written rules because the mapping from input to output is too complex or too poorly understood to specify directly. Neural networks matter because they can approximate such mappings directly from labeled examples, and because a small set of architectural ideas (layers, weighted sums, nonlinear activation, gradient-based learning) generalizes across a very wide range of problem domains.

## A brief history

| Era | Development |
|---|---|
| 1940s–1950s | McCulloch & Pitts model a neuron as a threshold logic unit; Rosenblatt's Perceptron (1958) is the first trainable single-layer network |
| 1969 | Minsky & Papert show the single-layer perceptron cannot learn functions like XOR, cooling research interest ("AI winter") |
| 1986 | Rumelhart, Hinton, and Williams popularize **backpropagation**, making it practical to train multi-layer networks |
| 1990s–2000s | Convolutional networks (LeCun, LeNet) succeed at digit recognition; broader adoption remains limited by compute and data |
| 2012 | AlexNet wins the ImageNet competition by a large margin, demonstrating deep networks trained on GPUs at scale — the start of the modern "deep learning" era |
| 2017 | The Transformer architecture ("Attention Is All You Need") replaces recurrence with self-attention, becoming the basis for most modern large language models |

## How a network produces an output, at a glance

1. Input data (numbers — pixel intensities, word embeddings, sensor readings) enters the first layer.
2. Each neuron computes a weighted sum of its inputs, adds a bias term, and passes the result through a nonlinear **activation function**.
3. The output of one layer becomes the input to the next, until the final layer produces the network's prediction.
4. During training, a **loss function** compares the prediction to the known correct answer, and **backpropagation** + **gradient descent** adjust every weight in the network to reduce that loss.

See [`02-core-concepts.md`](02-core-concepts.md) for precise definitions of each term used above, and [`03-training-and-learning.md`](03-training-and-learning.md) for how the learning procedure actually works.

## Scope of this reference

This repository is a personal learning reference, not a tutorial with runnable code. It exists to build a solid conceptual foundation — vocabulary, mechanics, and the major architecture families — before writing or studying any implementation.
