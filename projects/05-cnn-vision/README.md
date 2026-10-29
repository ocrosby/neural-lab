# 05 — CNN vision

Convolutional neural network for image classification. LeNet-style architecture: alternating Conv2d + ReLU + MaxPool2d, then a couple of fully connected layers, trained end-to-end on real MNIST.

## What this teaches

- **Convolutions as shared-weight linear ops.** A Conv2d layer applies the same small filter at every spatial position. That is the whole thing — no magic, just weight sharing. It means a filter that detects "vertical edge" works whether the edge is in the top-left, middle, or bottom-right of the image.
- **Why CNNs beat MLPs on pixel data.** MLPs learn a separate weight per input pixel; if the object moves one pixel over, the MLP has to relearn. CNNs are *translation equivariant* — move the input, the feature map moves the same way. This is why a small CNN hits 98.9% on MNIST in 3 epochs while an MLP takes longer for the same accuracy.
- **Pooling as downsampling + invariance.** `MaxPool2d(2)` halves the spatial resolution and gives the network small-shift invariance ("the digit is still a 7 whether the pixel is at row 12 or row 13"). Modern architectures replace pooling with strided convolutions but the principle is the same.
- **Feature maps and hierarchical composition.** The first conv layer's filters usually learn edges and simple textures. Deeper layers learn compositions of those. The demo saves the first-layer filters so you can see this happen.
- **`DataLoader` and batching.** MNIST is 60k train samples, too many for full-batch. Mini-batch SGD via `DataLoader` is the pattern for every larger dataset from here on.

## Architecture

```
(N, 1, 28, 28)                                           # MNIST input
      │
      ▼
Conv2d(1 → 16, kernel=3, padding=1)   → ReLU → MaxPool2d(2)   # (N, 16, 14, 14)
      │
      ▼
Conv2d(16 → 32, kernel=3, padding=1)  → ReLU → MaxPool2d(2)   # (N, 32,  7,  7)
      │
      ▼
Flatten                                                        # (N, 32 * 7 * 7 = 1568)
      │
      ▼
Linear(1568 → 128) → ReLU                                     # (N, 128)
      │
      ▼
Linear(128 → 10)                                              # (N, 10) logits
```

Total: 206,922 parameters. About 1000× smaller than a ResNet-18, ~10× smaller than a fully-connected MLP that would reach the same accuracy.

## Same-shape API as project 04, different input contract

| API surface | project 04 (MLP) | project 05 (CNN) |
|---|---|---|
| Constructor | `MLP(layer_sizes, ...)` | `CNN(in_channels, image_size, ...)` |
| `.fit(...)` | `.fit(X, y)` (NumPy) | `.fit(train_loader, n_epochs)` (DataLoader) |
| `.predict(X)` | X is `(N, features)` NumPy | X is `(N, C, H, W)` Tensor |
| `.evaluate(...)` | *(via score())* | `.evaluate(loader)` |
| `.predict_proba(X)` | valid distribution | valid distribution |

The DataLoader shift is deliberate — every real dataset from here on (Fashion-MNIST, CIFAR-10, text corpora, image folders) needs batching. Learn the pattern here on 60k MNIST samples where you can eyeball the results.

## Run

```bash
# Fast tests, no MNIST download
uv run pytest projects/05-cnn-vision/tests/
```

```bash
# Real demo — downloads MNIST (~11 MB) on first run, trains for 3 epochs (~1 min on CPU)
uv run python projects/05-cnn-vision/demo.py
```

Expected output:

```
Loading MNIST from .../data...
  train: 60000, test: 10000

=== CNN on MNIST ===
parameters: 206,922
epoch 1: train loss = 0.29, test accuracy = 97.60%
epoch 2: train loss = 0.07, test accuracy = 98.41%
epoch 3: train loss = 0.05, test accuracy = 98.89%

first-conv filter shape: (16, 1, 3, 3)
filters saved to .../first_conv_filters.png
```

Open `first_conv_filters.png` and look at the 16 filters. Depending on the seed you'll see some clear directional edge detectors — vertical, horizontal, diagonal — mixed with noisier filters that haven't specialized yet. Run for more epochs and the noise cleans up.

## Test

```bash
uv run pytest projects/05-cnn-vision/tests/
```

15 tests, all fast (no MNIST download). Uses a synthetic "bright square in one of 4 quadrants" dataset that a CNN can learn in 5 epochs. Two dataset-quality tests explicitly verify the synthetic data has the intended signal — belt-and-suspenders so a green training test can't be caused by an unlearnable dataset.

## What surprised me

_(fill this in after you finish the project)_

Ideas to explore before moving on:
- **Compare to an MLP.** Train `MLP([784, 128, 10])` from project 04 on the same MNIST for 3 epochs. Which wins? By how much? At what parameter count?
- **Fashion-MNIST.** Change `datasets.MNIST` to `datasets.FashionMNIST` in `demo.py`. Same shape (28×28 grayscale, 10 classes), harder problem (clothing items instead of digits). Accuracy will drop 5-10%.
- **CIFAR-10.** `datasets.CIFAR10` — 32×32 RGB, 10 classes. Build the model with `in_channels=3, image_size=32`. A 2-conv CNN will get to ~65-70% — far short of the ~99% MNIST result. This gap is why architectures got deeper (VGG, ResNet).
- **Visualize deeper filters.** Second-layer filters aren't as pretty because they operate on the first layer's output, not on pixels. Use activation visualization instead: pass a real image through and plot `self.features[3](self.features[:3](x))` — the second layer's output feature maps.
- **Read the Conv2d source.** `nn.Conv2d.forward` is one line: `F.conv2d(input, self.weight, self.bias, ...)`. All the work happens in the underlying `torch.conv2d` kernel (which on CPU dispatches to MKL, on CUDA to cuDNN). Same "Python orchestrates, native code computes" pattern that runs every LLM.

## Layout

```
05-cnn-vision/
├── cnn.py         # LeNet-style CNN
├── datasets.py    # Synthetic quadrant task (for fast tests)
├── demo.py        # Real MNIST training + filter visualization
├── first_conv_filters.png  # Generated by demo (gitignored)
└── tests/
    └── test_cnn.py
```
