# Neural network architectures: an overview

Every architecture below is built from the primitives defined in [`02-core-concepts.md`](02-core-concepts.md); each family simply constrains how neurons connect to solve a particular kind of problem well.

## Feedforward neural network (FNN / MLP)

The baseline architecture: input flows strictly forward through one or more fully-connected ("dense") hidden layers to an output layer. Also called a **multilayer perceptron (MLP)**.

- Every neuron in a layer connects to every neuron in the next layer.
- No memory of previous inputs, no assumption about spatial or sequential structure.
- Well suited to tabular data where features have no inherent spatial or temporal ordering.
- The universal building block — most other architectures still end with one or more dense layers.

## Convolutional neural network (CNN)

Designed for data with spatial structure, most commonly images.

- A **convolutional layer** slides a small learned filter (kernel) across the input, computing a weighted sum at each position. The same filter (the same weights) is reused across the whole input — this **parameter sharing** is what makes CNNs far more efficient than a dense layer for image-sized inputs, and what lets a filter that detects, say, an edge, detect that edge anywhere in the image.
- **Pooling layers** (e.g., max pooling) downsample the feature maps produced by convolution, reducing dimensionality and making the representation somewhat robust to small translations.
- Early layers tend to learn low-level features (edges, textures); deeper layers combine them into higher-level features (shapes, object parts).
- Dominant architecture for image classification, object detection, and segmentation.

## Recurrent neural network (RNN) and LSTM

Designed for sequential data (time series, text, audio), where order matters and inputs have variable length.

- An RNN processes a sequence one element at a time, maintaining a **hidden state** that is updated at each step and carries information from earlier elements forward.
- Plain RNNs suffer from the **vanishing/exploding gradient problem**: gradients propagated backward through many time steps shrink toward zero or grow without bound, making it hard to learn dependencies across long sequences.
- **LSTM** (Long Short-Term Memory) and **GRU** (Gated Recurrent Unit) cells address this with learned "gates" that control what information is kept, forgotten, or written to the hidden state at each step, allowing much longer-range dependencies to be learned.
- Largely superseded by Transformers for large-scale language tasks, but still used for smaller sequence problems and streaming/online settings where processing one element at a time is a natural fit.

## Transformer

The dominant architecture for language and increasingly for other sequence and even image tasks (introduced in "Attention Is All You Need," 2017).

- Replaces recurrence entirely with **self-attention**: for each element in a sequence, the model computes a weighted combination of *all* other elements, where the weights ("attention scores") are learned based on how relevant each pair of elements is to each other.
- Because attention looks at the whole sequence at once rather than one step at a time, Transformers parallelize far better than RNNs during training and capture long-range dependencies more directly.
- **Positional encoding** is added to the input to give the model information about element order, since self-attention itself has no inherent notion of sequence position.
- Composed of stacked blocks, each combining a self-attention layer with a feedforward (MLP) layer, plus normalization and residual ("skip") connections that help gradients flow through many stacked blocks.
- The basis for modern large language models (GPT, Claude, etc.) and vision transformers (ViT).

## Autoencoder

A network trained to reconstruct its own input, structured as an **encoder** that compresses the input into a smaller latent representation, and a **decoder** that reconstructs the original input from that representation.

- Forces the network to learn a compact, informative representation of the data, since the bottleneck prevents it from simply copying the input through.
- Used for dimensionality reduction, denoising, and anomaly detection (inputs that reconstruct poorly are flagged as unusual).
- **Variational autoencoders (VAEs)** extend this into a generative model by learning a probability distribution over the latent space rather than a single fixed encoding.

## Generative adversarial network (GAN)

Two networks trained in opposition:

- A **generator** tries to produce fake data realistic enough to fool a second network.
- A **discriminator** tries to distinguish real training data from the generator's fake data.
- The two are trained together; as the discriminator improves at catching fakes, the generator improves at producing more convincing ones, and vice versa.
- Historically important for image generation, though diffusion models have largely displaced GANs for that task in recent years.

## Choosing among these

| Data shape | Typical first choice |
|---|---|
| Tabular / structured features | Feedforward (MLP) |
| Images | Convolutional (CNN) |
| Short sequences, streaming/online | RNN / LSTM |
| Long sequences, language, large-scale pretraining | Transformer |
| Unsupervised representation learning, anomaly detection | Autoencoder |
| Realistic sample generation | GAN or diffusion model |

See [`05-glossary.md`](05-glossary.md) for a flat, alphabetical reference of every term introduced across this reference.
