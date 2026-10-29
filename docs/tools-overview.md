# Neural network tooling landscape

What you reach for at each stage of the project ladder. Written to answer "what do I actually need to install and learn before project N?" rather than to be an exhaustive catalog.

Full tool-by-tool survey (~110 tools across 20 categories) lives at `.claude/study/common-nn-tools/report.md` — scan that when you need a specialist tool not covered below.

Last reviewed: 2026-10-29. The LLM half of the stack moves fast; re-verify any specific recommendation before project 12 onward.

## Reading this doc

Each stage lists the tools you need and the ones you don't. "Don't need" is as useful as "need" — ecosystem tourism is the main failure mode for a learning repo.

Maturity labels used throughout:

- **Standard** — ecosystem default; you need a specific reason not to use it.
- **Mainstream** — healthy user base, safe pick, not the only option.
- **Niche** — specialist tool, used in real production for its niche.
- **Educational** — pedagogy over production.
- **Legacy** — still works, has been displaced for new work. Avoid for new code.

## Stages 01–03 — Foundations (NumPy only)

**What you need:** NumPy, matplotlib, pytest. That's it.

**What you don't need:** PyTorch, Jupyter for the artifact (notebooks are fine for exploration, but the committed artifact is `.py`). No autograd library — the whole point of 02 and 03 is to hand-derive and hand-implement the chain rule.

**Reference reading (don't install, just read the source):**

- [`karpathy/micrograd`](https://github.com/karpathy/micrograd) — ~150 lines of scalar autograd. Project 03 is a clone of this.
- [`HIPS/autograd`](https://github.com/HIPS/autograd) — pre-JAX NumPy autograd. Historical interest; shows what JAX replaced.
- [`tinygrad`](https://github.com/tinygrad/tinygrad) — <10k LOC teaching-grade framework with real multi-backend support. Sits pedagogically between micrograd and PyTorch.

## Stages 04–05 — Frameworks (PyTorch arrives)

**What you need:** PyTorch (`torch`), torchvision, pytest. Still no Lightning, no W&B, no HuggingFace. The point of 04 is to learn `nn.Module` + `optim` + `autograd.grad` natively — a Trainer abstraction obscures what you're trying to see.

**Framework landscape at a glance:**

| Framework | Status | When to reach for it |
|---|---|---|
| **PyTorch** | Standard | Default pick; everything in the LLM stack targets it first |
| **JAX** (+ Flax, Equinox) | Mainstream | TPU targets; Google Research code; functional training loops |
| **TensorFlow** | Legacy for research, mainstream in TFLite/mobile | Keras 3 cross-backend code; legacy production |
| **MLX** | Niche (Apple-only) | Local prototyping on M-series Macs; local finetune/inference without renting a GPU |
| **tinygrad** | Educational | Reading the source to understand what a framework is |

This ladder stays on PyTorch from 04 onward. The alternatives are listed so you know what you're not choosing.

**Compile path:** `torch.compile` (TorchDynamo + Inductor backend) is PyTorch's native graph-capture + fusion compiler. Default-on in PyTorch 2.x. Project 05 is a reasonable first place to try `model = torch.compile(model)` and measure the speedup. Legacy `torch.jit.script` / `torch.jit.trace` has been superseded.

## Stage 06 — Sequences (RNN land)

**What you need:** everything from the previous stage, plus access to a dataset of Shakespeare-style text (bundled in the project, no new install).

**Shape-manipulation libraries worth knowing from here on:**

- [`einops`](https://github.com/arogozhnikov/einops) — rearrange/reduce/repeat with named-axis strings. Standard. Makes attention reshapes readable (`b n (h d) -> b h n d`). Install at project 07a.
- `einx` — newer, more general einops variant. Niche.
- `jaxtyping` — shape/dtype type annotations that work in both PyTorch and JAX. Mainstream. Catches shape bugs at import time.
- `torchtyping` — the older PyTorch-only variant. Legacy; use `jaxtyping` instead.

## Stages 07–09 — Transformers through nanoGPT

This is where the tooling actually widens. You install:

- `einops` — attention reshapes are miserable without it.
- `tiktoken` (OpenAI) — the industrial BPE tokenizer; needed in 08 as the baseline to compare your from-scratch BPE against.
- `sentencepiece` (Google) — the other industrial tokenizer, used by LLaMA/T5/Mistral.
- `tokenizers` (Hugging Face, Rust-backed) — the HF ecosystem's tokenizer; faster than tiktoken for BPE training.

**Tokenization comparison:**

| Tool | Backend | Strength | Where it shows up |
|---|---|---|---|
| `minbpe` (Karpathy) | Pure Python | Readable reference impl | Project 08 reference reading |
| `tiktoken` | Rust via Python | GPT-2/3/4/5 compatibility | Project 08 baseline |
| `sentencepiece` | C++ | LLaMA/T5 compatibility | Loading pretrained LLM tokenizers |
| `tokenizers` | Rust | Fast training + full HF integration | Project 09+ when loading HF models |

**By project 09 you also meet Hugging Face.** You will load `GPT-2` from `transformers` as a comparison baseline against your nanoGPT. See `docs/huggingface.md` for the full orientation.

**Experiment tracking becomes worth it around 09:**

| Tool | Model | When to pick it |
|---|---|---|
| **TensorBoard** | Local files + web UI | Zero-friction start; good enough for 05–08 |
| **Weights & Biases** (`wandb`) | SaaS (free for public) | Standard pick from 09 onward; best sweep UI; shareable runs |
| **MLflow** | Self-hosted | Enterprise/compliance where SaaS is off the table |
| **Aim** | OSS self-hosted | Open-source alternative to W&B with similar UX |
| **Neptune / Comet** | SaaS | Alternatives to W&B; pick one and stick with it |

Default recommendation: TensorBoard through 08, switch to W&B at 09 when you want to compare runs across sweeps.

## Stage 10 — Data pipelines

**What you need:** `datasets` (Hugging Face). It's streamable, Arrow-backed, and the de facto standard for anything bigger than fits in RAM.

**Current data pipeline landscape:**

| Tool | Status | Signal |
|---|---|---|
| `datasets` (HF) | Standard | Default pick for text + multimodal corpora |
| `torchdata` DataPipes | **Legacy** | Deprecated path. Use plain `torch.utils.data.IterableDataset` instead |
| `WebDataset` | Niche | Tar-file sharded datasets; large-scale image/video |
| `FFCV` | Niche | Fast throughput for ImageNet-style vision workloads |
| Mosaic streaming (`streaming`) | Niche | Databricks/Mosaic shop; cloud-blob-backed streaming |
| NVIDIA DALI | Niche | GPU-accelerated data augmentation on Nvidia hardware |

**One gotcha:** `torchdata` DataPipes were PyTorch's bet on a composable dataset API; the project was de-emphasized. Don't learn DataPipes — write plain `IterableDataset` subclasses.

## Stage 11 — Scaling laws

**What you add:** hyperparameter sweeps. The Chinchilla mini-experiment is the point where you run dozens of training jobs and need to compare them.

**HPO landscape:**

| Tool | Status | Signal |
|---|---|---|
| `optuna` | Standard | Default for Python HPO; TPE sampler is good enough for most work |
| Ray Tune | Mainstream | When you're already on Ray; distributed trials across a cluster |
| W&B Sweeps | Mainstream | If you're already using W&B; nice UI, less flexible than Optuna |
| Hyperopt | **Legacy** | Older TPE library. Optuna is strictly better. |

**Compute reality for this stage:** rented A100 ($1–3/hr on RunPod, Lambda Labs, Vast.ai, Modal). Colab Pro+ gets you an A100 for shorter sessions. Verify current prices before committing.

## Stage 12 — Distributed training

The distributed landscape has moved. Current state:

| API | Status | Use case |
|---|---|---|
| `torch.distributed` **DDP** | Standard | Data parallelism; model fits on one GPU, scale by throwing more at it |
| `torch.distributed.fsdp.fully_shard` (**FSDP2**) | Standard | ZeRO-style sharded data parallel. Model too big for one GPU. **Recommended API** |
| `FullyShardedDataParallel` (FSDP1 class) | Legacy | Superseded by `fully_shard`; still works but new code should use FSDP2 |
| DeepSpeed | Mainstream | ZeRO stages 1/2/3 + CPU offload + MoE-friendly. Used by Microsoft and the OpenAssistant stack |
| Megatron-LM | Niche | Tensor parallelism + pipeline parallelism for very large models |
| `accelerate` (Hugging Face) | Mainstream | Thin wrapper over `torch.distributed` + DeepSpeed + FSDP. Hides boilerplate |

**Recommended ladder path:** write a plain DDP loop first, then rewrite with FSDP2 (`fully_shard`), then once more using `accelerate` to see the abstraction. Megatron-LM is read-only — study the tensor-parallel code; don't try to use it on a 2-GPU budget.

## Stage 13 — Finetune and align

**What you install:**

- `transformers` (HF) — model definitions.
- `peft` (HF) — LoRA/QLoRA/adapters/prefix tuning.
- `bitsandbytes` — 4-bit/8-bit quantized optimizers and linear layers (enables QLoRA).
- `trl` (HF) — SFT, DPO, GRPO, PPO trainers.

Optional but increasingly standard:

- `unsloth` — 2–5× faster LoRA/QLoRA finetuning through hand-written CUDA kernels. Use when you want finetune iteration speed.
- `axolotl` — YAML-config finetuning harness that wraps `transformers` + `peft` + `trl`. Use when you want reproducibility over writing training code.

**One gotcha that will bite:** `torchtune` was PyTorch Foundation's native finetuning library. Active development stopped July 2025. **Don't start new code against it.** The live stack is `trl` + `peft`, with `unsloth` or `axolotl` on top for speed.

**Raw `transformers.Trainer` is a slow path for 7B+ finetunes.** It's optimized for correctness and coverage, not throughput. For actual production finetuning, you layer `unsloth` or `axolotl` on top.

## Stage 14 — Inference optimization

**Inference server landscape (late 2025 / early 2026):**

| Server | Status | Pick it when |
|---|---|---|
| **vLLM** | Standard | Default LLM inference server; broadest model coverage; paged attention |
| **SGLang** | Mainstream | Shared-prefix workloads (agents, few-shot). Beats vLLM ~29% on raw H100 throughput; 3.1× on DeepSeek V3 |
| **TensorRT-LLM** | Niche (Nvidia-only) | Maximum throughput on Nvidia hardware; longer build cycle |
| **llama.cpp** | Niche | CPU/consumer-GPU inference; GGUF quantization. Standard for local LLMs |
| **Ollama** | Mainstream | `llama.cpp` wrapper with model management. Default for "run an LLM on my laptop" |
| **Triton Inference Server** | Niche (Nvidia) | Production serving of mixed workloads (NN + classical ML) |
| **TorchServe** | Legacy-ish | PyTorch's native server; largely displaced by vLLM/Triton |

**Quantization at this stage:**

| Format / tool | Status | Signal |
|---|---|---|
| `bitsandbytes` | Standard | 4-bit/8-bit in PyTorch; the QLoRA path at stage 13 |
| GPTQ (`auto-gptq`) | Mainstream | 4-bit post-training quantization; good quality retention |
| AWQ | Mainstream | Activation-aware weight quantization; often outperforms GPTQ |
| GGUF (`llama.cpp`) | Standard for local inference | 1-bit through 8-bit; the format Ollama/LM Studio consume |
| `torchao` | Emerging | PyTorch-native quantization + sparsity library |

**KV cache + speculative decoding** are built into vLLM and SGLang. Project 14 reimplementing them from scratch is pedagogy; you won't ship that code.

## Stage 15 — Evaluation

**What you install:** `lm-evaluation-harness` (EleutherAI). It's the standard for running the MMLU/HellaSwag/HumanEval/GSM8K suite reproducibly.

- `lm-evaluation-harness` (EleutherAI) — Standard. Covers ~200 benchmarks.
- HELM (Stanford) — Mainstream. Broader scope; slower; more benchmarks.
- `evaluate` (HF) — Mainstream. Metric computation library; smaller surface than the harness.

## Dev tooling (used across every stage)

Already in `pyproject.toml`:

- `uv` — Standard. Package + lockfile + Python-version manager. OpenAI acquired Astral (makers of uv and ruff) in March 2026, so pace of development remains high.
- `ruff` — Standard. Linter + formatter. Replaces black + isort + flake8.
- `mypy` — Standard. Type checker. Running in `strict` mode per `pyproject.toml`.
- `pytest` — Standard. Test runner.

Worth considering:

- `pre-commit` — Mainstream. Git hook orchestration for lint/format on commit.
- `jupyter` — Standard for exploration. The repo convention is that notebooks are for exploration; `.py` is the artifact.
- `marimo` — Emerging. Reactive notebook alternative to Jupyter; outputs are plain `.py`. Worth watching.

## Interactive demo tools (optional)

For when you want to share a trained model with someone:

- `gradio` (HF) — Standard. Fast way to wrap a model in a web UI. Deploy to HF Spaces.
- `streamlit` — Mainstream. Richer apps; longer learning curve than Gradio.
- `panel` — Niche. HoloViz-ecosystem alternative.

## Compute providers (relevant from project 09 onward)

| Provider | Model | Signal |
|---|---|---|
| **Colab / Colab Pro+** | SaaS, per-session | $10–50/mo. Good through project 09; limited for 10+ |
| **RunPod** | On-demand GPU rental | $1–3/hr A100; standard pick for projects 10–12 |
| **Lambda Labs** | On-demand GPU rental | Comparable to RunPod; sometimes cheaper for H100 |
| **Vast.ai** | GPU marketplace | Cheapest on-demand; variable availability |
| **Modal** | Serverless GPU + Python API | Pay-per-second; good for sweeps and batch inference |
| **Together / Replicate** | Hosted LLM inference | API-call pricing; not for training |
| **Paperspace** | On-demand + managed notebooks | Nicer DX than RunPod; moderate premium |

**For this ladder:** Colab through 09, RunPod/Lambda for 10–12. By 12 you need an 8×A100 or 8×H100 node ($10–30/hr), which is where the "learn vs cost" trade bites hardest.

## See also

- `docs/huggingface.md` — Hugging Face as company, hub, and library family.
- `.claude/study/common-nn-tools/report.md` — full ~110-tool survey organized by category.
- `README.md` — the project ladder itself.
