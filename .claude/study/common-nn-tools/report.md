# Common tools for neural-network development (late 2025 / early 2026)

Scope: tools organized by the stage of the `neural-lab` project ladder they serve. Ladder recap: 01 perceptron -> 02 MLP -> 03 micrograd -> 04 pytorch-mlp -> 05 cnn-vision -> 06 rnn-lm -> 07a/b transformer -> 08 tokenization -> 09 nanoGPT -> 10 data-pipeline -> 11 scaling-laws -> 12 distributed -> 13 finetune/align -> 14 inference-opt -> 15 evaluation.

Maturity-signal legend used throughout:
- **de facto standard** - ecosystem default; you need a specific reason not to use it
- **mainstream** - healthy user base, safe pick, not the only option
- **niche but production-grade** - specialist tool used in real production for its niche
- **educational / minimalist** - pedagogy over production
- **legacy / fading** - still works, has been displaced for new work

Verification dates for fast-moving entries: late October 2026 (see sources at end).

---

## 1. Core frameworks

| Tool | Pip / repo | Description | Ladder fit | Maturity | Reach-for signal |
|---|---|---|---|---|---|
| **PyTorch** | `torch` | Dynamic-graph tensor + autograd + nn library, CUDA/ROCm/MPS backends | 04 onward (every project after NumPy phase) | de facto standard | Default pick; everything in the LLM stack targets it first |
| **JAX** (+ Flax, Equinox) | `jax`, `flax`, `equinox` | Functional + XLA-compiled array programming; `jit`/`vmap`/`grad`/`pmap` primitives | Alternative path for 04-12; strong on 11 (scaling-laws) | mainstream | TPU targets, Google Research code, cleaner functional training loops |
| **TensorFlow** | `tensorflow` | Keras-fronted graph+eager framework; former standard | None required for this ladder | legacy / fading for research; mainstream in Google Cloud/mobile | Keras 3 cross-backend code, TFLite mobile targets, legacy production |
| **MLX** | `mlx`, `mlx-lm` | Apple-Silicon-only lazy-eval array framework; unified-memory design | 04-09 local prototyping on M-series Macs; 13/14 for local finetune+inference | niche but production-grade (within Apple ecosystem) | On Apple Silicon and never leaving it; `mlx-lm` for QLoRA/AWQ/GPTQ on M-series |
| **tinygrad** | `tinygrad` | <10k-LOC teaching-grade framework with real multi-backend support | Reference reading at 03-04 | educational / minimalist | Reading the source to understand what a framework actually is; not for shipping |

---

## 2. Model definition / high-level

- **torch.nn** - the native `Module`/`Parameter` layer inside PyTorch. Ladder fit: 04 onward. **de facto standard.** Reach for it by default.
- **PyTorch Lightning** (`lightning`) - training-loop + distributed + logging boilerplate removal on top of torch.nn. Ladder fit: 05-07 (clean loops), 12 (DDP/FSDP via Trainer). **mainstream.** Reach for it when a Trainer-shaped loop pays off; skip it when you're writing nanoGPT-style single-file training (09).
- **Keras 3** (`keras`) - multi-backend high-level API (TF/JAX/PyTorch backends). Ladder fit: none in this ladder (TF/multi-backend focus). **mainstream.** Reach for it in cross-framework teaching material or legacy codebases.
- **Flax** (`flax`) - JAX's neural-net library; `nnx` is the current recommended API. Ladder fit: JAX-path alternative for 04-12. **mainstream** within JAX world. Reach for it when you've chosen JAX for `vmap`/TPU reasons.
- **Equinox** (`equinox`) - JAX models as PyTrees; "callable dataclasses" style. **niche but production-grade.** Reach for it in JAX code when Flax's module system feels heavy.

---

## 3. Autograd / from-scratch learning tools

- **micrograd** (karpathy/micrograd) - ~150-line scalar autograd engine. Ladder fit: **project 03 (micrograd clone).** **educational / minimalist.** Reach for it to understand backprop; never for production.
- **tinygrad** (`tinygrad`) - see above; sits between micrograd and PyTorch pedagogically. Ladder fit: optional reading between 03 and 04.
- **autograd / HIPS** (`autograd`) - pre-JAX NumPy autograd from HIPS. Ladder fit: optional stop between 02 and 03 if you want array autograd without the GPU story. **legacy / fading** (JAX succeeded it). Reach for it only for historical interest.

---

## 4. Experiment tracking

| Tool | Pip / repo | Model | Ladder fit | Maturity | Reach-for signal |
|---|---|---|---|---|---|
| **Weights & Biases** | `wandb` | SaaS + optional self-host | 04 onward; essential by 11+ | de facto standard | Default pick for ML research tracking; best sweep UI |
| **MLflow** | `mlflow` | Open-source + self-host | 09+ if you care about model registry | mainstream | Enterprise self-host with registry + staging workflow |
| **TensorBoard** | `tensorboard` | Local file-based | 04 onward (free addition to any PyTorch loop) | mainstream (ubiquitous as a viewer) | Local scalar/image/graph visualization, zero-signup |
| **Neptune** | `neptune` | SaaS | 11+ for structured experiment metadata | mainstream | Preference for Neptune's UI/metadata model over W&B |
| **Aim** | `aim` | Local-first OSS | 04+ when you want W&B-shape UX without a SaaS | niche but production-grade | Air-gapped research, no outbound network |
| **Comet** | `comet-ml` | SaaS | 11+ | mainstream | Comet-specific feature need; otherwise W&B wins |

Default: TensorBoard from project 04, add W&B at project 09 when runs start costing money.

---

## 5. Hyperparameter tuning

- **Optuna** (`optuna`) - TPE/CMA-ES + pruning, framework-agnostic. Ladder fit: 05-08 (small-scale tuning). **de facto standard** for single-machine HPO.
- **Ray Tune** (`ray[tune]`) - distributed HPO + schedulers (ASHA, PBT). Ladder fit: 11+ when sweeps go multi-node. **mainstream.**
- **W&B Sweeps** - W&B-integrated grid/random/bayes. Ladder fit: 05+ if already on W&B. **mainstream.** Reach for it when the UI integration matters more than scheduler sophistication.
- **Hyperopt** (`hyperopt`) - original TPE library. **legacy / fading** - Optuna has replaced it for new work.

---

## 6. Data pipelines

| Tool | Pip / repo | What it is | Ladder fit | Maturity | Reach-for signal |
|---|---|---|---|---|---|
| **HuggingFace datasets** | `datasets` | Arrow-backed dataset library; streaming + memory-map | 06 (text corpora), 08-09, 15 | de facto standard | Default for text/vision benchmark datasets |
| **torchdata / `torch.utils.data`** | built-in | `Dataset`/`DataLoader` primitives | Every project 04+ | de facto standard | Default for custom data; `torchdata.datapipes` has been deprecated - use `IterableDataset` directly |
| **WebDataset** | `webdataset` | Tar-shard streaming format for S3/GCS | 10 (data-pipeline) | mainstream | Multi-TB corpora on object storage |
| **FFCV** | `ffcv` | Vision-specialized high-throughput loader | 05 (CIFAR/ImageNet throughput) | niche but production-grade | ImageNet-scale vision training where DataLoader is the bottleneck |
| **Mosaic Streaming** | `mosaicml-streaming` | Shard streaming for distributed training | 10, 12 | mainstream | Multi-node training with resumable streaming |
| **NVIDIA DALI** | `nvidia-dali-cuda*` | GPU-accelerated data loading pipeline | 05 (vision) | mainstream | DataLoader-bound and you have NVIDIA GPUs to spare |

---

## 7. Tokenization

- **tiktoken** (`tiktoken`) - OpenAI's fast BPE tokenizer; GPT-2/3.5/4 vocabs built in. Ladder fit: **08 (verify your BPE against theirs), 09 (nanoGPT), 15.** **de facto standard** for GPT-family tokenization.
- **sentencepiece** (`sentencepiece`) - Google's BPE/unigram implementation; LLaMA family uses it. Ladder fit: 08 (reference), 13 (finetuning LLaMA-family). **de facto standard** outside the OpenAI family.
- **HuggingFace tokenizers** (`tokenizers`) - Rust-backed, trains + encodes; underlies most HF model tokenizers. Ladder fit: 08-15. **de facto standard** inside the HF stack.
- **minbpe** (karpathy/minbpe) - Karpathy's teaching BPE in ~500 LOC. Ladder fit: **project 08 as the direct reference implementation.** **educational / minimalist.**

---

## 8. LLM training / finetuning

- **transformers** (`transformers`) - HF's model library (LLaMA, Qwen, Mistral, GPT-OSS, etc.). Ladder fit: 09 (optional reference), 13-15. **de facto standard.**
- **trl** (`trl`) - HF's SFT/DPO/GRPO/PPO trainers. Ladder fit: **project 13 (SFT->DPO/RLHF).** **de facto standard** for alignment work.
- **peft** (`peft`) - HF's LoRA/QLoRA/prefix-tuning library. Ladder fit: **project 13.** **de facto standard** for parameter-efficient finetuning.
- **bitsandbytes** (`bitsandbytes`) - 8-bit/4-bit optimizer + weight quantization kernels. Ladder fit: 13 (QLoRA), 14 (inference quantization). **de facto standard** for 4/8-bit on CUDA.
- **unsloth** (`unsloth`) - Triton-kernel-rewritten LoRA/QLoRA/full-finetune; 2x faster, ~70% less VRAM; MoE support (12x faster MoE forward); Unsloth Studio no-code UI (beta March 2026). Ladder fit: **project 13 as the fast path.** **mainstream.** Reach for it when single-GPU VRAM is the binding constraint.
- **axolotl** (`axolotl`) - YAML-driven finetuning wrapper; FSDP2 + DeepSpeed multi-node; ScatterMoE + fused LoRA Triton (0.16, April 2026); CPU-staged NF4 for FSDP2 QLoRA (0.20, September 2026). Ladder fit: project 13 for multi-node runs. **mainstream.** Reach for it for configurable multi-GPU/multi-node finetune.
- **torchtune** (`torchtune`) - **DEPRECATED**: active development stopped July 15, 2025; last release April 2025. Do not pick for new work. If citing in docs, mark as deprecated and point to axolotl or unsloth.

---

## 9. Distributed training

- **torch.distributed - DDP** (`torch.nn.parallel.DistributedDataParallel`) - one-process-per-GPU data parallel. Ladder fit: **project 12 phase 1.** **de facto standard** for data parallel.
- **torch.distributed - FSDP** (`torch.distributed.fsdp.FullyShardedDataParallel`, the FSDP1 class). Ladder fit: project 12 phase 2 reference. **mainstream but superseded** - see FSDP2.
- **torch.distributed - FSDP2** (`torch.distributed.fsdp.fully_shard`) - DTensor-based rewrite with better memory determinism, composability with tensor/pipeline parallel. Ladder fit: **project 12 phase 2+**, 13 (axolotl 0.20+ uses it). **de facto standard** for new PyTorch sharded training; FSDP1 is in maintenance.
- **DeepSpeed** (`deepspeed`) - Microsoft's ZeRO 1/2/3 + offload + pipeline parallel. Ladder fit: project 12 phase 3 as alternative to FSDP2. **mainstream.** Reach for it for ZeRO-Offload (CPU/NVMe offload) and MoE-specific optimizations.
- **Megatron-LM** (NVIDIA/Megatron-LM) - tensor + pipeline + sequence parallel for >=100B models. Ladder fit: project 12 phase 4 (tensor parallel). **niche but production-grade.** Reach for it at the >=70B-parameter scale.
- **Accelerate** (`accelerate`) - HF's thin wrapper that picks DDP/FSDP/DeepSpeed from a config. Ladder fit: 12 (easy orchestration), 13 (trl/axolotl use it). **mainstream.**

---

## 10. Inference / serving

| Tool | Pip / repo | Role | Ladder fit | Maturity | Reach-for signal |
|---|---|---|---|---|---|
| **vLLM** | `vllm` | PagedAttention-based LLM serving; V1 engine default since 2025; 0.32.0 stable Oct 2026 | **project 14 (reference)**, 15 (eval harness backend) | de facto standard | General-purpose LLM serving; broadest hardware + model coverage |
| **SGLang** | `sglang` | RadixAttention + structured decoding frontend; ~29% faster than vLLM on H100; 3.1x faster on DeepSeek V3 | project 14 (compare), production serving with shared-prefix workloads | mainstream | Agent loops, multi-turn chat, RL rollouts, long shared prefix |
| **TensorRT-LLM** | `tensorrt-llm` | NVIDIA's compiler+runtime for LLM inference on Hopper/Blackwell | project 14 for max NVIDIA throughput | niche but production-grade | NVIDIA-only, max-perf production |
| **llama.cpp** | `llama.cpp` (C++) | CPU/Metal/CUDA inference with GGUF quantized weights | **project 14 (CPU/edge path)** | de facto standard for local/CPU | Local macOS/Windows, edge, GGUF weights |
| **Ollama** | `ollama` | llama.cpp wrapper with model-manager UX | Local play alongside 14 | mainstream | One-command local LLM, no coding ceremony |
| **Triton Inference Server** | NVIDIA Triton | Multi-backend model server (not vLLM-specific) | Production of any model at 14 | mainstream | Multi-model serving, GRPC, dynamic batching |
| **TorchServe** | `torchserve` | PyTorch-native model server; now in maintenance mode | 05/14 if sticking to pure PyTorch | legacy / fading | Legacy pure-PyTorch deployment |
| **Ray Serve** | `ray[serve]` | Python-defined serving on Ray | 14 at app layer | mainstream | Python-first, multi-stage pipelines |
| **BentoML** | `bentoml` | Python-first "ML app" packaging + serving | 14 app layer | mainstream | Full-stack ML API with batteries |

---

## 11. Quantization

- **bitsandbytes** (`bitsandbytes`) - 4/8-bit on CUDA; NF4 for QLoRA. Ladder fit: 13 (QLoRA), 14. **de facto standard** for 4/8-bit finetune.
- **GPTQ / auto-gptq** (`auto-gptq`) - post-training 4-bit weight quantization. Ladder fit: 14. **mainstream.** Reach for it for cheap post-hoc 4-bit on CUDA.
- **AWQ** (`autoawq`) - activation-aware weight quantization; often beats GPTQ perplexity. Ladder fit: 14. **mainstream.** Reach for it when AWQ-quantized weights for your target model already exist.
- **GGUF (via llama.cpp)** - K-quants + i-quants (Q4_K_M, Q5_K_M, IQ4_XS...); CPU/Metal/CUDA. Ladder fit: **project 14 CPU/local path.** **de facto standard** for local inference quantization.
- **torchao** (`torchao`) - PyTorch-native quant + float8 training; works with torch.compile + FSDP2; 0.17/0.18 adds MXFP8 MoE, per-head FP8 attention (2026). Ladder fit: **project 11/12 (float8 training), 14 (inference quant).** **mainstream and ascendant** - the PyTorch-native path.

---

## 12. Compilation / graph optimization

- **torch.compile** (built-in) - TorchDynamo + Inductor; production-stable, broad model support, works with FSDP2 + torchao. Ladder fit: **09 onward where perf matters.** **de facto standard** for PyTorch speedup. Reach for it by default; disable per-model only when it breaks.
- **TensorRT** (NVIDIA) - ahead-of-time graph compiler to NVIDIA kernels. Ladder fit: 14 for max NVIDIA inference perf. **mainstream.**
- **ONNX / ONNX Runtime** (`onnx`, `onnxruntime`) - interchange format + cross-platform runtime. Ladder fit: 05/14 when deploying off-GPU/off-PyTorch. **mainstream** (fading as a research target, stable as a deployment target).
- **OpenVINO** (`openvino`) - Intel's inference compiler (CPU/iGPU/NPU). Ladder fit: 14 if Intel is the target. **niche but production-grade.**

---

## 13. Profiling

- **torch.profiler** (built-in) + Chrome trace - the primary PyTorch profiler; emits Perfetto/Chrome traces. Ladder fit: **05 onward, essential by 11.** **de facto standard.**
- **Nsight Systems** (NVIDIA) - system-level timeline (GPU + CPU + CUDA API). Ladder fit: 12/14 when you've exhausted torch.profiler. **mainstream.**
- **Nsight Compute** (NVIDIA) - single-kernel-level perf counters. Ladder fit: 12/14 when writing/tuning a kernel. **niche but production-grade.**
- **py-spy** (`py-spy`) - sampling profiler for Python processes; works on running processes. Ladder fit: 10 (data pipeline bottlenecks), 14 (server bottleneck). **mainstream.**
- **nvidia-smi** / `nvtop` - baseline GPU utilization + VRAM. Ladder fit: every CUDA project. **de facto standard** for a quick glance.

---

## 14. Evaluation

- **lm-evaluation-harness** (EleutherAI) - 60+ benchmark suites, hundreds of subtasks, 25+ model backends, declarative YAML. Ladder fit: **project 15.** **de facto standard** for LLM accuracy benchmarking (MMLU, HellaSwag, HumanEval, GSM8K are all task configs here).
- **HELM** (Stanford CRFM) - multi-metric, multi-scenario holistic eval (accuracy + calibration + robustness + fairness + bias + toxicity + efficiency). Ladder fit: project 15 for a *holistic* report rather than single scores. **niche but production-grade** (research-grade, slower-moving).
- Note: "EleutherAI eval harness" is just another name for lm-evaluation-harness - same project.

---

## 15. Shape / tensor manipulation

- **einops** (`einops`) - `rearrange` / `reduce` / `repeat` with named-axis strings. Ladder fit: **05 onward, essential by 07 (transformer).** **de facto standard** for readable tensor reshaping.
- **einx** (`einx`) - extension of einops with einsum-like generalization. **niche but production-grade.** Reach for it when einops is close but not expressive enough.
- **jaxtyping** (`jaxtyping`) - runtime+static tensor shape annotations; works with PyTorch too. Ladder fit: 07+ for self-documenting shapes. **mainstream.**
- **torchtyping** (`torchtyping`) - predecessor to jaxtyping; **legacy / fading** - new work uses jaxtyping.

---

## 16. Visualization and model inspection

- **TensorBoard** - see section 4. Dual purpose as both tracker and viewer.
- **Netron** (`netron`) - static graph/weight viewer for ONNX/TorchScript/safetensors files. Ladder fit: 05, 14. **mainstream.** Reach for it when you need to see a saved model's structure.
- **torchview** (`torchview`) - graphviz-style forward-pass visualization of a torch.nn.Module. Ladder fit: 04-07. **niche but production-grade.** Good teaching aid.
- **matplotlib** (`matplotlib`) - the universal plotting library. Ladder fit: every project (loss curves, confusion matrices, attention heatmaps). **de facto standard.**
- **plotly** (`plotly`) - interactive HTML plots. Ladder fit: optional at 15 for shareable eval dashboards. **mainstream.**

---

## 17. Interactive demos

- **gradio** (`gradio`) - one-file Python -> shareable web UI. Ladder fit: 05, 09, 13 for quick demos. **de facto standard** for HF-ecosystem demos.
- **streamlit** (`streamlit`) - script-top-down Python -> web app. Ladder fit: 05/15 for internal dashboards. **mainstream.**
- **panel** (`panel`) - Holoviz-stack Python -> web dashboards; richer composition than streamlit. **niche but production-grade.** Reach for it when streamlit's single-script model constrains you.

---

## 18. Dev tooling for NN projects

- **uv** (astral-sh/uv) - Rust-written Python package + env manager; replaces pip/pip-tools/virtualenv/pyenv/pipx; ~75-128M monthly PyPI downloads mid-2026; **OpenAI acquired Astral in March 2026**. Ladder fit: every project. **de facto standard for new Python work in 2026.** Already in the repo; keep using it.
- **ruff** (`ruff`) - Rust-written linter + formatter (replaces flake8 + black + isort). Already in the repo. **de facto standard.**
- **mypy** (`mypy`) - static type checker. Already in the repo. **de facto standard.** (ty / pyrefly are rising alternatives - watch, don't switch yet.)
- **pytest** (`pytest`) - test runner. Already in the repo. **de facto standard.**
- **pre-commit** (`pre-commit`) - git-hook harness for ruff/mypy/etc. Ladder fit: every project. **de facto standard.**
- **jupyter** (`jupyter`) - notebook runtime; Colab's native format. Ladder fit: 01-09 for exploration, 11+ for Colab/kernel work on managed GPUs. **de facto standard** for notebooks.
- **marimo** (`marimo`) - reactive Python notebook; stores as `.py`; DAG-based cell execution; VSCode/Cursor/PyCharm integration shipped 2026. Ladder fit: alternative to Jupyter for 01-08 where version control + reproducibility matter. **mainstream and rising.** Reach for it when the notebook will be committed to git; stay on Jupyter for Colab GPU runs.

---

## 19. Compute providers (relevant to projects 09+)

| Provider | Model | Best for ladder | Price shape | Notes |
|---|---|---|---|---|
| **Colab / Colab Pro+** | SaaS notebook, A100/L4/T4 | 05, 07, 09 (first LLM run) | $0 free tier, ~$50/mo Pro+ | Easiest first-time GPU; session caps still bite |
| **RunPod** | On-demand + spot pods | 09, 11, 13 | H100 ~$2-3/hr on-demand | Jupyter + SSH pods; community cloud is cheap/variable |
| **Lambda Labs** | On-demand + reserved | 11, 12, 13 | H100/H200 reserved cheaper than hyperscalers | Clean "AI-focused" DX; waitlists during capacity crunches |
| **Vast.ai** | Marketplace (peer-to-peer) | 09, 13 | Cheapest spot; variable reliability | Price-sensitive single-GPU jobs you can resume |
| **Modal** | Serverless Python functions on GPU | 13 (short runs), 14 (serving) | Pay per second; H100 ~$4/hr | Python-first, no container ceremony; cold-starts matter |
| **Together / Replicate** | Hosted model inference API | 15 (as an eval target), demos | Per-token / per-run | You want the inference, not the GPU |
| **Paperspace / Core** | On-demand notebooks + VMs | 05, 09 | Mid-tier pricing | DigitalOcean-owned; declining mindshare vs RunPod/Lambda |

---

## 20. Model hubs

- **HuggingFace Hub** (`huggingface_hub`) - the model + dataset + Space hub; `transformers`/`diffusers`/`datasets` all pull from it. Ladder fit: 08-15. **de facto standard.**
- **PyTorch Hub** (`torch.hub`) - curated model-loading entry points; small and old. Ladder fit: 05 (torchvision models load this way under the hood). **legacy / fading.**
- **Civitai** - Stable-Diffusion-era image-model community hub. Ladder fit: none in this ladder (image gen out of scope). **mainstream** within the SD community.

---

## Summary of ladder-specific picks

- **Projects 01-03 (NumPy foundations):** NumPy, matplotlib, pytest, ruff, mypy, uv, marimo (optional for reactive exploration), micrograd as reference. No framework yet.
- **Projects 04-05 (PyTorch intro):** torch, torch.nn, torchvision, TensorBoard, einops, torch.compile, torchview.
- **Projects 06-07 (sequences + transformer):** add einops (essential), jaxtyping, W&B.
- **Project 08 (tokenization):** minbpe (reference), tiktoken (verify), sentencepiece.
- **Project 09 (nanoGPT):** stays single-file; add W&B, torch.compile, Colab/RunPod.
- **Project 10 (data pipeline):** HuggingFace datasets, WebDataset or Mosaic Streaming, py-spy for bottleneck hunting.
- **Project 11 (scaling laws):** W&B essential; torch.profiler; torchao for float8 training.
- **Project 12 (distributed):** torch.distributed DDP -> FSDP2 (`fully_shard`) -> DeepSpeed -> Megatron-LM (tensor parallel). Accelerate for config orchestration.
- **Project 13 (finetune/align):** transformers + trl + peft + bitsandbytes; **unsloth** for single-GPU speed, **axolotl** for multi-node YAML. (Do not use torchtune - deprecated July 2025.)
- **Project 14 (inference opt):** vLLM (reference), SGLang (shared-prefix workloads), llama.cpp + GGUF (CPU/local), torchao (PyTorch-native quant), AWQ/GPTQ for post-training 4-bit on CUDA.
- **Project 15 (evaluation):** lm-evaluation-harness (primary), HELM (holistic report), gradio for a shareable demo.

---

## Notable state-of-the-world facts (late 2026) worth knowing

- **torchtune is deprecated.** Active development stopped July 15, 2025. Any tutorial recommending it is stale. Replace with axolotl (multi-node) or unsloth (single-GPU speed).
- **FSDP2 is the new default.** `torch.distributed.fsdp.fully_shard` has superseded the FSDP1 class for new code; FSDP1 is in maintenance. axolotl 0.20 (September 2026) adopted FSDP2 for CPU-staged NF4 QLoRA.
- **vLLM V1 engine is default.** The V1 rewrite (default since 2025) brought near-zero-overhead prefix caching, chunked prefill, and torch.compile integration. 0.32.0 stable as of October 2026.
- **SGLang has caught vLLM on raw throughput** (~29% faster on H100 for independent prompts) and is **substantially** faster for shared-prefix workloads (agent loops, multi-turn, RL rollouts).
- **Astral (uv, ruff) was acquired by OpenAI in March 2026.** uv is now at ~128M monthly PyPI downloads and is the de facto new Python package manager. Keep using it.
- **torchao is the ascendant PyTorch-native quantization path.** Works out of the box with torch.compile + FSDP2; adds float8 training (up to 1.5x end-to-end speedup at 405B/512-GPU scale).
- **Unsloth Studio** (no-code finetuning web UI) launched beta March 2026; the underlying library hits 2x faster / 70% less VRAM / 12x faster for MoE.
- **MLX** has matured significantly on Apple Silicon: v0.32 adds fused attention, distributed inference across multiple Macs via Thunderbolt/RDMA, and `mlx-lm` ships AWQ/GPTQ/DWQ CLIs.
- **torchdata datapipes are deprecated** - new code should use plain `IterableDataset` and build the pipeline in Python.
- **Hyperopt and torchtyping are both effectively superseded** (by Optuna and jaxtyping respectively) - flag them as `legacy / fading` in any new recommendation.

---

## Sources (verification searches run late October 2026)

- vLLM V1 blog: https://vllm.ai/blog/2025-01-27-v1-alpha-release
- What Is vLLM? 2026: https://www.yottalabs.ai/post/what-is-vllm-architecture-performance-and-why-teams-use-it-for-llm-inference
- vLLM vs SGLang vs LMDeploy 2026: https://www.premai.io/blog/vllm-vs-sglang-vs-lmdeploy-fastest-llm-inference-engine-in-2026/
- SGLang vs vLLM 2026 (Morph): https://www.morphllm.com/comparisons/sglang-vs-vllm
- Red Hat: vLLM vs SGLang (Oct 2026): https://developers.redhat.com/articles/2026/10/14/vllm-vs-sglang-choosing-the-right-inference-engine-for-your-workload
- Unsloth Changelog: https://unsloth.ai/docs/new/changelog
- Spheron: Axolotl vs Unsloth vs TorchTune 2026: https://www.spheron.network/blog/axolotl-vs-unsloth-vs-torchtune/
- torchtune GitHub: https://github.com/meta-pytorch/torchtune
- Axolotl Releases: https://github.com/axolotl-ai-cloud/axolotl/releases
- FSDP2 Tutorial: https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- FSDP docs (main): https://pytorch.org/docs/main/fsdp.html
- torchao 0.17 docs: https://docs.pytorch.org/ao/stable/workflows/training.html
- PyTorch blog on torchao: https://pytorch.org/blog/pytorch-native-architecture-optimization/
- uv 2026 state: https://dev.to/moksh/why-uv-became-the-go-to-python-package-manager-in-2026-2kag
- State of Python packaging 2026: https://repoforge.io/blog/posts/the-state-of-python-packaging-in-2026/
- MLX (Wikipedia): https://en.wikipedia.org/wiki/MLX_(Apple_ML_framework)
- MLX distributed inference (Apple): https://machinelearning.apple.com/research/mlx-distributed-inference
- LLM Eval Harness 2026: https://www.morphllm.com/llm-eval-harness
- Marimo vs Jupyter 2026: https://www.refontelearning.com/blog/marimo-vs-jupyter-reactive-notebooks
