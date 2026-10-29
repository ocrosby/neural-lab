# Hugging Face

Three things wearing the same name: a company, a hosting platform, and a family of Python libraries. All three matter from project 08 onward in the neural-lab ladder.

Last reviewed: 2026-10-29. Hugging Face moves fast; verify specific library APIs before relying on them in long-lived code.

## 1. The company

Founded 2016 in New York as a chatbot startup. Pivoted around 2018 when they open-sourced `pytorch-pretrained-bert` — which became `transformers` — and that library ate the NLP world. Offices in NYC and Paris. Investors include Nvidia, Google, Amazon, and Salesforce.

The company sells managed inference endpoints and private spaces; the libraries and public Hub are free and open-source.

## 2. The Hub

`huggingface.co` is a Git-backed hosting platform for three asset types. Every asset is a Git repository (large files via LFS) with a model card, config, and version history.

| Asset | What it stores | Example |
|---|---|---|
| **Models** | Pretrained weights + config + tokenizer | `meta-llama/Llama-3.1-8B` |
| **Datasets** | Training/eval corpora in a standard format, streamable | `HuggingFaceFW/fineweb` |
| **Spaces** | Hosted Gradio/Streamlit demo apps | `stabilityai/stable-diffusion` |

Effectively "GitHub for ML artifacts." Public repos are free; private repos and inference endpoints are paid. The CLI is `huggingface-cli`; most interaction happens through the libraries below.

Authentication: `huggingface-cli login` writes a token to `~/.cache/huggingface/token`. Any library that needs to pull a gated model (Llama, Gemma, etc.) reads from there.

## 3. The library family

All `pip install`-able, all pulling from the Hub by default. These are the ones you'll touch in this ladder:

| Library | What it does | First landing in the ladder |
|---|---|---|
| `transformers` | Reference implementations of ~every published architecture (BERT, GPT, LLaMA, ViT, Whisper, …) with a unified API | Reference reading at 07b; comparison baseline at 09 |
| `datasets` | Streaming + Arrow-backed dataset loader | Project 10 (data-pipeline) |
| `tokenizers` | Rust-backed BPE/WordPiece/Unigram tokenizers | Project 08 (tokenization) — industrial comparison to your from-scratch BPE |
| `accelerate` | Thin wrapper over `torch.distributed` + DeepSpeed + FSDP that handles device placement | Project 12 (distributed) |
| `peft` | Parameter-efficient finetuning — LoRA, QLoRA, adapters, prefix tuning | Project 13 (finetune/align) |
| `trl` | SFT, DPO, GRPO, PPO trainers built on `transformers` + `accelerate` | Project 13 (align) |
| `evaluate` | Metric computation library | Project 15, as one option alongside `lm-evaluation-harness` |
| `diffusers` | Diffusion model pipelines (Stable Diffusion, FLUX, etc.) | Out of scope for this ladder |

### `transformers` in two sentences

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")  # downloads from the Hub
```

The `Auto*` classes dispatch to the right architecture based on the config in the Hub repo. Project 09 uses this exact pattern to load GPT-2 as a baseline against your nanoGPT.

### `datasets` in two sentences

```python
from datasets import load_dataset

ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
for example in ds.take(10):
    print(example["text"][:200])
```

`streaming=True` avoids downloading the full corpus. Essential by project 10 — FineWeb is ~15TB and will not fit on your laptop.

### `tokenizers` in two sentences

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel()
tok.train(["corpus.txt"], trainers.BpeTrainer(vocab_size=50000))
```

Rust-backed, so training is 10–100× faster than a pure-Python BPE. Project 08 pits this against your from-scratch implementation.

## Why Hugging Face matters for neural-lab specifically

By project 08 you're loading pretrained tokenizers from the Hub to compare against your own. By 09 you're loading GPT-2 to compare against your nanoGPT. By 13 the entire SFT→DPO path is `transformers` + `peft` + `trl`.

**The Hub is effectively the ambient infrastructure for everything from project 08 onward** — not optional, not avoidable. The ladder choice is: build each stage from scratch first (your implementation), then compare to the HF reference. The HF implementation is the "ground truth" — if your transformer doesn't match a HF forward pass numerically, your transformer is wrong.

## Gotchas worth naming now

**`transformers` is a reference library, not a training library.** It's optimized for correctness and architecture coverage, not throughput. For actual LLM finetuning speed, you layer `unsloth` or `axolotl` on top:

- `transformers.Trainer` for a 7B QLoRA finetune: slow path.
- `unsloth` + `peft` + `trl` for the same job: 2–5× faster via hand-written CUDA kernels.
- `axolotl` wrapping `transformers` + `peft` + `trl`: config-driven, reproducible, moderate speedup.

**`torchtune` is deprecated.** PyTorch Foundation's native finetuning library stopped active development in July 2025. If a tutorial or old notebook recommends `torchtune`, substitute the live stack (`trl` + `peft`).

**Model licenses vary.** Llama, Gemma, Mistral, and many other "open" models require accepting a license on the Hub before `transformers` can download them. The library will raise `GatedRepoError`; the fix is `huggingface-cli login` + accepting the terms on the model's web page.

**`accelerate` vs raw `torch.distributed`.** `accelerate` is a convenience layer, not a replacement. For project 12 the recommended path is to write the DDP loop raw first, then rewrite with FSDP2 (`torch.distributed.fsdp.fully_shard`), then once more with `accelerate` — so you can see what the abstraction is hiding.

## See also

- `docs/tools-overview.md` — the full tooling landscape across all 15 projects.
- `.claude/study/common-nn-tools/report.md` — the ~110-tool survey underlying both docs.
- [`huggingface.co/docs`](https://huggingface.co/docs) — official library docs.
