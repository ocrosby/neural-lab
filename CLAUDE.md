# neural-lab

Progressive neural network projects from a single perceptron to a trained LLM. See `README.md` for the full 15-project ladder, learning resources, and compute reality.

## Project shape

- Python 3.12+, `uv`-managed. Run everything through `uv run ...` — never `python` or `pip` directly.
- Workspace layout: `projects/NN-slug/` is one standalone project; `shared/` holds utilities every project uses (`seeds`, dataset loaders, plotting).
- Each project has `README.md`, source, and `tests/`. `data/`, `checkpoints/`, `runs/` are gitignored — never commit artifacts.

## Hard rules for this repo

- **No cross-project imports.** `projects/05-cnn-vision/` does not import from `projects/03-micrograd/`. Shared code goes in `shared/`. Each project must stand alone so a reader can study it in isolation.
- **Call `shared.seeds.set_seed(42)` first in every entry point** (demo, training loop, test fixture that exercises randomness). Deterministic runs separate "fixed a bug" from "got lucky."
- **Tests are black-box** — assert on outcomes (final accuracy, XOR solved, gradient-check tolerance), never on internal weights, intermediate activations, or private attributes.
- **Respect the ladder.** Projects are numbered because each assumes the previous. When adding to project N, don't preemptively pull in machinery from project N+k.
- **ML notation beats PEP 8.** `X`, `W`, `A`, `Z` for matrices; `y`, `b`, `x_i` for vectors/scalars. `ruff` is already configured to ignore N803/N806 — don't "fix" these names.

## Pedagogical constraints

This is a learning progression, not a production codebase. When implementing or reviewing:

- Prefer transparency over cleverness. A visible 20-line backprop beats a 5-line one-liner that hides the math.
- Keep NumPy detours (02, 03, the NumPy half of 06, and 07a once built) framework-free. The whole point is that the student can see every multiply.
- When a project pairs a from-scratch impl with a framework impl (`02-mlp-from-scratch` ↔ `04-pytorch-mlp`, `06-rnn-lm` NumPy ↔ PyTorch halves), the behavioral contract and tests should align so the comparison is apples-to-apples.

## Common commands

```bash
uv sync                              # install/update deps
uv run pytest                        # collects everything via configured testpaths
uv run pytest projects/06-rnn-lm     # scope to one project (pass the path, don't cd in)
uv run ruff check .                  # lint
uv run ruff format .                 # format
uv run mypy .                        # type check (strict)
```
