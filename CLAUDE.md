# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This is the TiRex-2 project root. The source package lives under `src/tirex2/`, tests under `test/`, and examples under `examples/`. The repository was previously nested inside a wrapper directory; it has been flattened so the project root is also the Git root.

## Development environment

We use [uv](https://docs.astral.sh/uv/) for package management and a `Makefile` for common tasks. Do not use Pixi; it has been removed.

```bash
make install         # auto-detect GPU and install the project editable + extras
make install-cuda    # explicitly install with torch cu128 + local CUDA 12.8 toolkit if needed
make install-cpu     # explicitly install with CPU-only torch
```

### DGX Spark / CUDA 13.0 driver note

The upstream xlstm/FlashRNN CUDA extensions are tested against CUDA 12.8, so this project uses `torch==2.9.1+cu128` on NVIDIA hardware. On a machine with a CUDA 13.0 driver (e.g. DGX Spark), `make install-cuda` runs `scripts/setup_hardware.py` to download a local CUDA 12.8 toolkit and exports `CUDA_HOME` and `TORCH_CUDA_ARCH_LIST`. The CUDA 13.0 driver is backward compatible with the resulting CUDA 12.8 binaries.

## Common commands

```bash
make test                           # run the full pytest suite
make test-single FILE=...          # run a single test file
make minimal                       # sine-wave smoke example; expects ./model
make comparison                    # covariate demo; writes figures to output/
make notebook                      # launch Jupyter Lab with examples/getting_started.ipynb
make fevbench ARGS="..."           # fev-bench runner
make gifteval ARGS="..."           # GiftEval runner
make lint                          # ruff check
make format                        # ruff format
make clean                         # remove .venv, caches, outputs
```

For ad-hoc commands, use `uv run`:

```bash
uv run pytest test/test_forecast_model.py
uv run python examples/sine_wave.py
PYTHONPATH=examples/gifteval:$PYTHONPATH uv run python examples/gifteval/run_gifteval.py ...
```

## Code architecture

The package is `tirex2` under `src/tirex2/`:

- `base.py` — `load_model(...)`. Resolves a local checkpoint directory or a Hugging Face repo id (`org/repo` or `hf://org/repo`), downloads weights/config, constructs `TiRex2`, and wraps it in `ForecastModel`.
- `model/tirex2.py` — `TiRex2` `nn.Module`. Builds a multivariate block stack from `model-config.yaml`, runs tokenization/scaling, the stack, and an output patch head, then reverses the transforms. Also exposes `predict(...)` with test-time augmentation flags (`tta_sign_flip`, `tta_diff`).
- `model/types.py` — `TimeseriesType`, the container for target + past/future covariates.
- `model/component/` — building blocks: patch tokenizer, scaler, residual blocks, multivariate mixing blocks (time + variate mixers), mLSTM/xLSTM blocks, attention, layer norms, and the postprocessor.
- `api_adapter/forecast.py` — `ForecastModel`, the high-level wrapper. Batches series, handles adaptive OOM recovery (halves batch size on CUDA/MPS OOM), and converts outputs to `torch`, `numpy`, `gluonts`, or `fev` formats.
- `api_adapter/gluon.py` and `api_adapter/standard_adapter.py` — output format helpers for GluonTS and the default adapter.
- `demo.py` / `plotting.py` — demo data generators and plotting helpers for examples.
- `pro/training/` — training utilities: `PinballLoss`, `TiRexDataset`, `SyntheticCouplingPipeline`, optimizers/schedulers, checkpoint I/O, and LoRA helpers.
- `pro/finetuning/trainer.py` — `FineTuner` with `full`, `head-only`, `blocks`, and `lora` strategies.
- `pro/` — additional Pro skeletons: `streaming`, `classification`, `regression`, and `hardware`.

## Checkpoints

A local checkpoint must contain `model-config.yaml` and `model.ckpt`. The directory is conventionally referenced as `./model`, which is gitignored, as are `output/` and `*.csv` benchmark outputs. Fine-tuned checkpoints are saved in the same layout and can be reloaded with `load_model`.

## Tests

Tests live in `test/`:

- `test_tirex2_instantiation.py`
- `test_forecast_model.py`
- `test_api_adapter.py`
- `test_fev_adapter.py`
- `test_postprocessor_instantiation.py`
- `test_references.py`
- `test_training.py`

Run the full suite with `make test` before handing off changes.

## Packaging sanity

```bash
uv build
uv pip wheel . --no-deps -w /tmp/tirex2-wheel-test
```

## Important references

For detailed usage (FEV-bench, GiftEval, Docker, output formats), read `AGENT.md` and `README.md`. Do not commit generated artifacts such as `.pixi/`, `.venv/`, `__pycache__/`, `output/`, `model/`, `*.csv`, or `*.egg-info`.
