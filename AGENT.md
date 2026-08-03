# Agent instructions for this repo

This repo contains TiRex-2 inference code. Packaging, dependencies, and tasks are configured in `pyproject.toml` and `Makefile`. Source code uses a `src/` layout under `src/tirex2`.

## Package management

We use [uv](https://docs.astral.sh/uv/) for development and [Makefile](Makefile) targets for common tasks.

```bash
make install        # auto-detect GPU and install editable project + extras
make install-cuda   # install with torch cu128 + local CUDA 12.8 toolkit if needed
make install-cpu    # install with CPU-only torch
```

## Environments

- Default install uses CUDA 12.8 (`cu128`) on NVIDIA GPUs. This matches the CUDA version xlstm/FlashRNN are tested against.
- On a DGX Spark / CUDA 13.0 driver, `make install-cuda` runs `scripts/setup_hardware.py` to download a local CUDA 12.8 toolkit and exports `CUDA_HOME`/`TORCH_CUDA_ARCH_LIST`. The CUDA 13.0 driver is backward compatible with the resulting CUDA 12.8 binaries.
- CPU-only install uses `torch==2.9.1+cpu`.

## Common commands

```bash
make test                         # runs pytest in the project venv
make test-single FILE=...         # run a single test file
make train ARGS="..."             # run the training/fine-tuning CLI
make minimal                      # sine-wave smoke example, uses ./model
make comparison                   # covariate demo, writes figures to output/
make fevbench ARGS="..."          # fev-bench runner
make gifteval ARGS="..."          # GiftEval runner
make notebook                     # launch Jupyter Lab with examples/getting_started.ipynb
make lint                         # ruff check
make format                       # ruff format
make clean                        # remove venv, caches, outputs
```

`model` is expected to be a local checkpoint directory or symlink containing `model-config.yaml` and `model.ckpt`. It is gitignored, as are `output/` and `*.csv` benchmark outputs.

## Forecasting from Python

Minimal direct API usage:

```python
import torch
from tirex2 import TimeseriesType, load_model

model = load_model("./model", device="cpu")  # use device="cuda" in a CUDA env if needed

target = torch.randn(1, 512)  # shape: [num_target_variates, context_length]
ts = TimeseriesType(target=target, past_covariates=None, future_covariates=None)

# returns list of forecasts; each forecast has shape [num_target_variates, num_quantiles, prediction_length]
forecast = model.forecast([ts], prediction_length=64, output_type="torch", batch_size=512)[0]
```

For future-known covariates, pass `future_covariates` with shape `[num_covariates, context_length + prediction_length]`. Past-only covariates use `past_covariates` with shape `[num_covariates, context_length]`.

Supported output types: `"torch"`, `"numpy"`, `"gluonts"`, and `"fev"` where the latter two require the optional dependencies/envs. Extra kwargs passed to `forecast(...)` are forwarded to `TiRex2.predict`, e.g. `tta_diff=False` or `tta_sign_flip=True`.

## Local examples

Sine-wave smoke test:

```bash
make minimal
# or explicitly:
uv run python examples/sine_wave.py
```

Future-known covariate demo:

```bash
make comparison
# custom checkpoint/output/scenarios:
uv run python examples/covariate_forecasts.py \
  --ckpt ./model \
  --device cpu \
  --scenarios holidays nonstationary \
  --out output
```

The demo writes PNGs under `output/` by default.

## FEV-bench

Script: `examples/fevbench/run_fevbench.py`

Tasks are loaded from the paths configured in the YAML, typically the HuggingFace Hub (`autogluon/fev_datasets`). The model is always loaded from `NX-AI/TiRex-2-fevbench`.

To run offline against a local dataset snapshot, point HuggingFace's datasets cache at it instead of passing a path. The snapshot must use the standard cache layout (`<cache>/autogluon___fev_datasets/<config>/...`):

```bash
export HF_DATASETS_CACHE=/path/to/fev_store
export HF_HUB_OFFLINE=1
```

Run a quick/small benchmark first:

```bash
make fevbench ARGS="\
  --tasks examples/fevbench/tasks-mini.yaml \
  --out output/fevbench-mini.csv \
  --device cuda:0 \
  --batch_size 128"
```

Full configured task list:

```bash
make fevbench ARGS="\
  --tasks examples/fevbench/tasks.yaml \
  --out output/fevbench.csv \
  --device cuda:0 \
  --batch_size 512"
```

Useful options:

- `--max_tasks N` to run only the first N tasks from the YAML.
- `--as_univariate` to ignore covariates and forecast each target independently.
- `--model_name NAME` to set the model name in the output CSV.
- The script retries CUDA OOM by halving batch size; reduce `--batch_size` if needed.

## GiftEval

Script: `examples/gifteval/run_gifteval.py`

Download the GiftEval data once:

```bash
uv run huggingface-cli download Salesforce/GiftEval \
  --repo-type=dataset \
  --local-dir /path/to/gifteval_storage
```

Run the benchmark:

```bash
make gifteval ARGS="/path/to/gifteval_storage pretrained \
  --out output/gifteval.csv \
  --device cuda"
```

Use `zero-shot` instead of `pretrained` to load `NX-AI/TiRex-2-gifteval-zs`.

The script sets `GIFT_EVAL=/path/to/gifteval_storage` before importing the local GiftEval helpers. `examples/gifteval` is added to `PYTHONPATH` by the Makefile targets so `gift_eval_utils` imports correctly.

Interactive notebook:

```bash
make notebook
# open examples/gifteval/gifteval.ipynb
```

## Tests and validation

Before handing off changes, run at least:

```bash
make test
```

For packaging sanity:

```bash
uv build
```

Do not commit generated files/directories such as `.pixi/`, `.venv/`, `__pycache__/`, `output/`, `model`, `*.csv`, or `*.egg-info`.

## Training and fine-tuning

Training code lives under `src/tirex2/pro/training/` and `src/tirex2/pro/finetuning/`:

```python
from tirex2.pro.finetuning import FineTuner
from tirex2.pro.training import TiRexDataset, SyntheticCouplingPipeline

fine_tuner = FineTuner(model, strategy="head-only")
fine_tuner.fit(train_data, epochs=10, batch_size=8, learning_rate=1e-4, output_dir="./checkpoints")
fine_tuner.save("./fine_tuned_model")
```

Strategies: `full`, `head-only`, `blocks`, `lora`.

Command-line script:

```bash
python scripts/train.py --config configs/finetune.yaml --ckpt NX-AI/TiRex-2 --out ./checkpoints
```

## TiRex Pro skeletons

A `src/tirex2/pro/` subpackage contains modules for Pro capabilities:
`training`, `finetuning`, `streaming`, `classification`, `regression`, and `hardware`.
The `training`/`finetuning` modules are functional; the others are placeholder stubs for
upcoming Pro feature work.
