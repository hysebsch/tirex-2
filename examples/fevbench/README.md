# FEV-benchmark

Evaluate TiRex2 on the [FEV-Benchmark](https://huggingface.co/spaces/autogluon/fev-bench)
benchmark.

## Data

Data can be downloaded beforehand using:

```bash
uv run huggingface-cli download autogluon/fev_datasets --repo-type=dataset --local-dir </path/to/fevbench/store>
```

## Run

The script always loads `NX-AI/TiRex-2-fevbench` from Hugging Face:

```bash
make fevbench ARGS="[/path/to/fevbench_storage] [--tasks examples/fevbench/tasks.yaml]"
```

Or directly with uv:

```bash
uv run python examples/fevbench/run_fevbench.py \
  [/path/to/fevbench_storage] [--tasks examples/fevbench/tasks.yaml]
```

Note: if `/path/to/fevbench_storage` is not given, then the dataset is downloaded at runtime and stored in $HOME/.cache.
