#!/usr/bin/env python3
"""Command-line fine-tuning script for TiRex-2.

Reads a YAML configuration file, loads a TiRex-2 checkpoint, optionally generates
or loads training data, and fine-tunes with ``tirex2.pro.finetuning.FineTuner``.

Example config
--------------

```yaml
checkpoint: NX-AI/TiRex-2
device: cpu

synthetic_data:
  num_samples: 256
  num_variates: 4
  window_length: 512

strategy: head-only
strategy_kwargs: {}

epochs: 5
batch_size: 4
learning_rate: 0.0001
weight_decay: 0.01
grad_clip: 1.0
warmup_ratio: 0.1
early_stopping_patience: 2

output_dir: ./checkpoints/finetuned
log_interval: 5
```
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from tirex2 import TimeseriesType, load_model
from tirex2.pro.finetuning import FineTuner
from tirex2.pro.training import SyntheticCouplingPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tirex2.train")


def _load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_synthetic_data(cfg: dict[str, Any]) -> list[TimeseriesType]:
    """Generate multivariate training samples from random univariate series."""
    synth_cfg = cfg.get("synthetic_data", {})
    num_samples = synth_cfg.get("num_samples", 256)
    num_variates = synth_cfg.get("num_variates", 4)
    window_length = synth_cfg.get("window_length", 512)
    mechanisms = synth_cfg.get("mechanisms", None)

    # Build a small pool of random univariate series.
    pool_size = max(8, num_variates * 2)
    pool = [torch.randn(window_length) for _ in range(pool_size)]

    pipeline = SyntheticCouplingPipeline(
        mechanisms=mechanisms,
        num_variates=num_variates,
        window_length=window_length,
    )
    samples = pipeline.generate(pool, n_samples=num_samples)
    return [
        TimeseriesType(
            target=s["target"],
            past_covariates=s["past_covariates"],
            future_covariates=s["future_covariates"],
        )
        for s in samples
    ]


def _load_timeseries_files(path: str | Path) -> list[TimeseriesType]:
    """Load ``TimeseriesType`` objects from pickled ``.pt`` files in a directory."""
    directory = Path(path)
    if not directory.is_dir():
        raise ValueError(f"Data path must be a directory of .pt files, got {directory}")
    files = sorted(directory.glob("*.pt"))
    if not files:
        raise ValueError(f"No .pt files found in {directory}")
    series = []
    for file in files:
        obj = torch.load(file, weights_only=False)
        if isinstance(obj, TimeseriesType):
            series.append(obj)
        elif isinstance(obj, list):
            series.extend(obj)
        else:
            raise ValueError(f"Unexpected object type in {file}: {type(obj)}")
    return series


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune TiRex-2")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    parser.add_argument("--ckpt", help="Checkpoint path or HF repo id (overrides config)")
    parser.add_argument("--data", help="Directory of .pt TimeseriesType files (overrides synthetic)")
    parser.add_argument("--out", help="Output directory (overrides config)")
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)

    checkpoint = args.ckpt or cfg.get("checkpoint", "NX-AI/TiRex-2")
    device = cfg.get("device", "cpu")
    output_dir = args.out or cfg.get("output_dir", "./checkpoints/finetuned")

    logger.info("Loading checkpoint %s on %s", checkpoint, device)
    model = load_model(checkpoint, device=device)

    if args.data:
        logger.info("Loading training data from %s", args.data)
        train_data = _load_timeseries_files(args.data)
    else:
        logger.info("Generating synthetic training data")
        train_data = _build_synthetic_data(cfg)

    # Optionally split off validation data.
    val_fraction = cfg.get("val_fraction", 0.0)
    val_data = None
    if val_fraction > 0 and not args.data:
        split_idx = int(len(train_data) * (1 - val_fraction))
        val_data = train_data[split_idx:]
        train_data = train_data[:split_idx]
        logger.info("Split into %d train / %d validation samples", len(train_data), len(val_data))

    strategy = cfg.get("strategy", "head-only")
    strategy_kwargs = cfg.get("strategy_kwargs", {}) or {}

    fine_tuner = FineTuner(model, strategy=strategy, strategy_kwargs=strategy_kwargs)

    fit_kwargs = {
        "train_data": train_data,
        "val_data": val_data,
        "epochs": cfg.get("epochs", 10),
        "batch_size": cfg.get("batch_size", 4),
        "learning_rate": cfg.get("learning_rate", 1e-4),
        "weight_decay": cfg.get("weight_decay", 0.01),
        "grad_clip": cfg.get("grad_clip", 1.0),
        "warmup_ratio": cfg.get("warmup_ratio", 0.1),
        "early_stopping_patience": cfg.get("early_stopping_patience", None),
        "output_dir": output_dir,
        "log_interval": cfg.get("log_interval", 10),
        "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", 1),
        "context_length": cfg.get("context_length", None),
        "prediction_length": cfg.get("prediction_length", None),
    }
    # Remove None values so the FineTuner uses its defaults.
    fit_kwargs = {k: v for k, v in fit_kwargs.items() if v is not None}

    fine_tuner.fit(**fit_kwargs)
    fine_tuner.save(Path(output_dir) / "final")
    logger.info("Fine-tuned model saved to %s", Path(output_dir) / "final")
    return 0


if __name__ == "__main__":
    sys.exit(main())
