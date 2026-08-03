"""Training utilities: optimizers, schedulers, and checkpoint I/O for TiRex-2."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim import AdamW

CONFIG_FILENAME = "model-config.yaml"
CKPT_FILENAME = "model.ckpt"


def build_optimizer(
    parameters: list[torch.nn.Parameter],
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.999),
) -> AdamW:
    """Build an AdamW optimizer."""
    return AdamW(parameters, lr=learning_rate, weight_decay=weight_decay, betas=betas)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.1,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine schedule with linear warmup.

    Parameters
    ----------
    optimizer
        The optimizer to schedule.
    num_training_steps
        Total number of optimization steps.
    warmup_ratio
        Fraction of ``num_training_steps`` spent in linear warmup.
    min_lr_ratio
        Final learning rate as a fraction of the peak.
    """
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError(f"warmup_ratio must be in [0, 1), got {warmup_ratio}")
    warmup_steps = int(num_training_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, num_training_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_trainable_parameters(
    model: torch.nn.Module,
    strategy: str,
    strategy_kwargs: dict[str, Any] | None,
) -> list[torch.nn.Parameter]:
    """Return the parameters that should be trained for the chosen strategy.

    Strategies:
    - ``full``: all parameters.
    - ``head-only``: input/output patch embeddings and final normalization.
    - ``blocks``: freeze selected blocks; train the rest. Use
      ``strategy_kwargs={"blocks_to_train": [10, 11]}``.
    - ``lora``: parameters registered by LoRA adapters (handled separately).
    """
    strategy_kwargs = strategy_kwargs or {}

    if strategy == "full":
        return list(model.parameters())

    if strategy == "head-only":
        trainable: list[torch.nn.Parameter] = []
        if hasattr(model, "input_patch_embedding"):
            trainable.extend(model.input_patch_embedding.parameters())
        if hasattr(model, "output_patch_embedding"):
            trainable.extend(model.output_patch_embedding.parameters())
        if hasattr(model, "stack_out_norm") and not isinstance(model.stack_out_norm, torch.nn.Identity):
            trainable.extend(model.stack_out_norm.parameters())
        return trainable

    if strategy == "blocks":
        blocks_to_train = set(strategy_kwargs.get("blocks_to_train", []))
        trainable = []
        for name, param in model.named_parameters():
            # Try to identify block index from parameter names like "stack.3.*".
            in_selected_block = any(f"stack.{idx}." in name for idx in blocks_to_train)
            if in_selected_block:
                trainable.append(param)
        if not trainable:
            raise ValueError(f"No parameters matched blocks_to_train={blocks_to_train}")
        return trainable

    if strategy == "lora":
        # LoRA adapters register their parameters with requires_grad=True; backbone is frozen.
        return [p for p in model.parameters() if p.requires_grad]

    raise ValueError(f"Unknown fine-tuning strategy {strategy!r}")


def freeze_backbone(model: torch.nn.Module, strategy: str, strategy_kwargs: dict[str, Any] | None) -> None:
    """Freeze all parameters except those that should be trained."""
    for p in model.parameters():
        p.requires_grad = False
    for p in get_trainable_parameters(model, strategy, strategy_kwargs):
        p.requires_grad = True


META_FILENAME = "checkpoint_meta.yaml"
TRAINING_STATE_FILENAME = "training_state.pt"


def save_checkpoint(
    model: torch.nn.Module,
    output_dir: str | Path,
    extra_config: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    epoch: int | None = None,
    metrics: dict[str, float] | None = None,
) -> None:
    """Save a TiRex-2 checkpoint compatible with ``tirex2.load_model``.

    The directory will contain ``model-config.yaml`` and ``model.ckpt``.
    Optional training state (optimizer, scheduler, epoch, metrics) is saved
    to ``training_state.pt`` so fine-tuning can resume. Metadata is saved to
    ``checkpoint_meta.yaml``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config: dict[str, Any] = dict(getattr(model, "_init_kwargs", {}))
    # Ensure device is set to the current model device for reload.
    config["device"] = next(model.parameters()).device.type

    config_file = output_dir / CONFIG_FILENAME
    with config_file.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    state_dict = model.state_dict()
    ckpt: dict[str, Any] = {
        "state_dict": state_dict,
        "quantiles": state_dict.get("quantiles", torch.tensor([])).tolist(),
    }
    torch.save(ckpt, output_dir / CKPT_FILENAME)

    extra_config = dict(extra_config or {})
    if epoch is not None:
        extra_config["epoch"] = epoch
    if metrics is not None:
        extra_config["metrics"] = metrics
    if extra_config:
        meta_file = output_dir / META_FILENAME
        with meta_file.open("w") as f:
            yaml.safe_dump(extra_config, f, sort_keys=False)

    if optimizer is not None or scheduler is not None:
        training_state: dict[str, Any] = {}
        if optimizer is not None:
            training_state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            training_state["scheduler_state_dict"] = scheduler.state_dict()
        if epoch is not None:
            training_state["epoch"] = epoch
        torch.save(training_state, output_dir / TRAINING_STATE_FILENAME)


def load_training_state(output_dir: str | Path) -> dict[str, Any]:
    """Load a previously saved training state dict."""
    state_path = Path(output_dir) / TRAINING_STATE_FILENAME
    if not state_path.exists():
        return {}
    return torch.load(state_path, map_location="cpu", weights_only=False)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Return the number of (trainable) parameters."""
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))
