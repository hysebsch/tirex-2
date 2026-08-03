"""Low-rank adaptation (LoRA) helpers for fine-tuning TiRex-2.

LoRA adapters are injected into the variate-mixer attention projection layers
(WQ, WK, WV, WO). During fine-tuning only the low-rank adapter parameters are
updated, while the pre-trained backbone remains frozen.
"""

from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap a ``nn.Linear`` layer with trainable low-rank adapters.

    The effective weight during forward pass is ``W + alpha/r * B @ A``.
    The original ``W`` is frozen; only ``A`` and ``B`` are trained.

    Parameters
    ----------
    base_layer : nn.Linear
        The original linear projection to adapt.
    rank : int
        Rank of the adapter.
    alpha : float
        Scaling factor. Typical values: ``alpha == rank``.
    """

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Freeze base weights.
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Low-rank matrices.
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        adapter = x @ self.lora_A
        adapter = adapter @ self.lora_B.T
        return base + self.scaling * adapter


def _is_attention_projection(name: str) -> bool:
    """Heuristic: attention projection layers in the codebase are named WQ/WK/WV/WO."""
    return bool(re.search(r"\.(WQ|WK|WV|WO)$", name))


def inject_lora(
    model: nn.Module,
    target_modules: str | list[str] = "attention",
    rank: int = 8,
    alpha: float = 8.0,
) -> nn.Module:
    """Inject LoRA adapters into selected linear layers and freeze the backbone.

    Parameters
    ----------
    model : nn.Module
        The TiRex-2 model to adapt.
    target_modules : str | list[str]
        Which modules to target. ``"attention"`` matches attention projection
        layers (WQ/WK/WV/WO). A list of strings is treated as regex patterns
        matched against full parameter names.
    rank : int
        LoRA rank.
    alpha : float
        LoRA scaling.

    Returns
    -------
    nn.Module
        The modified model (modified in place).
    """
    if isinstance(target_modules, str) and target_modules == "attention":
        patterns = [r"\.variate_mixer.*\.(WQ|WK|WV|WO)$"]
    elif isinstance(target_modules, str):
        patterns = [target_modules]
    else:
        patterns = list(target_modules)

    compiled = [re.compile(p) for p in patterns]

    def _should_adapt(name: str) -> bool:
        return any(p.search(name) for p in compiled)

    # Recursively replace matching Linear modules with LoRALinear wrappers.
    _replace_linear_layers(model, _should_adapt, rank, alpha)

    # Freeze everything except LoRA parameters.
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model


def _replace_linear_layers(
    module: nn.Module,
    should_adapt: Any,
    rank: int,
    alpha: float,
    prefix: str = "",
) -> None:
    """Recursively replace linear layers in ``module`` with LoRALinear wrappers."""
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and should_adapt(full_name):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
        else:
            _replace_linear_layers(child, should_adapt, rank, alpha, prefix=full_name)


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA adapters into the base linear weights for inference.

    This permanently adds the adapter updates to ``base_layer.weight`` and removes
    the adapter parameters.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            child_name = name.rsplit(".", 1)[1] if "." in name else name
            parent = model.get_submodule(parent_name) if parent_name else model
            # Compute merged weight.
            merged_weight = module.base_layer.weight.data + module.scaling * (module.lora_B @ module.lora_A.T)
            module.base_layer.weight.data = merged_weight
            # Replace wrapper with the unfrozen base layer.
            module.base_layer.weight.requires_grad = True
            if module.base_layer.bias is not None:
                module.base_layer.bias.requires_grad = True
            setattr(parent, child_name, module.base_layer)
    return model
