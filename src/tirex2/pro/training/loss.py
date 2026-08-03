"""Pinball (quantile) loss for training TiRex-2.

The loss is applied at every observed output time step, matching Equation (3) in
the TiRex-2 paper:

    L = 1/(|Q| |T_obs|) Σ_{t∈T_obs} Σ_{q∈Q} [q (x_t - x̂_t^q)_+ + (1-q)(x̂_t^q - x_t)_+]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PinballLoss(nn.Module):
    """Pinball loss for quantile forecasting.

    Parameters
    ----------
    quantiles : torch.Tensor
        Quantile levels τ_1, ..., τ_Q. The median (0.5) is expected to be
        present for median-based evaluation.
    reduction : {"mean", "sum"}
        How to reduce the loss over observed positions.
    """

    def __init__(self, quantiles: torch.Tensor | list[float], reduction: str = "mean") -> None:
        super().__init__()
        if not isinstance(quantiles, torch.Tensor):
            quantiles = torch.tensor(quantiles, dtype=torch.float32)
        self.register_buffer("quantiles", quantiles.view(1, -1, 1))
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the pinball loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted quantiles, shape ``[..., Q, T]``.
        target : torch.Tensor
            Ground-truth values, shape ``[..., T]``. The quantile dimension is
            broadcast against ``pred``.
        mask : torch.Tensor, optional
            Boolean mask of observed positions, shape ``[..., T]``. If provided,
            only ``True`` positions contribute to the loss. NaN positions in
            ``target`` are automatically ignored even without a mask.

        Returns
        -------
        torch.Tensor
            Scalar loss (mean or sum over observed positions).
        """
        if pred.ndim < 2:
            raise ValueError(f"pred must have at least 2 dims [..., Q, T], got shape {pred.shape}")
        if target.shape != pred.shape[:-2] + pred.shape[-1:]:
            raise ValueError(f"target shape {tuple(target.shape)} incompatible with pred shape {tuple(pred.shape)}")

        # Expand target to [..., 1, T] to broadcast with quantile dimension.
        target = target.unsqueeze(-2)

        # Build mask from provided mask or from non-NaN target values.
        if mask is None:
            mask = ~torch.isnan(target)
        else:
            mask = mask.unsqueeze(-2) if mask.ndim < pred.ndim else mask
            mask = mask & ~torch.isnan(target)

        # Zero errors at masked positions before computing loss so NaNs do not
        # propagate through torch.where.
        safe_target = torch.where(mask, target, pred)
        errors = safe_target - pred  # negative when pred > target

        # q * (target - pred) for under-predictions + (1-q) * (pred - target) for over-predictions.
        loss = torch.where(
            errors >= 0,
            self.quantiles * errors,
            (self.quantiles - 1.0) * errors,
        )

        masked_loss = loss * mask.to(loss.dtype)
        total = masked_loss.sum()

        if self.reduction == "mean":
            count = mask.sum()
            # Avoid division by zero on an empty batch.
            return total / (count.clamp_min(1.0))
        return total


class MaskedMAELoss(nn.Module):
    """Masked mean absolute error on the predicted median.

    Useful as an auxiliary loss or for validation when quantiles are expensive.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute MAE on the median prediction.

        Parameters
        ----------
        pred : torch.Tensor
            Shape ``[..., Q, T]``.
        target : torch.Tensor
            Shape ``[..., T]``.
        mask : torch.Tensor, optional
            Boolean mask of observed positions.
        """
        # Locate median quantile index.
        q = pred.shape[-2]
        median_idx = q // 2
        median_pred = pred[..., median_idx, :]

        if mask is None:
            mask = ~torch.isnan(target)
        diff = (median_pred - target).abs()
        masked = diff * mask.to(diff.dtype)
        return masked.sum() / mask.sum().clamp_min(1.0)
