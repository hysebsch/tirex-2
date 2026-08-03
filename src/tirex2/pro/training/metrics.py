"""Validation metrics for TiRex-2 quantile forecasts.

This module implements standard time-series forecast scores that can be used to
monitor fine-tuning progress and benchmark the model on held-out data.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MASE(nn.Module):
    """Mean Absolute Scaled Error.

    MASE divides the mean absolute forecast error by the in-sample one-step naive
    forecast error computed on the context window. This makes it scale-free and
    comparable across datasets.

    Parameters
    ----------
    eps : float
        Small constant to avoid division by zero.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute MASE.

        Parameters
        ----------
        pred : torch.Tensor
            Predictions, shape ``[..., T]`` (typically the predicted median).
        target : torch.Tensor
            Ground truth, shape ``[..., T]``.
        context : torch.Tensor, optional
            Historical context used to estimate the naive scale, shape ``[..., T_c]``.
            If omitted, ``target`` is used, which gives a slightly different but
            still useful scale estimate.
        mask : torch.Tensor, optional
            Boolean mask of observed positions in ``target``.

        Returns
        -------
        torch.Tensor
            Scalar MASE.
        """
        if mask is None:
            mask = ~torch.isnan(target)

        abs_err = (pred - target).abs() * mask.to(pred.dtype)
        mae = abs_err.sum() / mask.sum().clamp_min(self.eps)

        scale_window = target if context is None else context
        scale_mask = ~torch.isnan(scale_window)
        # Naive one-step absolute differences.
        diffs = (scale_window[..., 1:] - scale_window[..., :-1]).abs()
        scale_mask = scale_mask[..., 1:] & scale_mask[..., :-1]
        naive_mae = (diffs * scale_mask.to(diffs.dtype)).sum() / scale_mask.sum().clamp_min(self.eps)

        return mae / naive_mae.clamp_min(self.eps)


class QuantileCRPS(nn.Module):
    """Continuous Ranked Probability Score for quantile forecasts.

    Approximates the CRPS by integrating the pinball loss over the predicted
    quantile levels:

        CRPS(y, F) = 2 ∫_0^1 ρ_τ(y, q_τ) dτ

    where ρ_τ is the pinball loss. For discrete quantiles we use a trapezoidal
    rule with the predicted quantile levels.
    """

    def __init__(self, quantiles: torch.Tensor | list[float], eps: float = 1e-8) -> None:
        super().__init__()
        if not isinstance(quantiles, torch.Tensor):
            quantiles = torch.tensor(quantiles, dtype=torch.float32)
        self.register_buffer("quantiles", quantiles.view(-1))
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute CRPS.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted quantiles, shape ``[..., Q, T]``.
        target : torch.Tensor
            Ground truth, shape ``[..., T]``.
        mask : torch.Tensor, optional
            Boolean mask of observed positions, shape ``[..., T]``.

        Returns
        -------
        torch.Tensor
            Scalar CRPS estimate.
        """
        if pred.ndim < 2:
            raise ValueError(f"pred must have at least 2 dims [..., Q, T], got shape {pred.shape}")
        if target.shape != pred.shape[:-2] + pred.shape[-1:]:
            raise ValueError(
                f"target shape {tuple(target.shape)} incompatible with pred shape {tuple(pred.shape)}"
            )

        target = target.unsqueeze(-2)
        if mask is None:
            mask = ~torch.isnan(target)
        else:
            mask = mask.unsqueeze(-2) if mask.ndim < pred.ndim else mask
            mask = mask & ~torch.isnan(target)

        safe_target = torch.where(mask, target, pred)
        errors = safe_target - pred
        loss = torch.where(
            errors >= 0,
            self.quantiles.view(1, -1, 1) * errors,
            (self.quantiles.view(1, -1, 1) - 1.0) * errors,
        )

        # Trapezoidal weights for the quantile axis. Boundaries are 0, quantiles, 1.
        q = self.quantiles
        weights = torch.empty_like(q)
        weights[0] = (q[0] - 0.0 + (q[1] - q[0]) / 2.0) if q.numel() > 1 else 1.0
        for i in range(1, q.numel() - 1):
            weights[i] = (q[i] - q[i - 1] + q[i + 1] - q[i]) / 2.0
        if q.numel() > 1:
            weights[-1] = (1.0 - q[-1] + (q[-1] - q[-2]) / 2.0)
        weights = weights.view(1, -1, 1)

        masked_loss = loss * weights * mask.to(loss.dtype)
        total = 2.0 * masked_loss.sum()
        count = mask.sum()
        return total / count.clamp_min(self.eps)


class MetricsTracker:
    """Track running averages of multiple validation metrics."""

    def __init__(self, quantiles: torch.Tensor | list[float] | None = None) -> None:
        self.mase = MASE()
        self.crps = QuantileCRPS(quantiles) if quantiles is not None else None
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Update running metrics with a single batch and return batch values."""
        batch_values: dict[str, float] = {}
        if mask is None:
            mask = ~torch.isnan(target)

        with torch.no_grad():
            mase_value = self.mase(pred, target, context=context, mask=mask).item()
        self._sums["mase"] = self._sums.get("mase", 0.0) + mase_value
        self._counts["mase"] = self._counts.get("mase", 0) + 1
        batch_values["mase"] = mase_value

        return batch_values

    def update_crps(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> float | None:
        """Update the running CRPS with a quantile prediction batch."""
        if self.crps is None:
            return None
        with torch.no_grad():
            crps_value = self.crps(pred, target, mask=mask).item()
        self._sums["crps"] = self._sums.get("crps", 0.0) + crps_value
        self._counts["crps"] = self._counts.get("crps", 0) + 1
        return crps_value

    def average(self) -> dict[str, float]:
        """Return the average of each tracked metric."""
        return {name: self._sums[name] / max(1, self._counts[name]) for name in self._sums}

    def reset(self) -> None:
        """Reset running sums."""
        self._sums.clear()
        self._counts.clear()
