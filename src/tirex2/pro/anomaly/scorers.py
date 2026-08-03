"""Scoring functions for forecast-deviation anomaly detection.

Each scorer receives a one-step-ahead quantile forecast and the observed value
and returns a per-variate anomaly score. Higher scores indicate more anomalous
observations.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

Scorer = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _find_quantile_index(quantiles: torch.Tensor, level: float) -> int:
    """Return the index of the quantile level closest to ``level``."""
    return int((quantiles - level).abs().argmin().item())


def iqr_deviation_scorer(quantiles: torch.Tensor) -> Scorer:
    """Return a scorer based on normalized deviation from the median.

    The score is ``|actual - median| / max(iqr, eps)`` where ``iqr`` is the
    inter-quantile range ``q_high - q_low`` (default 0.9 - 0.1).
    """
    median_idx = _find_quantile_index(quantiles, 0.5)
    low_idx = _find_quantile_index(quantiles, 0.1)
    high_idx = _find_quantile_index(quantiles, 0.9)

    def score(forecast: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        # forecast: [V, Q], actual: [V]
        median = forecast[:, median_idx]
        iqr = (forecast[:, high_idx] - forecast[:, low_idx]).abs()
        eps = 1e-6
        robust_iqr = torch.where(iqr > eps, iqr, torch.full_like(iqr, eps))
        return (actual - median).abs() / robust_iqr

    return score


def quantile_exceedance_scorer(quantiles: torch.Tensor) -> Scorer:
    """Return a scorer based on the empirical CDF of the observation.

    The score is ``2 * |p - 0.5|`` where ``p`` is the fraction of predicted
    quantiles below the observed value. It is bounded in ``[0, 1]``.
    """
    q = quantiles.view(-1, 1)  # [Q, 1]

    def score(forecast: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        # forecast: [V, Q], actual: [V]
        actual = actual.unsqueeze(-1)  # [V, 1]
        p = (forecast <= actual).to(torch.float32).mean(dim=-1)  # [V]
        return 2.0 * (p - 0.5).abs()

    return score


def crps_residual_scorer(quantiles: torch.Tensor) -> Scorer:
    """Return a scorer that uses the CRPS of the observation under the forecast."""
    q = quantiles.view(1, -1, 1)  # [1, Q, 1]

    def score(forecast: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        # forecast: [V, Q], actual: [V]
        target = actual.unsqueeze(-1)  # [V, 1]
        errors = target - forecast  # [V, Q]
        loss = torch.where(
            errors >= 0,
            q.squeeze(0) * errors,
            (q.squeeze(0) - 1.0) * errors,
        )
        return loss.mean(dim=-1)  # [V]

    return score


SCORERS: dict[str, Scorer] = {
    "iqr_deviation": iqr_deviation_scorer,
    "quantile_exceedance": quantile_exceedance_scorer,
    "crps_residual": crps_residual_scorer,
}


def get_scorer(name: str, quantiles: torch.Tensor) -> Scorer:
    """Resolve a scorer name to a callable."""
    if name not in SCORERS:
        raise ValueError(f"Unknown scorer {name!r}. Available: {list(SCORERS.keys())}")
    return SCORERS[name](quantiles)
