"""Scoring functions for forecast-deviation anomaly detection.

Each scorer receives a one-step-ahead quantile forecast and the observed value
and returns a per-variate anomaly score. Higher scores indicate more anomalous
observations.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

Scorer = Callable[..., torch.Tensor]


def _find_quantile_index(quantiles: torch.Tensor, level: float) -> int:
    """Return the index of the quantile level closest to ``level``."""
    return int((quantiles - level).abs().argmin().item())


def _median_iqr(quantiles: torch.Tensor, forecast: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return median and robust IQR from a [V, Q] forecast tensor."""
    median_idx = _find_quantile_index(quantiles, 0.5)
    low_idx = _find_quantile_index(quantiles, 0.1)
    high_idx = _find_quantile_index(quantiles, 0.9)

    median = forecast[:, median_idx]
    iqr = (forecast[:, high_idx] - forecast[:, low_idx]).abs()
    eps = 1e-6
    robust_iqr = torch.where(iqr > eps, iqr, torch.full_like(iqr, eps))
    return median, robust_iqr


def iqr_deviation_scorer(quantiles: torch.Tensor) -> Scorer:
    """Return a scorer based on normalized deviation from the median.

    The score is ``|actual - median| / max(iqr, eps)`` where ``iqr`` is the
    inter-quantile range ``q_high - q_low`` (default 0.9 - 0.1).
    """

    def score(forecast: torch.Tensor, actual: torch.Tensor, history: torch.Tensor | None = None) -> torch.Tensor:
        # forecast: [V, Q], actual: [V]
        median, robust_iqr = _median_iqr(quantiles, forecast)
        return (actual - median).abs() / robust_iqr

    return score


def quantile_exceedance_scorer(quantiles: torch.Tensor) -> Scorer:
    """Return a scorer based on the empirical CDF of the observation.

    The score is ``2 * |p - 0.5|`` where ``p`` is the fraction of predicted
    quantiles below the observed value. It is bounded in ``[0, 1]``.
    """
    q = quantiles.view(-1, 1)  # [Q, 1]

    def score(forecast: torch.Tensor, actual: torch.Tensor, history: torch.Tensor | None = None) -> torch.Tensor:
        # forecast: [V, Q], actual: [V]
        actual = actual.unsqueeze(-1)  # [V, 1]
        p = (forecast <= actual).to(torch.float32).mean(dim=-1)  # [V]
        return 2.0 * (p - 0.5).abs()

    return score


def crps_residual_scorer(quantiles: torch.Tensor) -> Scorer:
    """Return a scorer that uses the CRPS of the observation under the forecast."""
    q = quantiles.view(1, -1, 1)  # [1, Q, 1]

    def score(forecast: torch.Tensor, actual: torch.Tensor, history: torch.Tensor | None = None) -> torch.Tensor:
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


def trend_residual_scorer(quantiles: torch.Tensor, window: int = 6) -> Scorer:
    """Return a scorer that flags sustained directional anomalies.

    At each step the current signed, normalized residual is compared against the
    mean of the last ``window`` residuals. A large positive or negative gap flags
    a trend / level-shift anomaly. Scores are in units of normalized residuals.

    Parameters
    ----------
    quantiles
        Native quantile levels of the forecasting model.
    window
        Number of recent residuals to use as the local baseline.
    """

    def score(forecast: torch.Tensor, actual: torch.Tensor, history: torch.Tensor | None = None) -> torch.Tensor:
        # forecast: [V, Q], actual: [V], history: [V, window] or None
        median, robust_iqr = _median_iqr(quantiles, forecast)
        residual = (actual - median) / robust_iqr  # [V]

        if history is None or history.shape[-1] == 0:
            return residual.abs()

        # Local mean of recent signed residuals.
        local_mean = history.nanmean(dim=-1)  # [V]
        return (residual - local_mean).abs()

    return score


def volatility_residual_scorer(quantiles: torch.Tensor, window: int = 6) -> Scorer:
    """Return a scorer that flags volatility / variance bursts.

    The current absolute normalized residual is compared against the mean of
    recent absolute residuals. A ratio above 1 indicates the current step is more
    surprising than the recent local volatility baseline.

    Parameters
    ----------
    quantiles
        Native quantile levels of the forecasting model.
    window
        Number of recent absolute residuals to use as the local volatility
        baseline.
    """

    def score(forecast: torch.Tensor, actual: torch.Tensor, history: torch.Tensor | None = None) -> torch.Tensor:
        # forecast: [V, Q], actual: [V], history: [V, window] or None
        median, robust_iqr = _median_iqr(quantiles, forecast)
        residual_abs = (actual - median).abs() / robust_iqr  # [V]

        if history is None or history.shape[-1] == 0:
            return residual_abs

        local_vol = history.abs().nanmean(dim=-1)  # [V]
        eps = 1e-6
        local_vol = torch.where(local_vol > eps, local_vol, torch.full_like(local_vol, eps))
        return residual_abs / local_vol

    return score


SCORERS: dict[str, Callable[..., Scorer]] = {
    "iqr_deviation": iqr_deviation_scorer,
    "quantile_exceedance": quantile_exceedance_scorer,
    "crps_residual": crps_residual_scorer,
    "trend_residual": trend_residual_scorer,
    "volatility_residual": volatility_residual_scorer,
}


def get_scorer(name: str, quantiles: torch.Tensor, *, window: int = 6) -> Scorer:
    """Resolve a scorer name to a callable.

    For ``trend_residual`` and ``volatility_residual`` the ``window`` argument
    controls the number of recent residuals used to define the local baseline.
    """
    if name not in SCORERS:
        raise ValueError(f"Unknown scorer {name!r}. Available: {list(SCORERS.keys())}")
    if name in ("trend_residual", "volatility_residual"):
        return SCORERS[name](quantiles, window=window)
    return SCORERS[name](quantiles)
