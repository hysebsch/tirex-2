"""Smoke tests for the new trend and volatility anomaly scorers."""

import pytest
import torch

from tirex2 import TimeseriesType
from tirex2.pro.anomaly import TimeSeriesAnomalyDetector


def _clean_step_series(context_len, total_len, seed=5):
    rng = torch.Generator().manual_seed(seed)
    t = torch.arange(total_len, dtype=torch.float32)
    y = torch.sin(2 * 3.14159 * t / 50.0) + torch.randn(total_len, generator=rng) * 0.2
    return y.unsqueeze(0)


def _build_step_shift_series(context_len, total_len, seed=5):
    """Return a series with a clean level shift after the context."""
    y = _clean_step_series(context_len, total_len, seed).squeeze(0)
    y[context_len + 10 :] += 8.0
    return y.unsqueeze(0)


def _build_volatility_burst_series(context_len, total_len, seed=6, burst_std: float = 3.0):
    """Return a series with a low-noise region followed by a high-noise burst."""
    rng = torch.Generator().manual_seed(seed)
    t = torch.arange(total_len, dtype=torch.float32)
    y = torch.sin(2 * 3.14159 * t / 50.0)
    y[: context_len + 30] += torch.randn(context_len + 30, generator=rng) * 0.1
    y[context_len + 30 : context_len + 50] += torch.randn(20, generator=rng) * burst_std
    return y.unsqueeze(0)


@pytest.mark.parametrize("scorer", ["trend_residual", "iqr_deviation"])
def test_trend_scorer_flags_level_shift(build_small_model, scorer):
    model = build_small_model("cpu").eval()
    context_len = model.context_len
    total_len = context_len + 80

    clean = TimeseriesType(
        target=_build_step_shift_series(context_len, total_len, seed=10),
        past_covariates=None,
        future_covariates=torch.zeros(1, total_len),
    )
    test = TimeseriesType(
        target=_build_step_shift_series(context_len, total_len, seed=5),
        past_covariates=None,
        future_covariates=torch.zeros(1, total_len),
    )

    detector = TimeSeriesAnomalyDetector(
        model,
        prediction_length=1,
        scorer=scorer,
        context_length=context_len,
        local_window=6,
    )
    detector.fit_threshold([clean], percentile=98.0)
    result = detector.predict(test)

    shift_flags = result.global_labels[context_len + 10 :]
    assert shift_flags.any(), f"{scorer} should flag a level shift"


def test_volatility_residual_flags_burst(build_small_model):
    model = build_small_model("cpu").eval()
    context_len = model.context_len
    total_len = context_len + 80

    clean = TimeseriesType(
        target=_build_volatility_burst_series(context_len, total_len, seed=11, burst_std=4.0),
        past_covariates=None,
        future_covariates=torch.zeros(1, total_len),
    )
    test = TimeseriesType(
        target=_build_volatility_burst_series(context_len, total_len, seed=6, burst_std=4.0),
        past_covariates=None,
        future_covariates=torch.zeros(1, total_len),
    )

    detector = TimeSeriesAnomalyDetector(
        model,
        prediction_length=1,
        scorer="volatility_residual",
        context_length=context_len,
        local_window=6,
    )
    detector.fit_threshold([clean], percentile=95.0)
    result = detector.predict(test)

    burst_flags = result.global_labels[context_len + 30 : context_len + 50]
    assert burst_flags.any(), "volatility_residual should flag a volatility burst"
