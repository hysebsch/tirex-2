"""Tests for TiRex-2 anomaly detection."""

from __future__ import annotations

import pytest
import torch

from tirex2 import TimeseriesType
from tirex2.pro.anomaly import TimeSeriesAnomalyDetector


def _inject_anomaly(target: torch.Tensor, start: int, length: int, scale: float = 5.0) -> torch.Tensor:
    """Add a level-shift anomaly to a univariate target tensor."""
    out = target.clone()
    out[..., start : start + length] += scale * target.std()
    return out


def _build_series_with_anomaly(
    context_len: int,
    pred_len: int,
    anomaly_start: int,
    anomaly_len: int,
    scale: float = 5.0,
) -> tuple[TimeseriesType, TimeseriesType]:
    """Return (clean_series, anomalous_series) sharing a normal prefix."""
    total_len = anomaly_start + anomaly_len + 16
    normal_target = torch.sin(torch.linspace(0, 4 * 3.14159, total_len)) + torch.randn(total_len) * 0.1
    normal_target = normal_target.unsqueeze(0)
    anomalous_target = _inject_anomaly(normal_target, anomaly_start, anomaly_len, scale=scale)

    future_cov = torch.zeros(1, total_len)
    clean = TimeseriesType(target=normal_target, past_covariates=None, future_covariates=future_cov)
    anomalous = TimeseriesType(target=anomalous_target, past_covariates=None, future_covariates=future_cov)
    return clean, anomalous


def test_anomaly_detector_score_shape(build_small_model) -> None:
    model = build_small_model("cpu").eval()
    context_len = model.context_len
    total_len = context_len + 20

    target = torch.randn(2, total_len)
    future_cov = torch.zeros(1, total_len)
    ts = TimeseriesType(target=target, past_covariates=None, future_covariates=future_cov)

    detector = TimeSeriesAnomalyDetector(model, prediction_length=1)
    result = detector.score(ts)

    assert result.per_variate_scores.shape == (2, total_len)
    assert result.global_scores.shape == (total_len,)
    # First context_len steps are unscored.
    assert torch.isnan(result.per_variate_scores[:, :context_len]).all()
    assert torch.isfinite(result.per_variate_scores[:, context_len:]).all()


def test_anomaly_detector_flags_injected_anomaly(build_small_model) -> None:
    model = build_small_model("cpu").eval()
    context_len = model.context_len
    pred_len = model.future_len
    anomaly_start = context_len + 4
    anomaly_len = 4

    clean, anomalous = _build_series_with_anomaly(context_len, pred_len, anomaly_start, anomaly_len, scale=8.0)

    detector = TimeSeriesAnomalyDetector(model, prediction_length=1, scorer="iqr_deviation")
    detector.fit_threshold([clean], method="percentile", percentile=95.0)

    result = detector.predict(anomalous)
    # Anomaly region should be flagged.
    flagged = result.global_labels[anomaly_start : anomaly_start + anomaly_len]
    assert flagged.any(), "Expected at least one anomalous step to be flagged"


def test_anomaly_detector_quantile_exceedance_scorer(build_small_model) -> None:
    model = build_small_model("cpu").eval()
    context_len = model.context_len
    total_len = context_len + 10

    target = torch.randn(1, total_len)
    ts = TimeseriesType(target=target, past_covariates=None, future_covariates=torch.zeros(1, total_len))

    detector = TimeSeriesAnomalyDetector(model, prediction_length=1, scorer="quantile_exceedance")
    result = detector.score(ts)

    assert result.scorer == "quantile_exceedance"
    assert torch.isfinite(result.per_variate_scores[:, context_len:]).all()
    # Exceedance scores are bounded in [0, 1].
    valid_scores = result.per_variate_scores[:, context_len:]
    assert (valid_scores >= 0.0).all() and (valid_scores <= 1.0).all()


def test_anomaly_detector_contamination_threshold(build_small_model) -> None:
    model = build_small_model("cpu").eval()
    context_len = model.context_len
    total_len = context_len + 50

    target = torch.randn(1, total_len)
    ts = TimeseriesType(target=target, past_covariates=None, future_covariates=torch.zeros(1, total_len))

    detector = TimeSeriesAnomalyDetector(model, prediction_length=1)
    threshold = detector.fit_threshold([ts], method="contamination", contamination=0.02)

    result = detector.predict(ts)
    assert threshold >= 0.0
    # Around 2% of the scored steps should be flagged.
    scored = result.global_scores[context_len:]
    flagged = result.global_labels[context_len:]
    expected = int(0.02 * scored.numel())
    assert abs(int(flagged.sum().item()) - expected) <= 1


def test_anomaly_detector_predict_requires_threshold(build_small_model) -> None:
    model = build_small_model("cpu").eval()
    context_len = model.context_len
    ts = TimeseriesType(
        target=torch.randn(1, context_len + 5),
        past_covariates=None,
        future_covariates=torch.zeros(1, context_len + 5),
    )
    detector = TimeSeriesAnomalyDetector(model, prediction_length=1)
    with pytest.raises(RuntimeError, match="Threshold not set"):
        detector.predict(ts)
