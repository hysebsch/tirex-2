"""Unsupervised anomaly detection for multivariate time series using TiRex-2.

The detector compares observed values against the model's quantile forecasts
via a one-step-ahead rolling window. Anomaly thresholds are calibrated on a
reference dataset that is assumed to be mostly normal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from ...api_adapter.forecast import ForecastModel
from ...model.types import TimeseriesType
from .scorers import get_scorer


@dataclass
class AnomalyResult:
    """Container for anomaly detection outputs."""

    per_variate_scores: torch.Tensor  # [V_t, T]
    global_scores: torch.Tensor  # [T]
    per_variate_labels: torch.Tensor  # [V_t, T]
    global_labels: torch.Tensor  # [T]
    threshold: float
    scorer: str
    aggregation: str


class TimeSeriesAnomalyDetector:
    """Detect anomalous time steps in multivariate series with TiRex-2.

    Parameters
    ----------
    model
        An instantiated :class:`tirex2.TiRex2` or :class:`tirex2.ForecastModel`.
    prediction_length : int
        Forecast horizon used for scoring. For online point-level detection
        this is typically ``1``.
    scorer : {"iqr_deviation", "quantile_exceedance", "crps_residual"}
        Scoring function. ``iqr_deviation`` is the default.
    aggregation : {"max", "mean"}
        How to aggregate per-variate scores into a global score.
    context_length : int | None
        Length of the rolling context window. ``None`` uses the model's
        ``context_len``.

    Examples
    --------
    >>> detector = TimeSeriesAnomalyDetector(model, scorer="iqr_deviation")
    >>> detector.fit_threshold(reference_series, percentile=99.0)
    >>> result = detector.predict(series)
    >>> result.global_labels  # [T]
    """

    def __init__(
        self,
        model: Any,
        *,
        prediction_length: int = 1,
        scorer: Literal["iqr_deviation", "quantile_exceedance", "crps_residual"] = "iqr_deviation",
        aggregation: Literal["max", "mean"] = "max",
        context_length: int | None = None,
    ) -> None:
        if isinstance(model, ForecastModel):
            self.model = model
        else:
            self.model = ForecastModel(model)

        self.prediction_length = int(prediction_length)
        if self.prediction_length < 1:
            raise ValueError(f"prediction_length must be >= 1, got {prediction_length}")

        self.scorer_name = scorer
        self.aggregation = aggregation

        model_context_len = getattr(self.model, "context_len", None)
        if context_length is not None:
            self.context_length = int(context_length)
        elif model_context_len is not None:
            self.context_length = int(model_context_len)
        else:
            raise ValueError("context_length must be provided when the model has no context_len attribute.")

        if self.context_length < 1:
            raise ValueError(f"context_length must be >= 1, got {self.context_length}")

        quantiles = getattr(self.model, "quantiles", torch.tensor([0.1, 0.5, 0.9]))
        self._scorer = get_scorer(scorer, quantiles)
        self.threshold: float | None = None

    def fit_threshold(
        self,
        reference_data: Sequence[TimeseriesType],
        *,
        method: Literal["percentile", "contamination"] = "percentile",
        percentile: float = 99.0,
        contamination: float = 0.01,
    ) -> float:
        """Calibrate the global anomaly threshold on reference data.

        Parameters
        ----------
        reference_data
            Series assumed to contain mostly normal behavior.
        method
            ``"percentile"`` sets the threshold to the given percentile of the
            global score distribution. ``"contamination"`` sets it so the top
            ``contamination`` fraction of reference scores is flagged.
        percentile
            Percentile used when ``method="percentile"``.
        contamination
            Expected anomaly fraction used when ``method="contamination"``.
        """
        all_global_scores: list[torch.Tensor] = []
        for series in reference_data:
            result = self.score(series)
            all_global_scores.append(result.global_scores)

        scores = torch.cat([s[~torch.isnan(s)] for s in all_global_scores])
        if scores.numel() == 0:
            self.threshold = float("inf")
            return self.threshold

        if method == "percentile":
            q = percentile / 100.0
            self.threshold = float(torch.quantile(scores, q).item())
        elif method == "contamination":
            k = max(1, int(contamination * scores.numel()))
            topk = torch.topk(scores, k).values
            self.threshold = float(topk[-1].item())
        else:
            raise ValueError(f"Unknown threshold method {method!r}")

        return self.threshold

    def score(self, timeseries: TimeseriesType) -> AnomalyResult:
        """Compute anomaly scores for every time step in ``timeseries``."""
        target = timeseries.target
        if target.ndim == 1:
            target = target.unsqueeze(0)

        num_variates, total_length = target.shape
        per_variate_scores = torch.full(
            (num_variates, total_length),
            float("nan"),
            dtype=torch.float32,
            device=target.device,
        )

        # Score positions t = context_length ... total_length - 1 using a
        # one-step-ahead forecast from the preceding context_length steps.
        for t in range(self.context_length, total_length):
            context = self._slice_context(timeseries, t - self.context_length, t)
            forecast = self._forecast_one_step(context)
            actual = target[..., t]
            score = self._scorer(forecast, actual)  # [V]
            per_variate_scores[..., t] = score

        global_scores = self._aggregate(per_variate_scores)
        labels = self._label(global_scores, per_variate_scores)

        return AnomalyResult(
            per_variate_scores=per_variate_scores,
            global_scores=global_scores,
            per_variate_labels=labels[0],
            global_labels=labels[1],
            threshold=self.threshold if self.threshold is not None else float("nan"),
            scorer=self.scorer_name,
            aggregation=self.aggregation,
        )

    def predict(self, timeseries: TimeseriesType) -> AnomalyResult:
        """Return anomaly labels and scores for ``timeseries``.

        A threshold must have been calibrated with ``fit_threshold`` first,
        otherwise a RuntimeError is raised.
        """
        if self.threshold is None:
            raise RuntimeError("Threshold not set. Call fit_threshold() before predict().")
        return self.score(timeseries)

    def _slice_context(
        self,
        timeseries: TimeseriesType,
        start: int,
        end: int,
    ) -> TimeseriesType:
        """Return a ``TimeseriesType`` spanning ``[start, end)`` plus the
        required future-known covariate horizon."""
        target = timeseries.target[..., start:end]
        past_cov = (
            timeseries.past_covariates[..., start:end]
            if timeseries.past_covariates is not None
            else None
        )
        future_cov = (
            timeseries.future_covariates[..., start : end + self.prediction_length]
            if timeseries.future_covariates is not None
            else None
        )
        return TimeseriesType(target=target, past_covariates=past_cov, future_covariates=future_cov)

    def _forecast_one_step(self, context: TimeseriesType) -> torch.Tensor:
        """Run a one-step forecast and return quantiles per target variate."""
        device = next(self.model.parameters()).device
        forecast = self.model.forecast(
            [context],
            prediction_length=self.prediction_length,
            output_type="torch",
            batch_size=1,
        )
        # forecast is a list with one [V_t, Q, H] tensor.
        f = forecast[0].to(device)
        # Reduce to the first forecast step if prediction_length > 1.
        return f[..., 0]  # [V_t, Q]

    def _aggregate(self, per_variate_scores: torch.Tensor) -> torch.Tensor:
        """Aggregate per-variate scores into a global score per time step."""
        if self.aggregation == "max":
            return per_variate_scores.nan_to_num(nan=float("-inf")).max(dim=0).values
        elif self.aggregation == "mean":
            return per_variate_scores.nanmean(dim=0)
        else:
            raise ValueError(f"Unknown aggregation {self.aggregation!r}")

    def _label(
        self,
        global_scores: torch.Tensor,
        per_variate_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the fitted threshold to produce binary labels."""
        if self.threshold is None:
            threshold = float("inf")
        else:
            threshold = self.threshold

        per_labels = (per_variate_scores > threshold).to(torch.bool)
        # Only label positions that have a valid score.
        per_labels = per_labels & ~torch.isnan(per_variate_scores)
        global_labels = (global_scores > threshold).to(torch.bool)
        global_labels = global_labels & ~torch.isnan(global_scores)
        return per_labels, global_labels
