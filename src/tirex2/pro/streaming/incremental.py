"""Incremental / streaming forecast updates as new observations arrive.

The open-source TiRex-2 model recomputes over the full context on every call.
This Pro streaming wrapper maintains a rolling context buffer of at most
``context_length`` time steps and re-forecasts from that buffer whenever new
observations are ingested. This keeps inference latency bounded as the series
 grows, rather than recomputing over an ever-lengthening history.

True recurrent-state caching is not exposed by the public backbone; this module
implements the practical streaming contract (bounded-context updates) that the
backbone supports today.
"""

from __future__ import annotations

from typing import Any, Literal

import torch

from ...api_adapter.forecast import ForecastModel
from ...model.types import TimeseriesType


class IncrementalForecaster:
    """Stateful wrapper for low-latency incremental forecasts.

    Parameters
    ----------
    model
        An instantiated :class:`tirex2.TiRex2` or :class:`tirex2.ForecastModel`.
        Attribute access is delegated to the underlying model, so both work.
    prediction_length : int
        Number of future steps to predict on each call.
    context_length : int | None
        Maximum number of past time steps to keep in the rolling buffer. If
        ``None``, the model's ``context_len`` is used.
    output_type : {"torch", "numpy", "gluonts", "fev"}
        Format returned by :meth:`forecast`. Defaults to ``"torch"``.

    Examples
    --------
    >>> forecaster = IncrementalForecaster(model, prediction_length=24)
    >>> forecaster.update(history)        # seed with initial context
    >>> forecast = forecaster.forecast()  # first forecast
    >>> forecaster.update(new_observation)
    >>> forecast = forecaster.forecast()  # updated forecast
    """

    def __init__(
        self,
        model: Any,
        prediction_length: int,
        *,
        context_length: int | None = None,
        output_type: Literal["torch", "numpy", "gluonts", "fev"] = "torch",
    ) -> None:
        # Accept either a raw TiRex2 backbone or the public ForecastModel wrapper.
        if isinstance(model, ForecastModel):
            self.model = model
        else:
            self.model = ForecastModel(model)

        self.prediction_length = prediction_length
        self.output_type = output_type

        # Delegate context_len lookup to the wrapped model.
        model_context_len = getattr(self.model, "context_len", None)
        if context_length is not None:
            self.context_length = context_length
        elif model_context_len is not None:
            self.context_length = int(model_context_len)
        else:
            raise ValueError(
                "context_length must be provided when the model has no context_len attribute."
            )

        if self.context_length < 1:
            raise ValueError(f"context_length must be >= 1, got {self.context_length}")
        if prediction_length < 1:
            raise ValueError(f"prediction_length must be >= 1, got {prediction_length}")

        self._cached: TimeseriesType | None = None
        self._last_forecast: Any | None = None

    def update(self, timeseries: TimeseriesType) -> None:
        """Ingest new observations and update the rolling context buffer.

        ``timeseries`` may be either the full updated history or just the new
        slice. In both cases the last ``context_length`` time steps are retained.
        """
        if not isinstance(timeseries, TimeseriesType):
            raise TypeError(f"update expects a TimeseriesType, got {type(timeseries).__name__}")

        if timeseries.target.ndim == 1:
            new_target = timeseries.target.unsqueeze(0)
        elif timeseries.target.ndim == 2:
            new_target = timeseries.target
        else:
            raise ValueError(f"target must be 1D or 2D, got shape {tuple(timeseries.target.shape)}")

        # If we already have a cached context, concatenate along the time axis.
        if self._cached is not None:
            merged_target = torch.cat([self._cached.target, new_target], dim=-1)
            merged_past = self._concat_covariates(self._cached.past_covariates, timeseries.past_covariates)
            merged_future = self._concat_covariates(self._cached.future_covariates, timeseries.future_covariates)
        else:
            merged_target = new_target
            merged_past = timeseries.past_covariates
            merged_future = timeseries.future_covariates

        # Truncate to the rolling window size.
        self._cached = TimeseriesType(
            target=merged_target[..., -self.context_length :],
            past_covariates=self._truncate_covariate(merged_past, self.context_length),
            future_covariates=self._truncate_future_covariate(merged_future, self.context_length, self.prediction_length),
        )
        self._last_forecast = None

    def forecast(self, **predict_kwargs: Any) -> Any:
        """Return a forecast from the current cached context.

        Any extra keyword arguments are forwarded to the model's ``forecast``
        method (e.g. ``tta_diff`` or ``tta_sign_flip``).
        """
        if self._cached is None:
            raise RuntimeError("No context has been ingested. Call update() before forecast().")

        forecast = self.model.forecast(
            [self._cached],
            prediction_length=self.prediction_length,
            output_type=self.output_type,
            batch_size=1,
            **predict_kwargs,
        )
        # forecast is a list with one element.
        self._last_forecast = forecast[0] if isinstance(forecast, list) else forecast
        return self._last_forecast

    def reset(self) -> None:
        """Drop the cached context and force a fresh history on the next update."""
        self._cached = None
        self._last_forecast = None

    @staticmethod
    def _concat_covariates(
        a: torch.Tensor | None,
        b: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a
        return torch.cat([a, b], dim=-1)

    @staticmethod
    def _truncate_covariate(
        cov: torch.Tensor | None,
        context_length: int,
    ) -> torch.Tensor | None:
        if cov is None:
            return None
        return cov[..., -context_length:]

    @staticmethod
    def _truncate_future_covariate(
        cov: torch.Tensor | None,
        context_length: int,
        prediction_length: int,
    ) -> torch.Tensor | None:
        if cov is None:
            return None
        # Future-known covariates must cover context + prediction.
        required = context_length + prediction_length
        return cov[..., -required:]
