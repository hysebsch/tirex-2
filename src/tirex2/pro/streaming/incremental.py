"""Incremental forecast updates as new observations arrive.

The open-source TiRex-2 model recomputes over the full context on every call.
The Pro streaming API will cache recurrent state between calls so that adding a
single new observation does not require a full forward pass over the history.
"""

from __future__ import annotations

from typing import Any

from ...model.types import TimeseriesType


class IncrementalForecaster:
    """Stateful wrapper for low-latency incremental forecasts.

    Parameters
    ----------
    model
        An instantiated :class:`tirex2.TiRex2` or :class:`tirex2.ForecastModel`.
    prediction_length : int
        Number of future steps to predict on each call.
    """

    def __init__(self, model: Any, prediction_length: int) -> None:
        self.model = model
        self.prediction_length = prediction_length
        self._state: Any | None = None

    def update(self, timeseries: TimeseriesType) -> None:
        """Ingest a new observation and advance the cached recurrent state."""
        raise NotImplementedError("Streaming update is not implemented yet.")

    def forecast(self) -> Any:
        """Return a forecast from the current cached state."""
        raise NotImplementedError("Streaming forecast is not implemented yet.")

    def reset(self) -> None:
        """Drop the cached state and force a full-context forecast next time."""
        self._state = None
