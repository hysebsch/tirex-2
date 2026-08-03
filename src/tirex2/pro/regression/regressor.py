"""Regression head for scalar or vector targets derived from time series.

Examples include predicting the next value of a derived metric, estimating
remaining useful life, or forecasting a summary statistic.
"""

from __future__ import annotations

from typing import Any


class TimeSeriesRegressor:
    """Predict scalar/vector targets from a time series.

    Parameters
    ----------
    model
        An instantiated :class:`tirex2.TiRex2`.
    output_dim : int
        Dimensionality of the regression target.
    """

    def __init__(self, model: Any, output_dim: int) -> None:
        self.model = model
        self.output_dim = output_dim

    def fit(self, data: Any, *, epochs: int = 1, learning_rate: float = 1e-4) -> None:
        """Train the regression head."""
        raise NotImplementedError("Regression training is not implemented yet.")

    def predict(self, timeseries: Any) -> Any:
        """Return regression predictions."""
        raise NotImplementedError("Regression prediction is not implemented yet.")
