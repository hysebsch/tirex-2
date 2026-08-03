"""Time-series classification head for TiRex-2.

Adds a classification head on top of the TiRex-2 backbone for tasks such as
anomaly detection, fault classification, or activity recognition.
"""

from __future__ import annotations

from typing import Any


class TimeSeriesClassifier:
    """Classify a time series using a TiRex-2 backbone.

    Parameters
    ----------
    model
        An instantiated :class:`tirex2.TiRex2`.
    num_classes : int
        Number of output classes.
    """

    def __init__(self, model: Any, num_classes: int) -> None:
        self.model = model
        self.num_classes = num_classes

    def fit(self, data: Any, *, epochs: int = 1, learning_rate: float = 1e-4) -> None:
        """Train the classification head."""
        raise NotImplementedError("Classification training is not implemented yet.")

    def predict(self, timeseries: Any) -> Any:
        """Return predicted class labels."""
        raise NotImplementedError("Classification prediction is not implemented yet.")
