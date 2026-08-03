"""Anomaly detection support for multivariate time series."""

from __future__ import annotations

from .detector import AnomalyResult, TimeSeriesAnomalyDetector
from .scorers import SCORERS, get_scorer

__all__ = [
    "AnomalyResult",
    "TimeSeriesAnomalyDetector",
    "SCORERS",
    "get_scorer",
]
