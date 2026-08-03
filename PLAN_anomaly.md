# Plan: Multivariate Time-Series Anomaly Detection with TiRex-2

## Goal

Add a Pro anomaly-detection capability so TiRex-2 can flag anomalous time steps
(and anomalous variates) in multivariate series, using the pre-trained
quantile-forecasting backbone.

## Proposed approach: forecast-deviation anomaly detector

TiRex-2 already produces per-variate quantile forecasts. A natural, label-free
way to detect anomalies is to measure how "surprising" an observed value is
relative to the forecast distribution:

1. For a given context window, run `model.forecast(..., prediction_length=H)`.
2. Compare the observed future values (or the last observed values) against the
   predicted quantile band.
3. Emit per-variate and global anomaly scores.
4. Fit a threshold on clean reference data (or specify a contamination rate).

This is **unsupervised** at inference time: no anomaly labels are required.
Only normal/historical data are needed to calibrate the threshold.

### Scoring options (choose one default, expose the others)

| Scorer | Description | Output |
|---|---|---|
| `iqr_deviation` | `(actual - median) / (q_high - q_low)` clipped to a robust IQR band. | per variate / time-step |
| `quantile_exceedance` | Fraction of predicted quantiles below the actual value, converted to a signed z-like score. | per variate / time-step |
| `crps_residual` | Continuous ranked probability score of the observation under the forecast distribution, used as a raw anomaly energy. | per variate / time-step |

Default: `iqr_deviation` because it is intuitive and fast.

### Aggregation across variates

- `per_variate`: keep `[V_t, T]` scores.
- `max`: global score = max over variates at each time step.
- `mean`: global score = mean over variates.
- `weighted`: optional learned or user-provided variate weights.

### Threshold fitting

```python
detector.fit_threshold(
    reference_data,          # list[TimeseriesType] assumed mostly normal
    method="percentile",     # or "contamination"
    percentile=99.0,
    contamination=0.01,        # used when method="contamination"
)
```

Returns `threshold` and stores it for `predict()` / `score()`.

### Anomaly labels

`score()` returns raw scores; `predict()` returns binary labels using the fitted
threshold. Both return an `AnomalyResult` dataclass:

```python
@dataclass
class AnomalyResult:
    per_variate_scores: torch.Tensor   # [V_t, T]
    global_scores: torch.Tensor        # [T]
    per_variate_labels: torch.Tensor   # [V_t, T]
    global_labels: torch.Tensor        # [T]
    threshold: float
    scorer: str
    aggregation: str
```

## API sketch

```python
from tirex2.pro.anomaly import TimeSeriesAnomalyDetector

detector = TimeSeriesAnomalyDetector(
    model,
    prediction_length=24,
    scorer="iqr_deviation",
    aggregation="max",
    context_length=None,   # defaults to model.context_len
)

# Optional: calibrate threshold on reference data.
detector.fit_threshold(reference_series, percentile=99.5)

result = detector.predict(series)   # series: TimeseriesType
result.global_labels                # [T]
result.per_variate_scores           # [V_t, T]
```

## Files to add / modify

- `src/tirex2/pro/anomaly/__init__.py` — export `TimeSeriesAnomalyDetector`, `AnomalyResult`.
- `src/tirex2/pro/anomaly/detector.py` — core detector class and scorers.
- `src/tirex2/pro/anomaly/scorers.py` — scoring functions (iqr, quantile_exceedance, crps).
- `src/tirex2/pro/__init__.py` — export `TimeSeriesAnomalyDetector`.
- `test/test_anomaly.py` — tests using synthetic anomalies (spikes, level shifts, noise).
- `TASKS.md` — update closed/open tasks.

## Synthetic anomaly injection for tests

Re-use the existing augmentation module or add a small helper in the test file:

- `SpikeInjection` (already exists).
- `LevelShift`.
- `VarianceChange`.

Tests will verify:
1. An injected spike receives a higher anomaly score than the surrounding normal
   region.
2. `fit_threshold` on clean data produces a threshold that flags the injected
   anomalies but not the clean region.
3. Per-variate scores and global labels have expected shapes.

## Risks / notes

- The backbone forecasts are in level space (postprocessor reverses any
  differencing), so scores are directly interpretable on the original scale.
- `prediction_length` determines the forecast horizon used for scoring. For
  online detection this can be paired with `IncrementalForecaster` so each new
  observation is scored against a fresh forecast.
- Missing values (NaNs) are ignored both in forecasts and in scoring.
- The detector is unsupervised by default; if the user has anomaly labels, a
  supervised `TimeSeriesClassifier` head can be added later.

## Open questions for you

1. **Scoring target:** Do you want point-level anomaly scores (every time step),
   or subsequence/segment-level detection?  
   *Default: point-level.*
2. **Supervision:** Are anomaly labels available for training, or should this be
   fully unsupervised / threshold-based?  
   *Default: unsupervised threshold-based.*
3. **Streaming:** Do you need online scoring (one new observation at a time),
   or batch scoring over complete series?  
   *Default: batch first; optional streaming via `IncrementalForecaster`.*
4. **Output preference:** Should the detector return scores only, or also
   binary labels plus a calibrated threshold?  
   *Default: both scores and binary labels.*

Please confirm or adjust these choices, then I will implement the detector and
add tests.
