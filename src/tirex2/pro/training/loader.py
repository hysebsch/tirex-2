"""Data loaders for converting common formats into ``TimeseriesType`` batches.

This module provides helpers to build lists of ``TimeseriesType`` from:

- CSV files (one file per series or one long-form file)
- Parquet files
- GluonTS datasets
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from ...model.types import TimeseriesType


def _to_tensor(x: Any) -> torch.Tensor:
    """Convert a sequence or scalar to a float tensor."""
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(x, dtype=torch.float32)


def load_series_from_csv(
    path: str | Path,
    *,
    target_columns: str | Sequence[str] | None = None,
    past_covariate_columns: Sequence[str] | None = None,
    future_covariate_columns: Sequence[str] | None = None,
    time_column: str | None = None,
) -> list[TimeseriesType]:
    """Load ``TimeseriesType`` objects from a CSV file.

    Parameters
    ----------
    path
        Path to a CSV file.
    target_columns
        Column name(s) holding target values. ``None`` means all columns that
        are not covariates. A string loads a single target variate.
    past_covariate_columns
        Column names for past-only covariates.
    future_covariate_columns
        Column names for future-known covariates. These columns must extend
        ``prediction_length`` steps beyond the target if they are used during
        forecasting.
    time_column
        Optional column used to sort rows if the file is not already ordered.

    Returns
    -------
    list[TimeseriesType]
        One ``TimeseriesType`` per series. If the CSV contains multiple series,
        use a ``series_id`` column and call ``load_series_from_long_csv`` instead.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("load_series_from_csv requires pandas: pip install pandas") from exc

    df = pd.read_csv(path)
    if time_column is not None:
        df = df.sort_values(time_column)

    past_covariate_columns = list(past_covariate_columns or [])
    future_covariate_columns = list(future_covariate_columns or [])

    if target_columns is None:
        excluded = set(past_covariate_columns + future_covariate_columns + ([time_column] if time_column else []))
        target_columns = [c for c in df.columns if c not in excluded]
    elif isinstance(target_columns, str):
        target_columns = [target_columns]
    else:
        target_columns = list(target_columns)

    target = _to_tensor(df[target_columns].to_numpy().T)  # [V_t, T]
    past_cov = _to_tensor(df[past_covariate_columns].to_numpy().T) if past_covariate_columns else None
    future_cov = _to_tensor(df[future_covariate_columns].to_numpy().T) if future_covariate_columns else None

    return [TimeseriesType(target=target, past_covariates=past_cov, future_covariates=future_cov)]


def load_series_from_long_csv(
    path: str | Path,
    series_id_column: str,
    *,
    target_columns: str | Sequence[str],
    past_covariate_columns: Sequence[str] | None = None,
    future_covariate_columns: Sequence[str] | None = None,
    time_column: str | None = None,
) -> list[TimeseriesType]:
    """Load multiple series from a long-form CSV file.

    Each unique value in ``series_id_column`` becomes one ``TimeseriesType``.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("load_series_from_long_csv requires pandas: pip install pandas") from exc

    df = pd.read_csv(path)
    if time_column is not None:
        df = df.sort_values([series_id_column, time_column])

    if isinstance(target_columns, str):
        target_columns = [target_columns]
    else:
        target_columns = list(target_columns)
    past_covariate_columns = list(past_covariate_columns or [])
    future_covariate_columns = list(future_covariate_columns or [])

    series = []
    for _, group in df.groupby(series_id_column):
        target = _to_tensor(group[target_columns].to_numpy().T)
        past_cov = _to_tensor(group[past_covariate_columns].to_numpy().T) if past_covariate_columns else None
        future_cov = _to_tensor(group[future_covariate_columns].to_numpy().T) if future_covariate_columns else None
        series.append(
            TimeseriesType(target=target, past_covariates=past_cov, future_covariates=future_cov)
        )
    return series


def load_series_from_parquet(
    path: str | Path,
    *,
    target_columns: str | Sequence[str] | None = None,
    past_covariate_columns: Sequence[str] | None = None,
    future_covariate_columns: Sequence[str] | None = None,
    time_column: str | None = None,
) -> list[TimeseriesType]:
    """Load ``TimeseriesType`` objects from a Parquet file.

    Parameters match ``load_series_from_csv``.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("load_series_from_parquet requires pandas: pip install pandas") from exc

    df = pd.read_parquet(path)
    if time_column is not None:
        df = df.sort_values(time_column)

    past_covariate_columns = list(past_covariate_columns or [])
    future_covariate_columns = list(future_covariate_columns or [])

    if target_columns is None:
        excluded = set(past_covariate_columns + future_covariate_columns + ([time_column] if time_column else []))
        target_columns = [c for c in df.columns if c not in excluded]
    elif isinstance(target_columns, str):
        target_columns = [target_columns]
    else:
        target_columns = list(target_columns)

    target = _to_tensor(df[target_columns].to_numpy().T)
    past_cov = _to_tensor(df[past_covariate_columns].to_numpy().T) if past_covariate_columns else None
    future_cov = _to_tensor(df[future_covariate_columns].to_numpy().T) if future_covariate_columns else None

    return [TimeseriesType(target=target, past_covariates=past_cov, future_covariates=future_cov)]


def load_series_from_gluonts(
    dataset: Any,
    *,
    max_samples: int | None = None,
    prediction_length: int | None = None,
) -> list[TimeseriesType]:
    """Load ``TimeseriesType`` objects from a GluonTS dataset.

    Parameters
    ----------
    dataset
        A GluonTS ``Dataset`` (e.g. from ``gluonts.dataset.common.load_datasets``)
        or any iterable of entries with ``target``, ``start``, and optional
        ``feat_dynamic_real`` / ``feat_static_cat`` fields.
    max_samples
        If given, load at most this many entries.
    prediction_length
        Forecast horizon used to determine the required length of future-known
        covariates. If omitted, covariates are loaded as past-only.

    Returns
    -------
    list[TimeseriesType]
    """
    try:
        from gluonts.dataset.field_names import FieldName
    except ImportError as exc:
        raise ImportError("load_series_from_gluonts requires gluonts: pip install gluonts") from exc

    series = []
    for idx, entry in enumerate(dataset):
        if max_samples is not None and idx >= max_samples:
            break

        target = _to_tensor(entry[FieldName.TARGET])  # may be [T] or [V, T]
        if target.ndim == 1:
            target = target.unsqueeze(0)

        past_cov: torch.Tensor | None = None
        future_cov: torch.Tensor | None = None

        # Dynamic real features are typically [C, T].
        dynamic = entry.get(FieldName.FEAT_DYNAMIC_REAL)
        if dynamic is not None:
            dynamic = _to_tensor(dynamic)
            if dynamic.ndim == 1:
                dynamic = dynamic.unsqueeze(0)
            total_len = dynamic.shape[-1]
            target_len = target.shape[-1]
            if prediction_length is not None and total_len >= target_len + prediction_length:
                future_cov = dynamic[..., : target_len + prediction_length]
                past_cov = dynamic[..., :target_len]
            else:
                past_cov = dynamic

        series.append(
            TimeseriesType(target=target, past_covariates=past_cov, future_covariates=future_cov)
        )

    return series


def save_series_to_pt(
    series: Sequence[TimeseriesType],
    output_dir: str | Path,
    *,
    prefix: str = "series",
) -> list[Path]:
    """Save a list of ``TimeseriesType`` as ``.pt`` files for fast reloading.

    Returns the list of written file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, ts in enumerate(series):
        path = output_dir / f"{prefix}_{i:05d}.pt"
        torch.save(ts, path)
        paths.append(path)
    return paths


def load_series_from_pt(
    directory: str | Path,
    *,
    prefix: str = "series",
    max_samples: int | None = None,
) -> list[TimeseriesType]:
    """Load ``TimeseriesType`` objects from ``.pt`` files in a directory."""
    directory = Path(directory)
    files = sorted(directory.glob(f"{prefix}_*.pt"))
    if max_samples is not None:
        files = files[:max_samples]
    return [torch.load(f, weights_only=False) for f in files]
