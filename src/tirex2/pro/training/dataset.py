"""Dataset and collation utilities for TiRex-2 training/fine-tuning.

A training sample is a :class:`TimeseriesType` whose ``target`` tensor contains
both the observed context and the future ground truth. The future target values
are masked with NaN before being fed to the model, preserving causality.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import Dataset

from ...model.types import TimeseriesType


class TiRexDataset(Dataset):
    """Sliding-window dataset over a collection of multivariate time series.

    Parameters
    ----------
    series : list[TimeseriesType]
        Source time series. Each ``TimeseriesType.target`` must have shape
        ``[V_t, total_length]`` where ``total_length >= context_length``.
    context_length : int
        Number of historical time steps to condition on.
    prediction_length : int
        Number of future time steps to forecast.
    stride : int
        Step size between consecutive windows.
    pad_future_target : bool
        If ``True`` (default), keep the future target values in the returned
        sample so the loss can supervise them. The training loop is responsible
        for masking them before calling the model.
    """

    def __init__(
        self,
        series: Sequence[TimeseriesType],
        context_length: int,
        prediction_length: int,
        stride: int = 1,
        pad_future_target: bool = True,
    ) -> None:
        if context_length < 1 or prediction_length < 1:
            raise ValueError("context_length and prediction_length must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.pad_future_target = pad_future_target
        self._window_length = context_length + prediction_length

        self._windows: list[tuple[int, int]] = []
        for series_idx, ts in enumerate(series):
            target_len = ts.target.shape[-1]
            if target_len < self._window_length:
                continue
            num_windows = (target_len - self._window_length) // stride + 1
            for w in range(num_windows):
                start = w * stride
                self._windows.append((series_idx, start))
        self._series = list(series)

        if not self._windows:
            raise ValueError(
                f"TiRexDataset produced zero windows: each series target length must be >= "
                f"context_length + prediction_length ({self._window_length}), but the longest "
                f"target length was {max((ts.target.shape[-1] for ts in series), default=0)}. "
                f"Either use longer series, a smaller context_length, or a smaller prediction_length."
            )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> TimeseriesType:
        series_idx, start = self._windows[idx]
        ts = self._series[series_idx]
        end = start + self._window_length

        target_window = ts.target[..., start:end]
        past_cov = self._slice_covariate(ts.past_covariates, start, end)

        # Future-known covariates span the whole window.
        if ts.future_covariates is not None:
            future_cov = ts.future_covariates[..., start:end]
        else:
            future_cov = None

        if self.pad_future_target:
            # Keep the real future targets; the training loop masks them before forward.
            return TimeseriesType(
                target=target_window,
                past_covariates=past_cov,
                future_covariates=future_cov,
            )

        # Replace future target values with NaN so the sample is already causal.
        mask = torch.ones_like(target_window, dtype=torch.bool)
        mask[..., self.context_length :] = False
        causal_target = target_window.clone()
        causal_target[~mask] = float("nan")
        return TimeseriesType(
            target=causal_target,
            past_covariates=past_cov,
            future_covariates=future_cov,
        )

    def _slice_covariate(
        self,
        cov: torch.Tensor | None,
        start: int,
        end: int,
    ) -> torch.Tensor | None:
        if cov is None:
            return None
        if cov.shape[-1] < end:
            raise ValueError(f"Covariate length {cov.shape[-1]} is shorter than requested window end {end}")
        return cov[..., start:end]


def collate_timeseries(
    samples: Sequence[TimeseriesType],
    postprocessor: Any,
    prediction_length: int,
    device: torch.device | str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Collate a list of ``TimeseriesType`` into a model batch and ground-truth targets.

    This uses the model's postprocessor to pack targets and covariates into the
    group-vector format the backbone expects, while preserving the raw full targets
    for loss computation. The postprocessor expects target tensors of context-only
    length and future-known covariates of length ``context + prediction``; this
    function splits full-window samples accordingly.

    Parameters
    ----------
    samples : list[TimeseriesType]
        Training windows. ``target`` is expected to have shape ``[V_t, context + prediction]``.
    postprocessor
        The model's ``postprocessor`` instance (e.g. ``model.postprocessor``).
    prediction_length : int
        Forecast horizon.
    device : torch.device | str | None
        Device to move tensors to. If ``None``, uses the device of the first target.

    Returns
    -------
    input_batch : dict[str, torch.Tensor]
        Batch ready for ``TiRex2.forward``: ``x``, ``group_vector``, ``target_mask``.
    targets : dict[str, torch.Tensor]
        Ground-truth ``target`` tensor and an ``observed`` mask.
    """
    if not samples:
        raise ValueError("collate_timeseries received an empty sample list")

    # Infer context length from the sample target length.
    window_lengths = [ts.target.shape[-1] for ts in samples]
    window_length = max(window_lengths)
    context_length = window_length - prediction_length
    if context_length < 1:
        raise ValueError(f"Sample target length {window_length} must be > prediction_length {prediction_length}")

    # Build full targets for the loss before the postprocessor masks them.
    full_targets = torch.cat([ts.target for ts in samples], dim=0)
    if device is None:
        device = full_targets.device
    else:
        full_targets = full_targets.to(device)
    observed = ~torch.isnan(full_targets)

    # Split target into context-only for the model input; keep full target for loss.
    context_targets = [ts.target[..., :context_length].clone() for ts in samples]
    past_covariates = [
        ts.past_covariates[..., :context_length].clone() if ts.past_covariates is not None else None for ts in samples
    ]

    # Future-known covariates already span the whole window.
    future_covariates = [ts.future_covariates for ts in samples]

    input_batch, _args, kwargs = postprocessor.transform_input(
        target=context_targets,
        prediction_length=prediction_length,
        past_covariates=past_covariates,
        past_future_covariates=future_covariates,
        tta_diff=False,
    )
    input_batch = {k: v.to(device) for k, v in input_batch.items()}

    return input_batch, {"target": full_targets, "observed": observed}


def pad_to_model_length(
    input_batch: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Pad/truncate ``x`` to the length expected by the model during inference.

    Mirrors the logic in ``TiRex2._predict``: right-pad to ``future_len``,
    then left-pad or truncate to ``context_len + future_len``.
    """
    x = input_batch["x"]
    prediction_length = getattr(model, "future_len", x.shape[-1])
    future_len = getattr(model, "future_len", prediction_length)
    context_len = getattr(model, "context_len", x.shape[-1])
    max_ts_len = context_len + future_len

    # Right-pad to future_len (postprocessor already pads by prediction_length).
    if x.shape[-1] < future_len:
        x = torch.nn.functional.pad(x, (0, future_len - x.shape[-1]), value=float("nan"))

    # Left-pad or truncate to the model's maximum sequence length.
    if x.shape[-1] < max_ts_len:
        x = torch.nn.functional.pad(x, (max_ts_len - x.shape[-1], 0), value=float("nan"))
    elif x.shape[-1] > max_ts_len:
        x = x[..., -max_ts_len:]

    return {**input_batch, "x": x}
