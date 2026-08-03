"""Regression head for scalar or vector targets derived from time series.

Examples include predicting the next value of a derived metric, estimating
remaining useful life, or forecasting a summary statistic.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ...api_adapter.forecast import ForecastModel
from ...model.types import TimeseriesType
from ..training.dataset import collate_timeseries, pad_to_model_length
from ..training.utils import build_optimizer, build_scheduler


class _RegressionDataset(Dataset):
    """Torch dataset for time-series regression pairs."""

    def __init__(
        self,
        series: Sequence[TimeseriesType],
        targets: Sequence[torch.Tensor],
        context_length: int,
        prediction_length: int,
    ) -> None:
        if len(series) != len(targets):
            raise ValueError(f"series and targets must have the same length, got {len(series)} and {len(targets)}")
        self.series = list(series)
        self.targets = list(targets)
        self.context_length = context_length
        self.prediction_length = prediction_length

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, idx: int) -> tuple[TimeseriesType, torch.Tensor]:
        return self.series[idx], self.targets[idx]


class TimeSeriesRegressor:
    """Predict scalar/vector targets from a time series.

    The regression head is placed on top of the frozen TiRex-2 backbone. For
    each target variate the backbone output is mean-pooled over the observed
    time positions, then passed through a small MLP to ``output_dim``.

    Parameters
    ----------
    model
        An instantiated :class:`tirex2.TiRex2` or :class:`tirex2.ForecastModel`.
    output_dim : int
        Dimensionality of the regression target for each target variate.
    hidden_dim : int | None
        Hidden dimension of the regression MLP. If ``None``, defaults to the
        backbone's embedding dimension.
    freeze_backbone : bool
        If ``True`` (default), only the regression head parameters are trained.
    """

    def __init__(
        self,
        model: Any,
        output_dim: int,
        *,
        hidden_dim: int | None = None,
        freeze_backbone: bool = True,
    ) -> None:
        if isinstance(model, ForecastModel):
            self.model = model
        else:
            self.model = ForecastModel(model)
        self.output_dim = int(output_dim)
        self.freeze_backbone = freeze_backbone

        embedding_dim = int(getattr(self.model, "embedding_dim", 128))
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else embedding_dim

        self.head = nn.Sequential(
            nn.Linear(embedding_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(getattr(self.model, "dropout", 0.0)),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

        self.device = next(self.model.parameters()).device
        self.head.to(self.device)

        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

    def fit(
        self,
        data: Sequence[tuple[TimeseriesType, torch.Tensor]],
        *,
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        warmup_ratio: float = 0.1,
        context_length: int | None = None,
        prediction_length: int | None = None,
        output_dir: str | Path | None = None,
        log_interval: int = 10,
    ) -> None:
        """Train the regression head on ``(timeseries, target)`` pairs.

        Parameters
        ----------
        data
            Each element is ``(TimeseriesType, target)`` where ``target`` has
            shape ``[num_target_variates, output_dim]``.
        epochs
            Number of training epochs.
        batch_size
            Samples per batch.
        learning_rate
            Peak learning rate for AdamW.
        weight_decay
            AdamW weight decay.
        grad_clip
            Max gradient norm for clipping.
        warmup_ratio
            Fraction of total steps in linear LR warmup.
        context_length
            If ``None``, the model's ``context_len`` is used.
        prediction_length
            If ``None``, the model's ``future_len`` is used.
        output_dir
            Optional directory to save the trained regression head.
        log_interval
            Log training loss every N batches.
        """
        context_length = context_length if context_length is not None else getattr(self.model, "context_len", 2048)
        prediction_length = prediction_length if prediction_length is not None else getattr(self.model, "future_len", 512)

        series = [ts for ts, _ in data]
        targets = [tgt for _, tgt in data]

        # Validate target shape early.
        for tgt in targets:
            if tgt.ndim != 2 or tgt.shape[-1] != self.output_dim:
                raise ValueError(
                    f"Each target must have shape [num_variates, {self.output_dim}], got {tuple(tgt.shape)}"
                )

        dataset = _RegressionDataset(series, targets, context_length, prediction_length)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: self._collate(batch, context_length, prediction_length),
        )

        total_steps = max(1, len(loader) * epochs)
        optimizer = build_optimizer(
            self.head.parameters(),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = build_scheduler(optimizer, num_training_steps=total_steps, warmup_ratio=warmup_ratio)

        self.head.train()
        if not self.freeze_backbone:
            self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx, (features, tgt, mask) in enumerate(loader):
                optimizer.zero_grad()
                pred = self.head(features)  # [B, V_t, output_dim]
                loss = self._masked_mse(pred, tgt, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.head.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

                if (batch_idx + 1) % log_interval == 0:
                    print(
                        f"Epoch {epoch} batch {batch_idx + 1}/{len(loader)}: loss={loss.item():.6f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e}"
                    )

            print(f"Epoch {epoch}: avg_loss={total_loss / max(1, len(loader)):.6f}")

        self.head.eval()
        self.model.eval()

        if output_dir is not None:
            self.save_head(output_dir)

    def predict(self, timeseries: TimeseriesType | Sequence[TimeseriesType]) -> torch.Tensor | list[torch.Tensor]:
        """Return regression predictions for one or more time series.

        Returns a tensor of shape ``[V_t, output_dim]`` for a single series or a
        list of such tensors for a batch.
        """
        single = isinstance(timeseries, TimeseriesType)
        series = [timeseries] if single else list(timeseries)
        if not series:
            return [] if not single else torch.empty(0, self.output_dim)

        context_length = getattr(self.model, "context_len", 2048)
        prediction_length = getattr(self.model, "future_len", 512)

        # Build one batch via the standard collate path.
        input_batch, _ = collate_timeseries(
            series,
            self.model.postprocessor,
            prediction_length,
            device=self.device,
        )

        input_batch = pad_to_model_length(input_batch, self.model)
        self.head.eval()
        with torch.no_grad():
            features, target_mask = self._extract_features(input_batch, context_length)
            pred = self.head(features)  # [V_t_total, output_dim]

        # Split per series using target_mask group boundaries.
        result = self._split_per_series(pred, target_mask, input_batch, len(series))
        return result[0] if single else result

    def save_head(self, output_dir: str | Path) -> None:
        """Save the regression head checkpoint alongside the backbone layout."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "output_dim": self.output_dim,
                "hidden_dim": self.hidden_dim,
                "freeze_backbone": self.freeze_backbone,
                "state_dict": self.head.state_dict(),
            },
            output_dir / "regression_head.ckpt",
        )

    def load_head(self, output_dir: str | Path) -> None:
        """Load a previously saved regression head."""
        path = Path(output_dir) / "regression_head.ckpt"
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.output_dim = ckpt["output_dim"]
        self.hidden_dim = ckpt["hidden_dim"]
        self.freeze_backbone = ckpt["freeze_backbone"]
        self.head.load_state_dict(ckpt["state_dict"])

    def _collate(
        self,
        batch: list[tuple[TimeseriesType, torch.Tensor]],
        context_length: int,
        prediction_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collate a regression batch into features, targets, and masks."""
        samples, targets = zip(*batch)
        input_batch, _ = collate_timeseries(
            samples,
            self.model.postprocessor,
            prediction_length,
            device=self.device,
        )
        # Truncate/pad x to model length.
        input_batch = pad_to_model_length(input_batch, self.model)
        features, target_mask = self._extract_features(input_batch, context_length)

        # Stack targets and align variate count per sample.
        max_variates = max(t.shape[0] for t in targets)
        padded_targets = []
        masks = []
        for t in targets:
            pad_v = max_variates - t.shape[0]
            if pad_v > 0:
                padded = torch.nn.functional.pad(t, (0, 0, 0, pad_v), value=float("nan"))
                mask = torch.cat([torch.ones(t.shape[0], dtype=torch.bool), torch.zeros(pad_v, dtype=torch.bool)])
            else:
                padded = t
                mask = torch.ones(t.shape[0], dtype=torch.bool)
            padded_targets.append(padded)
            masks.append(mask)

        target_tensor = torch.stack(padded_targets, dim=0).to(self.device)
        mask_tensor = torch.stack(masks, dim=0).to(self.device)
        return features, target_tensor, mask_tensor

    def _extract_features(
        self,
        input_batch: dict[str, torch.Tensor],
        context_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract mean-pooled target features from the backbone.

        Returns
        -------
        features : torch.Tensor
            Per-target-variate features of shape ``[V_t_total, embedding_dim]``
            for the whole batch.
        target_mask : torch.Tensor
            Boolean mask of target rows in ``input_batch['x']``.
        """
        raw_model = self.model.model if isinstance(self.model, ForecastModel) else self.model
        features, _ = raw_model.forward_features(input_batch)
        target_mask = input_batch["target_mask"]
        target_features = features[target_mask]  # [V_t_total, L, D]

        # Mean-pool over the token dimension. Tokens corresponding to fully
        # NaN input windows are masked out so they do not contribute.
        token_mask = ~torch.isnan(target_features).all(dim=-1)  # [V_t_total, L]
        lengths = token_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)  # [V_t_total, 1]
        pooled = (target_features * token_mask.unsqueeze(-1).to(target_features.dtype)).sum(dim=1) / lengths
        return pooled, target_mask

    @staticmethod
    def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked MSE over valid variates."""
        diff = (pred - target) ** 2
        masked = diff * mask.unsqueeze(-1).to(diff.dtype)
        return masked.sum() / mask.sum().clamp_min(1.0)

    def _split_per_series(
        self,
        pred: torch.Tensor,
        target_mask: torch.Tensor,
        input_batch: dict[str, torch.Tensor],
        num_series: int,
    ) -> list[torch.Tensor]:
        """Split a flat target prediction tensor back into per-series tensors."""
        group_vector = input_batch.get("group_vector")
        # collate_timeseries assigns one group per series, so each series occupies
        # a contiguous block of target rows. We can split by counting target rows
        # per group in group_vector order.
        if group_vector is None:
            # No grouping information; assume equal split (best effort).
            chunk_size = pred.shape[0] // num_series
            return [pred[i * chunk_size : (i + 1) * chunk_size] for i in range(num_series)]

        # Map group ids to target row counts.
        groups = group_vector[target_mask].unique(sorted=True)
        result = []
        for g in groups:
            result.append(pred[group_vector[target_mask] == g])
        return result
