"""Lightweight fine-tuning API for TiRex-2.

Supports full fine-tuning, head-only adaptation, selective block training, and
LoRA. Checkpoints are saved in the same layout ``tirex2.load_model`` expects.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ...model.types import TimeseriesType
from ..training.dataset import TiRexDataset, collate_timeseries, pad_to_model_length
from ..training.lora import inject_lora
from ..training.loss import PinballLoss
from ..training.metrics import MASE, MetricsTracker, QuantileCRPS
from ..training.utils import (
    build_optimizer,
    build_scheduler,
    count_parameters,
    freeze_backbone,
    save_checkpoint,
)

logger = logging.getLogger(__name__)


class FineTuner:
    """Fine-tune a TiRex-2 checkpoint on custom data.

    Parameters
    ----------
    model : torch.nn.Module
        An instantiated :class:`tirex2.TiRex2` (or a :class:`tirex2.ForecastModel`
        wrapper; the underlying model is extracted automatically).
    strategy : {"full", "head-only", "blocks", "lora"}
        Which parameters to train.
    strategy_kwargs : dict, optional
        Extra arguments for the strategy. For ``"blocks"`` use
        ``{"blocks_to_train": [10, 11]}``. For ``"lora"`` use
        ``{"rank": 8, "alpha": 8.0}``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        strategy: str = "head-only",
        strategy_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.strategy = strategy
        self.strategy_kwargs = strategy_kwargs or {}

        # Unwrap ForecastModel delegate if needed.
        if hasattr(model, "model"):
            model = getattr(model, "model")
        self.model = model

        self._model_max_length = getattr(self.model, "context_len", 2048) + getattr(self.model, "future_len", 512)
        self._prediction_length = getattr(self.model, "future_len", 512)
        self._device = next(self.model.parameters()).device

        if strategy == "lora":
            rank = self.strategy_kwargs.get("rank", 8)
            alpha = self.strategy_kwargs.get("alpha", float(rank))
            inject_lora(self.model, rank=rank, alpha=alpha)
        else:
            freeze_backbone(self.model, strategy, self.strategy_kwargs)

        self._trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        logger.info(
            "Fine-tuning strategy=%s, trainable parameters=%d / %d",
            strategy,
            sum(p.numel() for p in self._trainable_params),
            count_parameters(self.model),
        )

        self.loss_fn = PinballLoss(self.model.quantiles)
        self.mase_fn = MASE()
        self.crps_fn = QuantileCRPS(self.model.quantiles)
        self.metrics_tracker = MetricsTracker(self.model.quantiles)
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        self.best_val_loss = float("inf")
        self.best_state_dict: dict[str, torch.Tensor] | None = None
        self.current_epoch = 0
        self.best_metrics: dict[str, float] = {}

    def fit(
        self,
        train_data: Sequence[TimeseriesType] | TiRexDataset,
        val_data: Sequence[TimeseriesType] | TiRexDataset | None = None,
        *,
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        warmup_ratio: float = 0.1,
        min_lr_ratio: float = 0.0,
        early_stopping_patience: int | None = None,
        output_dir: str | Path | None = None,
        log_interval: int = 10,
        gradient_accumulation_steps: int = 1,
        context_length: int | None = None,
        prediction_length: int | None = None,
        resume_from: str | Path | None = None,
    ) -> None:
        """Run the fine-tuning loop.

        Parameters
        ----------
        train_data
            Training series or an already-built dataset.
        val_data
            Optional validation series or dataset.
        epochs
            Number of training epochs.
        batch_size
            Number of samples per optimization step. With variable-length
            series the postprocessor packs them into one group-vector batch.
        learning_rate
            Peak learning rate for AdamW.
        weight_decay
            AdamW weight decay.
        grad_clip
            Max gradient norm for clipping.
        warmup_ratio
            Fraction of total steps spent in linear LR warmup.
        min_lr_ratio
            Final LR as a fraction of the peak.
        early_stopping_patience
            Stop training if validation loss does not improve for this many
            epochs. Requires ``val_data``.
        output_dir
            Directory to save per-epoch and best checkpoints.
        log_interval
            Log training loss every N batches.
        gradient_accumulation_steps
            Accumulate gradients over this many batches before an optimizer step.
        context_length
            If ``train_data`` is a list of series, the sliding-window context length.
        prediction_length
            If ``train_data`` is a list of series, the forecast horizon.
        resume_from
            Directory of a previous checkpoint to resume training from. Loads
            model weights, optimizer state, scheduler state, and the last epoch.
        """
        train_dataset = self._ensure_dataset(
            train_data, context_length=context_length, prediction_length=prediction_length
        )
        val_dataset = (
            self._ensure_dataset(val_data, context_length=context_length, prediction_length=prediction_length)
            if val_data is not None
            else None
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda samples: collate_timeseries(
                samples, self.model.postprocessor, self._prediction_length, device=self._device
            ),
            drop_last=False,
        )

        total_steps = (len(train_loader) // gradient_accumulation_steps) * epochs
        self.optimizer = build_optimizer(
            self._trainable_params,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            num_training_steps=max(1, total_steps),
            warmup_ratio=warmup_ratio,
            min_lr_ratio=min_lr_ratio,
        )

        start_epoch = 0
        resume_dir = resume_from if resume_from is None else Path(resume_from)
        if resume_dir is not None:
            from ..training.utils import load_training_state

            training_state = load_training_state(resume_dir)
            if "optimizer_state_dict" in training_state:
                self.optimizer.load_state_dict(training_state["optimizer_state_dict"])
            if "scheduler_state_dict" in training_state:
                self.scheduler.load_state_dict(training_state["scheduler_state_dict"])
            start_epoch = int(training_state.get("epoch", -1)) + 1
            logger.info("Resumed training from epoch %d", start_epoch)

        patience_counter = 0
        self.model.train()

        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            train_loss = self._train_epoch(
                train_loader,
                grad_clip=grad_clip,
                log_interval=log_interval,
                accumulation_steps=gradient_accumulation_steps,
            )
            logger.info("Epoch %d: train_loss=%.6f", epoch, train_loss)

            if val_dataset is not None:
                val_loss, val_metrics = self._eval_dataset(val_dataset, batch_size=batch_size)
                metric_msg = ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                logger.info("Epoch %d: val_loss=%.6f, %s", epoch, val_loss, metric_msg)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_metrics = dict(val_metrics)
                    self.best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                    logger.info("Early stopping triggered at epoch %d", epoch)
                    break

            if output_dir is not None:
                epoch_dir = Path(output_dir) / f"epoch-{epoch}"
                save_checkpoint(
                    self.model,
                    epoch_dir,
                    extra_config={"epoch": epoch},
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                )

        # Restore best validation weights.
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        if output_dir is not None:
            save_checkpoint(self.model, Path(output_dir) / "best")

    def save(self, path: str | Path) -> None:
        """Save the fine-tuned model to ``path`` as a ``load_model``-compatible checkpoint."""
        save_checkpoint(self.model, path)

    def _ensure_dataset(
        self,
        data: Sequence[TimeseriesType] | TiRexDataset,
        context_length: int | None,
        prediction_length: int | None,
    ) -> TiRexDataset:
        if isinstance(data, TiRexDataset):
            return data
        pred = prediction_length if prediction_length is not None else self._prediction_length
        model_context_len = getattr(self.model, "context_len", 2048)
        if context_length is not None:
            ctx = context_length
        else:
            # Use the model's maximum context length if the series are long enough;
            # otherwise shrink to the largest context that still yields windows.
            max_target_len = max((ts.target.shape[-1] for ts in data), default=0)
            ctx = min(model_context_len, max(1, max_target_len - pred))
        return TiRexDataset(data, context_length=ctx, prediction_length=pred)

    def _train_epoch(
        self,
        loader: DataLoader,
        *,
        grad_clip: float,
        log_interval: int,
        accumulation_steps: int,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad()

        for batch_idx, (input_batch, targets) in enumerate(loader):
            loss = self._forward_loss(input_batch, targets)
            loss = loss / accumulation_steps
            loss.backward()

            total_loss += loss.item() * accumulation_steps
            num_batches += 1

            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(loader) - 1:
                torch.nn.utils.clip_grad_norm_(self._trainable_params, grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            if (batch_idx + 1) % log_interval == 0:
                logger.info(
                    "Epoch %d batch %d/%d: loss=%.6f lr=%.2e",
                    self.current_epoch,
                    batch_idx + 1,
                    len(loader),
                    loss.item() * accumulation_steps,
                    self.optimizer.param_groups[0]["lr"],
                )

        return total_loss / max(1, num_batches)

    def _eval_dataset(
        self,
        dataset: TiRexDataset,
        batch_size: int,
    ) -> tuple[float, dict[str, float]]:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda samples: collate_timeseries(
                samples, self.model.postprocessor, self._prediction_length, device=self._device
            ),
        )
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        self.metrics_tracker.reset()
        with torch.no_grad():
            for input_batch, targets in loader:
                loss, pred_targets, y_true, observed, context = self._forward_loss(
                    input_batch, targets, return_context=True
                )
                total_loss += loss.item()
                num_batches += 1

                # Median prediction for MASE.
                q = pred_targets.shape[-2]
                median_pred = pred_targets[..., q // 2, :]
                self.metrics_tracker.update(median_pred, y_true, context=context, mask=observed)
                self.metrics_tracker.update_crps(pred_targets, y_true, mask=observed)

        self.model.train()
        avg_loss = total_loss / max(1, num_batches)
        metrics = self.metrics_tracker.average()
        return avg_loss, metrics

    def _forward_loss(
        self,
        input_batch: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        return_context: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run a forward pass and return the pinball loss on target variates.

        If ``return_context`` is True, also returns the predicted target
        quantiles, ground truth, observed mask, and the context portion of the
        target for metric computation.
        """
        input_batch = pad_to_model_length(input_batch, self.model)
        x = input_batch["x"]
        pred = self.model(input_batch)  # [V_total, Q, T]

        target_mask = input_batch["target_mask"]
        pred_targets = pred[target_mask]  # [V_t, Q, T]

        y_true = targets["target"].to(pred.device)
        observed = targets["observed"].to(pred.device)

        # Align time dimension: if the model padded the input, predictions may be longer.
        pred_len = pred_targets.shape[-1]
        if pred_len > y_true.shape[-1]:
            pred_targets = pred_targets[..., -y_true.shape[-1] :]
        elif pred_len < y_true.shape[-1]:
            pad = y_true.shape[-1] - pred_len
            pred_targets = F.pad(pred_targets, (pad, 0), value=float("nan"))

        loss = self.loss_fn(pred_targets, y_true, mask=observed)
        if not return_context:
            return loss

        # Recover the context portion of the full-window target from the input.
        # The postprocessor packs target rows first, then covariates; their
        # positions in ``x`` correspond to the target_mask True entries.
        context_len = self._prediction_length if y_true.shape[-1] == self._prediction_length else y_true.shape[-1] - self._prediction_length
        context = y_true[..., :context_len]
        return loss, pred_targets, y_true, observed, context
