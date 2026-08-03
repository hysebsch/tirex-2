"""Tests for the TiRex-2 training / fine-tuning package."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from tirex2 import TimeseriesType, load_model
from tirex2.model.types import TimeseriesType as TimeseriesTypeAlias
from tirex2.pro.finetuning import FineTuner
from tirex2.pro.training import (
    GaussianNoise,
    MaskedMAELoss,
    PinballLoss,
    SyntheticCouplingPipeline,
    TiRexDataset,
    build_optimizer,
    build_scheduler,
    save_checkpoint,
)


def _small_series(num: int = 2, length: int = 64, num_targets: int = 1) -> list[TimeseriesTypeAlias]:
    series = []
    for i in range(num):
        target = torch.sin(torch.linspace(0, 4 * 3.14159 * (i + 1), length)) + torch.randn(length) * 0.1
        target = target.unsqueeze(0)  # [V_t, T]
        future_cov = torch.zeros(2, length)
        future_cov[0, length // 2 :] = 1.0
        series.append(
            TimeseriesType(
                target=target,
                past_covariates=None,
                future_covariates=future_cov,
            )
        )
    return series


def test_pinball_loss_zero_when_all_quantiles_match():
    quantiles = torch.tensor([0.1, 0.5, 0.9])
    loss_fn = PinballLoss(quantiles)

    pred = torch.ones(2, 3, 8)  # all quantiles predict the true value
    target = torch.ones(2, 8)

    loss = loss_fn(pred, target)
    assert loss.shape == ()
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_pinball_loss_masks_nan():
    quantiles = torch.tensor([0.5])
    loss_fn = PinballLoss(quantiles, reduction="sum")

    pred = torch.zeros(1, 1, 4)
    target = torch.tensor([[1.0, float("nan"), 3.0, float("nan")]])

    loss = loss_fn(pred, target)
    # Only positions 0 and 2 are observed: 0.5*|1| + 0.5*|3| = 2.0.
    assert loss.item() == pytest.approx(2.0, abs=1e-6)


def test_masked_mae_loss():
    pred = torch.zeros(2, 3, 8)
    pred[:, 1, :] = 2.0
    target = torch.ones(2, 8) * 2.0
    loss_fn = MaskedMAELoss()
    assert loss_fn(pred, target).item() == pytest.approx(0.0, abs=1e-6)


def test_dataset_windowing():
    series = _small_series(num=1, length=128)
    ds = TiRexDataset(series, context_length=32, prediction_length=16, stride=8)
    assert len(ds) == (128 - 48) // 8 + 1
    sample = ds[0]
    assert sample.target.shape == (1, 48)
    assert sample.future_covariates is not None
    assert sample.future_covariates.shape == (2, 48)


def test_dataset_future_target_masked_when_requested():
    series = _small_series(num=1, length=64)
    ds = TiRexDataset(series, context_length=32, prediction_length=16, pad_future_target=False)
    sample = ds[0]
    assert torch.isnan(sample.target[:, -16:]).all()
    assert torch.isfinite(sample.target[:, :32]).all()


def test_synthetic_coupling_pipeline():
    pool = [torch.randn(128) for _ in range(8)]
    pipeline = SyntheticCouplingPipeline(
        mechanisms=["identity", "functional", "linear_mixing"],
        num_variates=3,
        window_length=64,
    )
    samples = pipeline.generate(pool, n_samples=4)
    assert len(samples) == 4
    assert samples[0]["target"].shape[0] == 1
    assert samples[0]["target"].shape[-1] == 64


def test_gaussian_noise_preserves_shape():
    x = torch.randn(1, 32)
    aug = GaussianNoise(std_range=(0.01, 0.02))
    out = aug(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_optimizer_and_scheduler():
    param = torch.nn.Parameter(torch.randn(2, 2))
    opt = build_optimizer([param], learning_rate=1e-3, weight_decay=0.01)
    sched = build_scheduler(opt, num_training_steps=100, warmup_ratio=0.1)
    # At step 0 the warmup schedule returns 0; after the first step lr is positive.
    sched.step()
    assert opt.param_groups[0]["lr"] > 0.0
    initial_lr = opt.param_groups[0]["lr"]
    # Warmup should increase LR toward the peak.
    for _ in range(5):
        sched.step()
    assert opt.param_groups[0]["lr"] > initial_lr


def test_finetuner_head_only_smoke(build_small_model):
    model = build_small_model("cpu")
    series = _small_series(num=4, length=64, num_targets=1)

    fine_tuner = FineTuner(model, strategy="head-only")
    with tempfile.TemporaryDirectory() as tmp:
        fine_tuner.fit(
            series,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            context_length=16,
            prediction_length=8,
            output_dir=tmp,
            log_interval=1,
        )
        out_dir = Path(tmp) / "final"
        fine_tuner.save(out_dir)
        assert (out_dir / "model-config.yaml").is_file()
        assert (out_dir / "model.ckpt").is_file()


def test_finetuner_checkpoint_reloadable(build_small_model):
    model = build_small_model("cpu")
    series = _small_series(num=2, length=64, num_targets=1)

    fine_tuner = FineTuner(model, strategy="full")
    with tempfile.TemporaryDirectory() as tmp:
        fine_tuner.fit(
            series,
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            context_length=32,
            prediction_length=8,
            output_dir=tmp,
        )
        out_dir = Path(tmp) / "final"
        fine_tuner.save(out_dir)
        # Reload and ensure it can forecast with a context-only input.
        reloaded = load_model(out_dir, device="cpu")
        ctx_len = 32
        pred_len = 8
        ts = TimeseriesTypeAlias(
            target=series[0].target[..., :ctx_len],
            past_covariates=None,
            future_covariates=series[0].future_covariates[..., : ctx_len + pred_len],
        )
        forecast = reloaded.forecast([ts], prediction_length=pred_len, output_type="torch", batch_size=1)
        assert forecast[0].shape[0] == ts.target.shape[0]


def test_save_checkpoint_roundtrip(build_small_model):
    model = build_small_model("cpu")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "ckpt"
        save_checkpoint(model, out, extra_config={"version": "test"})
        assert (out / "model-config.yaml").is_file()
        assert (out / "model.ckpt").is_file()
        reloaded = load_model(out, device="cpu")
        assert reloaded is not None
