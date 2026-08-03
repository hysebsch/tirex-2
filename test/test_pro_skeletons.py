"""Tests for TiRex Pro skeleton modules (hardware, streaming, classification, regression)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tirex2 import TimeseriesType
from tirex2.pro.classification import TimeSeriesClassifier
from tirex2.pro.hardware import HardwareInfo, HardwareOptimizer, detect_hardware, print_hardware_report
from tirex2.pro.regression import TimeSeriesRegressor
from tirex2.pro.streaming import IncrementalForecaster


def test_detect_hardware_returns_info() -> None:
    info = detect_hardware()
    assert isinstance(info, HardwareInfo)
    assert isinstance(info.has_gpu, bool)
    assert isinstance(info.gpu_names, list)
    assert info.recommended_device in ("cpu", "cuda")


def test_detect_hardware_uses_cuda_home_env_var(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_HOME", "/fake/cuda")
    info = detect_hardware()
    assert info.cuda_home == "/fake/cuda"


def test_detect_hardware_falls_back_to_cuda_path(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setenv("CUDA_PATH", "/fake/cuda-path")
    info = detect_hardware()
    assert info.cuda_home == "/fake/cuda-path"


def test_print_hardware_report(capsys) -> None:
    info = HardwareInfo(
        platform="Linux",
        architecture="aarch64",
        has_gpu=True,
        gpu_names=["GB10"],
        driver_cuda_version="13.0",
        cuda_home="/usr/local/cuda",
        recommended_device="cuda",
        recommended_matmul_precision="high",
    )
    print_hardware_report(info)
    captured = capsys.readouterr().out
    assert "GB10" in captured
    assert "13.0" in captured
    assert "cuda" in captured


def test_hardware_optimizer_detect(build_small_model) -> None:
    model = build_small_model("cpu")
    optimizer = HardwareOptimizer(model)
    info = optimizer.detect()
    assert isinstance(info, HardwareInfo)
    assert optimizer._hardware is info


def test_hardware_optimizer_compile_not_implemented(build_small_model) -> None:
    model = build_small_model("cpu")
    optimizer = HardwareOptimizer(model)
    with pytest.raises(NotImplementedError, match="torch.compile"):
        optimizer.compile()


def test_hardware_optimizer_quantize_not_implemented(build_small_model) -> None:
    model = build_small_model("cpu")
    optimizer = HardwareOptimizer(model)
    with pytest.raises(NotImplementedError, match="Quantization"):
        optimizer.quantize()


def test_incremental_forecaster_rolling_window(build_small_model) -> None:
    model = build_small_model("cpu").eval()
    context_len = model.context_len
    pred_len = model.future_len

    forecaster = IncrementalForecaster(model, prediction_length=pred_len)
    assert forecaster.context_length == context_len

    # Seed with a context that exceeds the model limit.
    full_history = TimeseriesType(
        target=torch.randn(1, context_len + 20),
        past_covariates=None,
        future_covariates=torch.zeros(1, context_len + 20 + pred_len),
    )
    forecaster.update(full_history)
    assert forecaster._cached is not None
    assert forecaster._cached.target.shape[-1] == context_len
    assert forecaster._cached.future_covariates is not None

    forecast1 = forecaster.forecast()
    assert isinstance(forecast1, torch.Tensor)
    assert forecast1.shape == (1, model.num_quantiles, pred_len)

    # Add a single new observation.
    new_step = TimeseriesType(
        target=torch.randn(1, 1),
        past_covariates=None,
        future_covariates=torch.zeros(1, 1 + pred_len),
    )
    forecaster.update(new_step)
    assert forecaster._cached.target.shape[-1] == context_len
    assert forecaster._last_forecast is None  # invalidated by update

    forecast2 = forecaster.forecast()
    assert isinstance(forecast2, torch.Tensor)
    assert forecast2.shape == (1, model.num_quantiles, pred_len)

    forecaster.reset()
    assert forecaster._cached is None
    with pytest.raises(RuntimeError, match="No context has been ingested"):
        forecaster.forecast()


def test_incremental_forecaster_with_past_covariates(build_small_model) -> None:
    model = build_small_model("cpu").eval()
    context_len = model.context_len
    pred_len = model.future_len

    forecaster = IncrementalForecaster(model, prediction_length=pred_len, context_length=context_len)
    ts = TimeseriesType(
        target=torch.randn(1, context_len),
        past_covariates=torch.randn(2, context_len),
        future_covariates=None,
    )
    forecaster.update(ts)
    assert forecaster._cached.past_covariates is not None
    assert forecaster._cached.past_covariates.shape[-1] == context_len

    forecast = forecaster.forecast()
    assert isinstance(forecast, torch.Tensor)


def test_incremental_forecaster_invalid_prediction_length(build_small_model) -> None:
    with pytest.raises(ValueError, match="prediction_length must be >= 1"):
        IncrementalForecaster(build_small_model("cpu"), prediction_length=0)


def test_incremental_forecaster_update_requires_timeseries_type(build_small_model) -> None:
    forecaster = IncrementalForecaster(build_small_model("cpu"), prediction_length=4)
    with pytest.raises(TypeError, match="TimeseriesType"):
        forecaster.update(torch.randn(1, 8))


def test_time_series_classifier_not_implemented(build_small_model) -> None:
    model = build_small_model("cpu")
    classifier = TimeSeriesClassifier(model, num_classes=3)
    with pytest.raises(NotImplementedError, match="Classification training"):
        classifier.fit([])
    with pytest.raises(NotImplementedError, match="Classification prediction"):
        classifier.predict(None)


def test_time_series_regressor_smoke(build_small_model) -> None:
    model = build_small_model("cpu").eval()
    context_len = model.context_len
    pred_len = model.future_len

    regressor = TimeSeriesRegressor(model, output_dim=2, hidden_dim=32)
    # Backbone should be frozen by default.
    assert all(not p.requires_grad for p in regressor.model.parameters())
    assert any(p.requires_grad for p in regressor.head.parameters())

    # Build a few synthetic (series, target) pairs.
    data = []
    for i in range(4):
        target = torch.randn(1, context_len)
        future_cov = torch.zeros(1, context_len + pred_len)
        ts = TimeseriesType(target=target, past_covariates=None, future_covariates=future_cov)
        label = torch.tensor([[float(i), float(i) * 0.5]])  # [1, 2]
        data.append((ts, label))

    regressor.fit(data, epochs=2, batch_size=2, learning_rate=1e-3, context_length=context_len, log_interval=1)

    pred = regressor.predict(data[0][0])
    assert pred.shape == (1, 2)
    assert torch.isfinite(pred).all()


def test_time_series_regressor_predict_batch(build_small_model) -> None:
    model = build_small_model("cpu").eval()
    context_len = model.context_len
    pred_len = model.future_len

    regressor = TimeSeriesRegressor(model, output_dim=1, hidden_dim=16)
    series = [
        TimeseriesType(
            target=torch.randn(1, context_len),
            past_covariates=None,
            future_covariates=torch.zeros(1, context_len + pred_len),
        )
        for _ in range(3)
    ]
    labels = [torch.tensor([[1.0]]), torch.tensor([[2.0]]), torch.tensor([[3.0]])]
    regressor.fit(list(zip(series, labels)), epochs=1, batch_size=2, context_length=context_len)

    preds = regressor.predict(series)
    assert len(preds) == 3
    for p in preds:
        assert p.shape == (1, 1)
        assert torch.isfinite(p).all()


def test_time_series_regressor_save_load_head(build_small_model, tmp_path) -> None:
    model = build_small_model("cpu").eval()
    context_len = model.context_len
    pred_len = model.future_len

    regressor = TimeSeriesRegressor(model, output_dim=2, hidden_dim=16)
    ts = TimeseriesType(
        target=torch.randn(1, context_len),
        past_covariates=None,
        future_covariates=torch.zeros(1, context_len + pred_len),
    )
    regressor.fit([(ts, torch.tensor([[1.0, 2.0]]))], epochs=1, context_length=context_len)
    pred_before = regressor.predict(ts)

    regressor.save_head(tmp_path)
    reloaded = TimeSeriesRegressor(model, output_dim=2, hidden_dim=16)
    reloaded.load_head(tmp_path)
    pred_after = reloaded.predict(ts)
    assert torch.allclose(pred_before, pred_after, atol=1e-6)
