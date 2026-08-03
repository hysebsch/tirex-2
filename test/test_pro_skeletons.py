"""Tests for TiRex Pro skeleton modules (hardware, streaming, classification, regression)."""

from __future__ import annotations

import pytest
import torch

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


def test_incremental_forecaster_not_implemented(build_small_model) -> None:
    model = build_small_model("cpu")
    forecaster = IncrementalForecaster(model, prediction_length=4)
    from tirex2 import TimeseriesType
    ts = TimeseriesType(target=torch.randn(1, 16), past_covariates=None, future_covariates=None)
    with pytest.raises(NotImplementedError, match="Streaming update"):
        forecaster.update(ts)
    with pytest.raises(NotImplementedError, match="Streaming forecast"):
        forecaster.forecast()
    forecaster.reset()
    assert forecaster._state is None


def test_time_series_classifier_not_implemented(build_small_model) -> None:
    model = build_small_model("cpu")
    classifier = TimeSeriesClassifier(model, num_classes=3)
    with pytest.raises(NotImplementedError, match="Classification training"):
        classifier.fit([])
    with pytest.raises(NotImplementedError, match="Classification prediction"):
        classifier.predict(None)


def test_time_series_regressor_not_implemented(build_small_model) -> None:
    model = build_small_model("cpu")
    regressor = TimeSeriesRegressor(model, output_dim=2)
    with pytest.raises(NotImplementedError, match="Regression training"):
        regressor.fit([])
    with pytest.raises(NotImplementedError, match="Regression prediction"):
        regressor.predict(None)
