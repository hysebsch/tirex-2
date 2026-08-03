"""Detect the available hardware and recommend runtime settings."""

from __future__ import annotations

import dataclasses
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class HardwareInfo:
    """Summary of the inference hardware environment."""

    platform: str
    architecture: str
    has_gpu: bool
    gpu_names: list[str]
    driver_cuda_version: str | None
    recommended_device: str
    recommended_matmul_precision: str | None


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def detect_hardware() -> HardwareInfo:
    """Inspect the host and return a :class:`HardwareInfo` record."""
    has_nvidia_smi = shutil.which("nvidia-smi") is not None
    gpu_names: list[str] = []
    driver_cuda: str | None = None

    if has_nvidia_smi:
        out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        if out:
            gpu_names = [line.strip() for line in out.splitlines() if line.strip()]
        full = _run(["nvidia-smi"])
        if full:
            for line in full.splitlines():
                if "CUDA Version:" in line:
                    driver_cuda = line.split("CUDA Version:")[-1].strip()
                    break

    has_gpu = bool(gpu_names)
    recommended_device = "cuda" if has_gpu else "cpu"
    recommended_matmul = "high" if has_gpu else None

    return HardwareInfo(
        platform=platform.system(),
        architecture=platform.machine(),
        has_gpu=has_gpu,
        gpu_names=gpu_names,
        driver_cuda_version=driver_cuda,
        recommended_device=recommended_device,
        recommended_matmul_precision=recommended_matmul,
    )


def print_hardware_report(info: HardwareInfo | None = None) -> None:
    """Print a human-readable summary of ``info`` (defaults to ``detect_hardware``)."""
    info = info or detect_hardware()
    print(f"Platform:        {info.platform} ({info.architecture})")
    print(f"GPU(s):          {', '.join(info.gpu_names) if info.gpu_names else 'none'}")
    print(f"Driver CUDA:     {info.driver_cuda_version or 'unknown'}")
    print(f"Recommended:     device={info.recommended_device}, matmul={info.recommended_matmul_precision}")
