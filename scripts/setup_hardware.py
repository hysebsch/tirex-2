#!/usr/bin/env python3
"""Detect the local NVIDIA/CUDA situation and prepare a CUDA 12.8 toolchain.

This script is intended for the DGX Spark and similar machines that ship a
CUDA 13.0 driver but still need a CUDA 12.8 toolkit to build xlstm/FlashRNN's
JIT CUDA extensions against a PyTorch ``cu128`` wheel.

CUDA drivers are backward compatible: a CUDA 13.0 host driver can run binaries
built with the CUDA 12.8 toolkit. The important constraint is that the CUDA
toolkit used to compile the extensions matches the CUDA version PyTorch was
built with.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Persist the CUDA toolkit outside the project tree so ``make clean`` does not
# delete a multi-gigabyte download.
CACHE_DIR = Path.home() / ".cache" / "tirex2" / "cuda"
CUDA_12_8_RUNFILE = "cuda_12.8.0_570.86.10_linux_sbsa.run"
CUDA_12_8_URL = (
    "https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/"
    + CUDA_12_8_RUNFILE
)


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def detect_driver_cuda_version() -> str | None:
    """Return the driver-level CUDA version reported by nvidia-smi (e.g. '13.0')."""
    out = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if not out:
        return None
    # nvidia-smi also prints the driver CUDA version in the top header, which is easier.
    full = _run(["nvidia-smi"])
    if not full:
        return None
    for line in full.splitlines():
        if "CUDA Version:" in line:
            return line.split("CUDA Version:")[-1].strip()
    return None


def detect_gpus() -> list[str]:
    """Return a list of GPU names reported by nvidia-smi."""
    out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if not out:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def detect_toolkit_cuda_version(cuda_home: Path) -> str | None:
    """Return the CUDA toolkit version under ``cuda_home`` using nvcc --version."""
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.exists():
        return None
    out = _run([str(nvcc), "--version"])
    if not out:
        return None
    for line in out.splitlines():
        if "release" in line:
            parts = line.split("release")
            if len(parts) > 1:
                return parts[-1].split(",")[0].strip()
    return None


def find_cuda_12_8_toolkit() -> Path | None:
    """Look for an existing CUDA 12.8 toolkit, preferring project-local installs."""
    candidates = [
        CACHE_DIR / "12.8",
        Path("/usr/local/cuda-12.8"),
    ]
    for candidate in candidates:
        if (candidate / "bin" / "nvcc").exists():
            return candidate
    return None


def download_cuda_12_8_toolkit() -> Path:
    """Download and extract the CUDA 12.8 runfile for aarch64 into the project cache."""
    dest = CACHE_DIR / "12.8"
    runfile = CACHE_DIR / CUDA_12_8_RUNFILE
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if dest.exists() and (dest / "bin" / "nvcc").exists():
        print(f"CUDA 12.8 toolkit already present at {dest}", file=sys.stderr)
        return dest

    if not runfile.exists():
        print(f"Downloading CUDA 12.8 runfile from {CUDA_12_8_URL}...", file=sys.stderr)
        try:
            urllib.request.urlretrieve(CUDA_12_8_URL, runfile)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to download CUDA 12.8 runfile: {exc}", file=sys.stderr)
            sys.exit(1)

    print(f"Extracting CUDA 12.8 toolkit to {dest}...", file=sys.stderr)
    subprocess.check_call(
        [str(runfile), "--silent", "--toolkit", f"--toolkitpath={dest}", "--override"],
        cwd=str(CACHE_DIR),
    )
    return dest


def recommend_arch_list() -> str:
    """Return a sensible TORCH_CUDA_ARCH_LIST for current GPUs."""
    gpus = detect_gpus()
    # DGX Spark / GB10 is Blackwell (sm_100/sm_121). Also keep common server GPUs.
    # If we cannot detect GPUs, use a broad list.
    if any("GB10" in g or "Blackwell" in g for g in gpus):
        return "8.0;8.6;9.0;10.0;12.1"
    if gpus:
        return "8.0;8.6;9.0"
    return "8.0;8.6;9.0;10.0"


def emit_env_file(path: Path, cuda_home: Path) -> None:
    """Write a shell snippet that exports the required build/run environment."""
    arch_list = recommend_arch_list()
    lines = [
        f"export CUDA_HOME={cuda_home}",
        f"export PATH={cuda_home / 'bin'}:$PATH",
        f"export LD_LIBRARY_PATH={cuda_home / 'lib64'}:$LD_LIBRARY_PATH",
        f"export TORCH_CUDA_ARCH_LIST='{arch_list}'",
        # Force PyTorch to use the matching CUDA libraries when building extensions.
        f"export NVCC={cuda_home / 'bin' / 'nvcc'}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote CUDA environment to {path}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect and set up CUDA for TiRex-2")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and extract a project-local CUDA 12.8 toolkit if needed.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=CACHE_DIR / "env.sh",
        help="Path to write a shell env file that can be sourced by Make/scripts.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only print the detection report and exit.",
    )
    args = parser.parse_args()

    arch = platform.machine()
    driver_cuda = detect_driver_cuda_version()
    gpus = detect_gpus()
    cuda_home = find_cuda_12_8_toolkit()
    toolkit_cuda = detect_toolkit_cuda_version(cuda_home) if cuda_home else None

    print("Hardware / driver report", file=sys.stderr)
    print(f"  Platform:      {platform.system()} {arch}", file=sys.stderr)
    print(f"  GPUs:          {gpus or 'none detected'}", file=sys.stderr)
    print(f"  Driver CUDA:   {driver_cuda or 'unknown'}", file=sys.stderr)
    print(f"  CUDA 12.8 kit: {cuda_home or 'not found'}", file=sys.stderr)
    print(f"  Toolkit CUDA:  {toolkit_cuda or 'unknown'}", file=sys.stderr)

    if args.check:
        return 0

    if not gpus and arch != "aarch64":
        print("No NVIDIA GPU detected; defaulting to CPU-only path.", file=sys.stderr)
        return 0

    if driver_cuda and driver_cuda.startswith("13"):
        print(
            "CUDA 13.0 driver detected. xlstm/FlashRNN are currently tested against "
            "CUDA 12.8, so we will use a CUDA 12.8 toolkit + torch cu128 wheel. "
            "The CUDA 13.0 driver is backward compatible with CUDA 12.8 binaries.",
            file=sys.stderr,
        )
        if not cuda_home:
            if args.download:
                cuda_home = download_cuda_12_8_toolkit()
            else:
                print(
                    "No CUDA 12.8 toolkit found. Run with --download to fetch one to "
                    f"{CACHE_DIR / '12.8'}, or install cuda-toolkit-12-8 system-wide.",
                    file=sys.stderr,
                )
                return 1

    if not cuda_home:
        print(
            "No CUDA 12.8 toolkit found. Attempting to locate any CUDA toolkit...",
            file=sys.stderr,
        )
        for candidate in [Path("/usr/local/cuda")]:
            if (candidate / "bin" / "nvcc").exists():
                cuda_home = candidate
                break

    if cuda_home:
        emit_env_file(args.env_file, cuda_home)
        print("\nTo apply these settings in your shell, run:", file=sys.stderr)
        print(f"  source {args.env_file}", file=sys.stderr)
        return 0

    print("Could not determine a usable CUDA_HOME.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
