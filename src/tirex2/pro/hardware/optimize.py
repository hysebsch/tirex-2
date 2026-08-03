"""Hardware-optimized inference helpers.

This module will host compilation / quantization / kernel-selection helpers
for edge, embedded, and server deployments such as DGX Spark (Blackwell) and
NVIDIA Jetson devices.
"""

from __future__ import annotations

from typing import Any

from .detect import HardwareInfo, detect_hardware


class HardwareOptimizer:
    """Apply backend-specific optimizations to a TiRex-2 model.

    Parameters
    ----------
    model
        An instantiated :class:`tirex2.TiRex2`.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self._hardware: HardwareInfo | None = None

    def detect(self) -> HardwareInfo:
        """Refresh the cached hardware report."""
        self._hardware = detect_hardware()
        return self._hardware

    def compile(self, mode: str | None = None) -> Any:
        """Apply ``torch.compile`` with a backend suitable for the current hardware."""
        raise NotImplementedError("torch.compile optimization is not implemented yet.")

    def quantize(self, dtype: str = "int8") -> Any:
        """Return a quantized version of the model for the target dtype."""
        raise NotImplementedError("Quantization is not implemented yet.")
