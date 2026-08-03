"""Data augmentation and the synthetic multivariate coupling pipeline from TiRex-2.

The pipeline is described in Section 3.4 of the paper. It takes a batch of
univariate series, augments each independently, and then couples them through
one of several mechanisms (identity, functional, linear mixing, cointegration,
linear/nonlinear SCM). Post-processing adds realistic covariate structure such
as time warping, discretization, and future masking.

All transforms operate on 1D tensors (single variate) and are applied inside
``TiRexDataset`` or as a preprocessing step before training.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Per-series augmentations
# ---------------------------------------------------------------------------


class PerSeriesAugment(ABC):
    """Base class for independent per-series augmentations."""

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the augmentation to a 1D tensor."""


class AmplitudeTrend(PerSeriesAugment):
    """Piecewise-linear amplitude scaling and bias trend."""

    def __init__(
        self,
        num_segments: int = 3,
        scale_range: tuple[float, float] = (0.8, 1.2),
        bias_range: tuple[float, float] = (-0.3, 0.3),
    ) -> None:
        self.num_segments = max(1, num_segments)
        self.scale_range = scale_range
        self.bias_range = bias_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        length = x.shape[-1]
        # Random knots.
        knots = sorted(np.random.choice(length, size=self.num_segments - 1, replace=False).tolist())
        knots = [0] + knots + [length]
        out = x.clone()
        valid = out[~torch.isnan(out)]
        series_std = valid.std().item() if valid.numel() > 1 else 1.0
        for i in range(len(knots) - 1):
            lo, hi = knots[i], knots[i + 1]
            scale = float(np.random.uniform(*self.scale_range))
            bias = float(np.random.uniform(*self.bias_range)) * series_std
            segment = out[..., lo:hi]
            out[..., lo:hi] = torch.where(torch.isnan(segment), segment, segment * scale + bias)
        return out


class QuantileCensor(PerSeriesAugment):
    """Clip extreme values to a sampled quantile range."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99) -> None:
        self.lower = lower
        self.upper = upper

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        valid = x[~torch.isnan(x)]
        if valid.numel() == 0:
            return x
        lo = torch.quantile(valid, float(np.random.uniform(0.0, self.lower)))
        hi = torch.quantile(valid, float(np.random.uniform(self.upper, 1.0)))
        return torch.clamp(x, min=lo, max=hi)


class GaussianNoise(PerSeriesAugment):
    """Add Gaussian noise scaled by the series standard deviation."""

    def __init__(self, std_range: tuple[float, float] = (0.01, 0.05)) -> None:
        self.std_range = std_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        valid = x[~torch.isnan(x)]
        std = valid.std().item() if valid.numel() > 1 else 0.0
        scale = float(np.random.uniform(*self.std_range)) * max(std, 1e-6)
        noise = torch.randn_like(x) * scale
        return torch.where(torch.isnan(x), x, x + noise)


class SpikeInjection(PerSeriesAugment):
    """Inject sparse synthetic spikes with random kernel shapes."""

    def __init__(
        self,
        prob: float = 0.05,
        scale_range: tuple[float, float] = (2.0, 5.0),
        kernel: str = "gaussian",
    ) -> None:
        self.prob = prob
        self.scale_range = scale_range
        if kernel not in {"gaussian", "triangular", "rectangular"}:
            raise ValueError(f"Unknown spike kernel {kernel!r}")
        self.kernel = kernel

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        length = x.shape[-1]
        mask = torch.rand(length) < self.prob
        if not mask.any():
            return x
        valid = x[~torch.isnan(x)]
        series_std = valid.std().item() if valid.numel() > 1 else 1.0
        scale = float(np.random.uniform(*self.scale_range)) * series_std
        spikes = torch.zeros_like(x)
        center = torch.arange(length, dtype=torch.float32)
        for t in torch.where(mask)[0]:
            width = int(np.random.randint(1, 4))
            window = (center - t.item()).abs() <= width
            if self.kernel == "gaussian":
                weights = torch.exp(-((center - t.item()) ** 2) / (2 * width))
            elif self.kernel == "triangular":
                weights = torch.clamp(1.0 - (center - t.item()).abs() / (width + 1), min=0.0)
            else:  # rectangular
                weights = window.float()
            sign = 1 if np.random.rand() > 0.5 else -1
            spikes = spikes + sign * scale * weights * window.float()
        return x + spikes


class SmoothTimeWarp(PerSeriesAugment):
    """Apply a smooth Brownian-bridge-like time warp via interpolation."""

    def __init__(self, strength: float = 0.05, num_knots: int = 5) -> None:
        self.strength = strength
        self.num_knots = max(2, num_knots)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        length = x.shape[-1]
        original_indices = torch.arange(length, dtype=torch.float32)
        # Brownian bridge: zero at endpoints, random intermediate offsets.
        knots = torch.linspace(0, length - 1, self.num_knots)
        offsets = torch.randn(self.num_knots) * length * self.strength
        offsets[0] = 0.0
        offsets[-1] = 0.0
        # Linearly interpolate offsets to full length.
        warped_indices = original_indices + torch.nn.functional.interpolate(
            offsets.view(1, 1, -1),
            size=length,
            mode="linear",
            align_corners=True,
        ).view(-1)
        warped_indices = torch.clamp(warped_indices, 0.0, length - 1)
        # Gather via linear interpolation.
        idx_lo = warped_indices.long()
        idx_hi = torch.clamp(idx_lo + 1, max=length - 1)
        weight = (warped_indices - idx_lo.float()).to(x.dtype)
        return (1 - weight) * x[..., idx_lo] + weight * x[..., idx_hi]


class RandomTimeCrop(PerSeriesAugment):
    """Crop a contiguous subsequence to the requested length."""

    def __init__(self, length: int) -> None:
        self.length = length

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        total = x.shape[-1]
        if total <= self.length:
            # Pad on the right with NaN.
            pad = self.length - total
            return torch.nn.functional.pad(x, (0, pad), value=float("nan"))
        start = int(np.random.randint(0, total - self.length + 1))
        return x[..., start : start + self.length]


# ---------------------------------------------------------------------------
# Coupling mechanisms
# ---------------------------------------------------------------------------


class CouplingMechanism(ABC):
    """Base class for multivariate coupling mechanisms."""

    @abstractmethod
    def __call__(self, series: torch.Tensor) -> torch.Tensor:
        """Couple Q univariate series into Q multivariate variates.

        Parameters
        ----------
        series : torch.Tensor
            Shape ``[Q, T]`` of augmented univariate series.

        Returns
        -------
        torch.Tensor
            Shape ``[Q, T]`` of coupled multivariate variates.
        """


class IdentityCoupling(CouplingMechanism):
    """Identity / pass-through: x_j = z_j (no coupling)."""

    def __call__(self, series: torch.Tensor) -> torch.Tensor:
        return series.clone()


class UnivariateCoupling(CouplingMechanism):
    """Return a single variate sampled from the pool (degenerate coupling)."""

    def __call__(self, series: torch.Tensor) -> torch.Tensor:
        idx = int(np.random.randint(series.shape[0]))
        out = torch.zeros_like(series)
        out[0] = series[idx]
        return out


class FunctionalCoupling(CouplingMechanism):
    """x_j = f_j(z_j) + ε_j with monotone / compressive / piecewise-linear f."""

    def __init__(self, noise_std: float = 0.02) -> None:
        self.noise_std = noise_std

    def __call__(self, series: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(series)
        for j in range(series.shape[0]):
            z = series[j]
            valid = z[~torch.isnan(z)]
            if valid.numel() == 0:
                out[j] = z
                continue
            mode = np.random.choice(["log", "exp", "sigmoid", "piecewise"])
            scale = valid.std().item() or 1.0
            if mode == "log":
                shifted = z - valid.min().item() + 1.0
                y = torch.log(torch.clamp(shifted, min=1e-6))
            elif mode == "exp":
                y = torch.exp(z / max(scale, 1e-6))
            elif mode == "sigmoid":
                y = torch.sigmoid(z / max(scale, 1e-6))
            else:  # piecewise linear via abs
                y = torch.abs(z) * 0.5 + z * 0.5
            noise = torch.randn_like(y) * self.noise_std * scale
            out[j] = y + noise
        return out


class LinearMixingCoupling(CouplingMechanism):
    """x = A z where A is a QxQ mixing matrix with structured spectrum."""

    def __init__(self, spectrum: str = "dominant") -> None:
        if spectrum not in {"dominant", "uniform", "power_law"}:
            raise ValueError(f"Unknown spectrum {spectrum!r}")
        self.spectrum = spectrum

    def __call__(self, series: torch.Tensor) -> torch.Tensor:
        q = series.shape[0]
        if q == 1:
            return series.clone()
        a = self._build_matrix(q)
        return a @ series

    def _build_matrix(self, q: int) -> torch.Tensor:
        if self.spectrum == "dominant":
            s = torch.tensor([1.0] + [0.2 / i for i in range(1, q)])
        elif self.spectrum == "uniform":
            s = torch.ones(q)
        else:  # power_law
            s = 1.0 / (torch.arange(q, dtype=torch.float32) + 1.0)
        # Random orthogonal-ish matrix via QR of Gaussian.
        g = torch.randn(q, q)
        qmat, _ = torch.linalg.qr(g)
        return qmat @ torch.diag(s) @ qmat.T


class CointegrationCoupling(CouplingMechanism):
    """x_j = Λ_j τ + ξ_j with shared random-walk trends and stationary AR(1) residuals."""

    def __init__(self, ar_coef: float = 0.5) -> None:
        self.ar_coef = ar_coef

    def __call__(self, series: torch.Tensor) -> torch.Tensor:
        q, length = series.shape
        if length == 0:
            return series.clone()
        # Use the first series as a proxy for a shared random-walk trend.
        trend = torch.cumsum(torch.randn(length) * 0.1, dim=0)
        trend = trend - trend.mean()
        out = torch.empty_like(series)
        for j in range(q):
            load = float(np.random.uniform(0.5, 2.0))
            xi = torch.zeros(length)
            noise = torch.randn(length) * 0.1
            for t in range(1, length):
                xi[t] = self.ar_coef * xi[t - 1] + noise[t]
            out[j] = load * trend + xi
        return out


class LinearSCMCoupling(CouplingMechanism):
    """Linear structural causal model on a random DAG with lagged edges."""

    def __init__(self, max_lag: int = 4, edge_prob: float = 0.3) -> None:
        self.max_lag = max_lag
        self.edge_prob = edge_prob

    def __call__(self, series: torch.Tensor) -> torch.Tensor:
        q, length = series.shape
        if q == 1 or length <= self.max_lag:
            return series.clone()
        # Build a random lower-triangular DAG adjacency.
        adj = torch.rand(q, q) < self.edge_prob
        adj = torch.tril(adj, diagonal=-1)
        out = torch.empty_like(series)
        for t in range(length):
            z_t = series[:, t].clone()
            for j in range(q):
                for i in range(j):
                    if adj[j, i] and t >= self.max_lag:
                        lag = int(np.random.randint(1, self.max_lag + 1))
                        coef = float(np.random.uniform(-0.5, 0.5))
                        z_t[j] = z_t[j] + coef * series[i, t - lag]
            out[:, t] = z_t
        return out


class NonlinearSCMCoupling(CouplingMechanism):
    """Nonlinear SCM with state-dependent coupling and optional multiplicative gate."""

    def __init__(self, max_lag: int = 4, edge_prob: float = 0.3) -> None:
        self.max_lag = max_lag
        self.edge_prob = edge_prob

    def __call__(self, series: torch.Tensor) -> torch.Tensor:
        q, length = series.shape
        if q == 1 or length <= self.max_lag:
            return series.clone()
        adj = torch.rand(q, q) < self.edge_prob
        adj = torch.tril(adj, diagonal=-1)
        out = torch.empty_like(series)
        for t in range(length):
            x_t = series[:, t].clone()
            for j in range(q):
                gate = torch.sigmoid(x_t[j])
                for i in range(j):
                    if adj[j, i] and t >= self.max_lag:
                        lag = int(np.random.randint(1, self.max_lag + 1))
                        coef = float(np.random.uniform(-0.5, 0.5))
                        nonlinear = torch.tanh(series[i, t - lag])
                        x_t[j] = x_t[j] + gate * coef * nonlinear
            out[:, t] = x_t
        return out


# ---------------------------------------------------------------------------
# Post-processing enrichment
# ---------------------------------------------------------------------------


class PostProcessEnrichment:
    """Realistic covariate enrichment after coupling.

    - Variate permutation
    - Smooth time warping per variate
    - Patch masking with contiguous NaN blocks
    - Partial future observability by truncating random covariate futures
    - Discretization in value and time
    """

    def __init__(
        self,
        warp_prob: float = 0.5,
        nan_block_prob: float = 0.2,
        nan_block_max_len: int = 8,
        discretize_prob: float = 0.3,
        future_mask_prob: float = 0.2,
    ) -> None:
        self.warp = warp_prob
        self.nan_block_prob = nan_block_prob
        self.nan_block_max_len = nan_block_max_len
        self.discretize_prob = discretize_prob
        self.future_mask_prob = future_mask_prob

    def __call__(
        self, series: torch.Tensor, future_length: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return (target, past_covariates, future_covariates).

        The first variate is treated as the target; remaining variates are split
        into past and future-known covariates with partial future masking.
        """
        q, length = series.shape
        if q == 0:
            return series, None, None

        # Optionally permute variates.
        perm = torch.randperm(q)
        series = series[perm]

        # Smooth warp per variate.
        warper = SmoothTimeWarp(strength=0.03)
        for j in range(q):
            if np.random.rand() < self.warp:
                series[j] = warper(series[j].unsqueeze(0)).squeeze(0)

        # Contiguous NaN blocks.
        series = self._inject_nan_blocks(series)

        # Discretize some variates.
        series = self._discretize(series)

        # Staircase / freeze in time for some variates.
        series = self._time_freeze(series)

        target = series[0].unsqueeze(0)
        if q == 1:
            return target, None, None

        # Split remaining variates into past-only and future-known.
        num_future = max(1, q // 2)
        future_cov = series[1 : 1 + num_future]
        past_cov = series[1 + num_future :] if 1 + num_future < q else None

        # Partial future observability: mask out future portions of some covariates.
        if future_length > 0 and np.random.rand() < self.future_mask_prob:
            idx = int(np.random.randint(future_cov.shape[0]))
            future_cov[idx, -future_length:] = float("nan")

        return target, past_cov, future_cov

    def _inject_nan_blocks(self, series: torch.Tensor) -> torch.Tensor:
        q, length = series.shape
        for j in range(q):
            if np.random.rand() < self.nan_block_prob:
                block_len = int(np.random.randint(1, min(self.nan_block_max_len, length) + 1))
                start = int(np.random.randint(0, length - block_len + 1))
                series[j, start : start + block_len] = float("nan")
        return series

    def _discretize(self, series: torch.Tensor) -> torch.Tensor:
        q = series.shape[0]
        for j in range(1, q):  # keep target continuous usually
            if np.random.rand() < self.discretize_prob:
                mode = np.random.choice(["uniform", "quantile", "binary"])
                valid = series[j][~torch.isnan(series[j])]
                if valid.numel() == 0:
                    continue
                if mode == "uniform":
                    lo, hi = valid.min().item(), valid.max().item()
                    bins = torch.linspace(lo, hi, 5)
                    series[j] = torch.bucketize(series[j], bins).float()
                elif mode == "quantile":
                    qs = torch.quantile(valid, torch.linspace(0.0, 1.0, 5))
                    series[j] = torch.bucketize(series[j], qs).float()
                else:  # binary
                    median = torch.quantile(valid, 0.5)
                    series[j] = (series[j] > median).float()
        return series

    def _time_freeze(self, series: torch.Tensor) -> torch.Tensor:
        q, length = series.shape
        for j in range(q):
            if np.random.rand() < 0.2:
                # Staircase: downsample and hold.
                step = int(np.random.randint(2, 5))
                for t in range(0, length, step):
                    block_end = min(t + step, length)
                    series[j, t:block_end] = series[j, t]
        return series


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


class SyntheticCouplingPipeline:
    """Generate multivariate training samples from a pool of univariate series.

    Parameters
    ----------
    mechanisms : list[str] | None
        Coupling mechanisms to sample from. Defaults to all implemented mechanisms.
    num_variates : int
        Number of output variates Q to generate per sample.
    window_length : int
        Length T of each generated sample.
    """

    _MECHANISM_CLASSES: dict[str, type[CouplingMechanism]] = {
        "identity": IdentityCoupling,
        "univariate": UnivariateCoupling,
        "functional": FunctionalCoupling,
        "linear_mixing": LinearMixingCoupling,
        "cointegration": CointegrationCoupling,
        "linear_scm": LinearSCMCoupling,
        "nonlinear_scm": NonlinearSCMCoupling,
    }

    def __init__(
        self,
        mechanisms: Sequence[str] | None = None,
        num_variates: int = 4,
        window_length: int = 512,
    ) -> None:
        mechanisms = list(mechanisms) if mechanisms is not None else list(self._MECHANISM_CLASSES.keys())
        for name in mechanisms:
            if name not in self._MECHANISM_CLASSES:
                raise ValueError(
                    f"Unknown coupling mechanism {name!r}. Available: {list(self._MECHANISM_CLASSES.keys())}"
                )
        self.mechanisms = mechanisms
        self.num_variates = max(1, num_variates)
        self.window_length = window_length
        self._per_series_augment = [
            AmplitudeTrend(),
            QuantileCensor(),
            GaussianNoise(),
            SpikeInjection(),
        ]
        self._crop = RandomTimeCrop(window_length)
        self._enrichment = PostProcessEnrichment()

    def generate(
        self,
        univariate_pool: Sequence[torch.Tensor],
        n_samples: int,
    ) -> list[dict[str, Any]]:
        """Generate ``n_samples`` multivariate windows.

        Returns a list of dicts with keys ``target``, ``past_covariates``,
        ``future_covariates`` (tensors).
        """
        if len(univariate_pool) == 0:
            raise ValueError("univariate_pool must not be empty")

        samples = []
        for _ in range(n_samples):
            # Sample Q augmented univariate series.
            chosen = [univariate_pool[int(np.random.randint(len(univariate_pool)))] for _ in range(self.num_variates)]
            augmented = []
            for z in chosen:
                z = self._crop(z)
                for aug in self._per_series_augment:
                    z = aug(z)
                augmented.append(z)

            series = torch.stack(augmented)  # [Q, T]
            mechanism_name = str(np.random.choice(self.mechanisms))
            mechanism = self._MECHANISM_CLASSES[mechanism_name]()
            coupled = mechanism(series)

            target, past_cov, future_cov = self._enrichment(
                coupled,
                future_length=self.window_length // 4,
            )
            samples.append(
                {
                    "target": target,
                    "past_covariates": past_cov,
                    "future_covariates": future_cov,
                }
            )
        return samples


class NoAugment:
    """Identity augmentation for use as a default."""

    def __call__(self, ts: Any) -> Any:
        return ts


class TimeSeriesAugment:
    """Composable augmentation applied to ``TimeseriesType`` training samples.

    This is a lighter-weight alternative to the full synthetic coupling
    pipeline and is convenient for domain fine-tuning.
    """

    def __init__(
        self,
        transforms: Sequence[PerSeriesAugment] | None = None,
        target_prob: float = 0.5,
        covariate_prob: float = 0.3,
    ) -> None:
        self.transforms = (
            list(transforms)
            if transforms is not None
            else [
                AmplitudeTrend(),
                GaussianNoise(),
                SpikeInjection(),
            ]
        )
        self.target_prob = target_prob
        self.covariate_prob = covariate_prob

    def __call__(self, ts: Any) -> Any:
        target = self._apply(ts.target, self.target_prob)
        past_cov = self._apply(ts.past_covariates, self.covariate_prob) if ts.past_covariates is not None else None
        future_cov = (
            self._apply(ts.future_covariates, self.covariate_prob) if ts.future_covariates is not None else None
        )
        return type(ts)(target=target, past_covariates=past_cov, future_covariates=future_cov)

    def _apply(self, x: torch.Tensor | None, prob: float) -> torch.Tensor | None:
        if x is None:
            return None
        if np.random.rand() > prob:
            return x
        for transform in self.transforms:
            # Apply the same random transform across all variates in the tensor.
            x = torch.stack([transform(v) for v in x.unbind(dim=0)], dim=0)
        return x
