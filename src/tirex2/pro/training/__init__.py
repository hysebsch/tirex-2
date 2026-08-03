"""Training and fine-tuning support for TiRex-2."""

from .augment import (
    AmplitudeTrend,
    GaussianNoise,
    NoAugment,
    QuantileCensor,
    SmoothTimeWarp,
    SpikeInjection,
    SyntheticCouplingPipeline,
    TimeSeriesAugment,
)
from .dataset import TiRexDataset, collate_timeseries, pad_to_model_length
from .loader import (
    load_series_from_csv,
    load_series_from_gluonts,
    load_series_from_long_csv,
    load_series_from_parquet,
    load_series_from_pt,
    save_series_to_pt,
)
from .lora import LoRALinear, inject_lora, merge_lora_weights
from .loss import MaskedMAELoss, PinballLoss
from .metrics import MASE, MetricsTracker, QuantileCRPS
from .utils import (
    build_optimizer,
    build_scheduler,
    count_parameters,
    freeze_backbone,
    get_trainable_parameters,
    load_training_state,
    save_checkpoint,
)

__all__ = [
    "AmplitudeTrend",
    "GaussianNoise",
    "NoAugment",
    "QuantileCensor",
    "SmoothTimeWarp",
    "SpikeInjection",
    "SyntheticCouplingPipeline",
    "TimeSeriesAugment",
    "TiRexDataset",
    "collate_timeseries",
    "pad_to_model_length",
    "PinballLoss",
    "MaskedMAELoss",
    "MASE",
    "MetricsTracker",
    "QuantileCRPS",
    "load_series_from_csv",
    "load_series_from_long_csv",
    "load_series_from_parquet",
    "load_series_from_gluonts",
    "load_series_from_pt",
    "save_series_to_pt",
    "LoRALinear",
    "inject_lora",
    "merge_lora_weights",
    "build_optimizer",
    "build_scheduler",
    "count_parameters",
    "freeze_backbone",
    "get_trainable_parameters",
    "save_checkpoint",
    "load_training_state",
]
