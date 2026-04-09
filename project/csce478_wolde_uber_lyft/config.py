"""Paths and hyperparameters for the Uber/Lyft price-bin classification project."""

from __future__ import annotations

import os
from pathlib import Path

# Default: KaggleHub cache after `python download_data.py`
_DEFAULT_KAGGLE = Path.home() / ".cache/kagglehub/datasets/ravi72munde/uber-lyft-cab-prices/versions/4"

PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_data_dir() -> Path:
    env = os.environ.get("UBER_LYFT_DATA_DIR", "").strip()
    if env:
        return Path(env)
    local = PROJECT_ROOT / "data"
    if (local / "cab_rides.csv").is_file() and (local / "weather.csv").is_file():
        return local
    return _DEFAULT_KAGGLE


DATA_DIR = _resolve_data_dir()
CAB_CSV = DATA_DIR / "cab_rides.csv"
WEATHER_CSV = DATA_DIR / "weather.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORT_FIG_DIR = PROJECT_ROOT / "report" / "figures"

# Subsample for faster runs (None = use all rows after cleaning). Override with env MAX_SAMPLES.
def _max_samples() -> int | None:
    raw = os.environ.get("MAX_SAMPLES", "").strip()
    if not raw:
        return None
    return int(raw)


MAX_SAMPLES: int | None = _max_samples()

RANDOM_STATE = 42
N_PRICE_BINS = 5
TEST_SIZE = 0.2
VAL_FRACTION_OF_TRAIN = 0.1

# Training
BATCH_SIZE = 512
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5

# Bootstrap confidence intervals (test-set metrics)
N_BOOTSTRAP = 500
