"""Cleaning, one-hot + scaling, and stratified splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import N_PRICE_BINS, RANDOM_STATE, TEST_SIZE, VAL_FRACTION_OF_TRAIN


CAT_COLS = ["cab_type", "destination", "location", "name", "product_id"]
NUM_COLS = [
    "distance",
    "surge_multiplier",
    "temp",
    "clouds",
    "pressure",
    "rain",
    "humidity",
    "wind",
]


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.dropna(subset=["price"]).copy()
    if "location_wx" in out.columns:
        out = out.drop(columns=["location_wx"])
    # Weather source often omits rain; treat missing as zero precipitation.
    if "rain" in out.columns:
        out["rain"] = pd.to_numeric(out["rain"], errors="coerce").fillna(0.0)
    out[CAT_COLS] = out[CAT_COLS].astype(str)
    for c in NUM_COLS:
        if c == "rain":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=NUM_COLS)
    return out


def add_price_bin_labels(df: pd.DataFrame, n_bins: int = N_PRICE_BINS) -> pd.DataFrame:
    out = df.copy()
    out["price_bin"], _ = pd.qcut(
        out["price"],
        q=n_bins,
        labels=False,
        retbins=True,
        duplicates="drop",
    )
    return out


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ]
    )


def prepare_arrays(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer | None = None,
    fit: bool = True,
):
    y = df["price_bin"].to_numpy(dtype=np.int64)
    X_df = df[NUM_COLS + CAT_COLS]
    if preprocessor is None:
        preprocessor = build_preprocessor()
    if fit:
        X = preprocessor.fit_transform(X_df)
    else:
        X = preprocessor.transform(X_df)
    return X.astype(np.float32), y, preprocessor


def train_val_test_split_stratified(X, y, random_state: int = RANDOM_STATE):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr,
        y_tr,
        test_size=VAL_FRACTION_OF_TRAIN,
        random_state=random_state,
        stratify=y_tr,
    )
    return X_train, X_val, X_te, y_train, y_val, y_te
