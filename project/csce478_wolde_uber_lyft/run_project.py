"""
End-to-end pipeline: load → preprocess → train two DL models → evaluate.

Place Kaggle files under DATA_DIR (see config.py) or run with kagglehub once
so cab_rides.csv and weather.csv exist.
"""

from __future__ import annotations

import json
import pickle
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import config as cfg
from data_loader import load_merged_dataframe
from evaluate import evaluate_and_plot
from models import DeepBatchNormMLP, ShallowMLP
from preprocess import (
    add_price_bin_labels,
    clean_frame,
    prepare_arrays,
    train_val_test_split_stratified,
)
from train import predict_labels, save_model, train_one_model


def set_seed(seed: int = cfg.RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    set_seed()
    cfg.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Data folder: {cfg.DATA_DIR}")
    print("Loading and merging cab + weather …")
    df = load_merged_dataframe()
    df = clean_frame(df)
    df = add_price_bin_labels(df)

    if cfg.MAX_SAMPLES is not None and len(df) > cfg.MAX_SAMPLES:
        df, _ = train_test_split(
            df,
            train_size=cfg.MAX_SAMPLES,
            stratify=df["price_bin"],
            random_state=cfg.RANDOM_STATE,
        )
    n_classes = int(df["price_bin"].max()) + 1
    labels = list(range(n_classes))

    print(f"Rows after cleaning: {len(df)}, classes: {n_classes}")

    X, y, preproc = prepare_arrays(df, fit=True)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_stratified(
        X, y
    )
    print(f"Train {len(y_train)}, val {len(y_val)}, test {len(y_test)}, features {X.shape[1]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    results = {}

    specs = [
        (
            "ShallowMLP",
            ShallowMLP(X.shape[1], n_classes),
        ),
        (
            "DeepBatchNormMLP",
            DeepBatchNormMLP(X.shape[1], n_classes),
        ),
    ]

    for name, model in specs:
        print(f"\n=== Training {name} ===")
        info = train_one_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            device,
        )
        pred_test = predict_labels(model, X_test, device)
        ev = evaluate_and_plot(name, y_test, pred_test, labels)
        results[name] = {**ev, "train_info": info}
        save_model(
            cfg.ARTIFACTS_DIR / f"{name}.pt",
            model,
            {"name": name, "n_features": X.shape[1], "n_classes": n_classes},
        )

    with open(cfg.ARTIFACTS_DIR / "preprocessor.pkl", "wb") as f:
        pickle.dump(preproc, f)

    summary_path = cfg.ARTIFACTS_DIR / "summary_for_report.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. Artifacts in {cfg.ARTIFACTS_DIR}, summary: {summary_path}")


if __name__ == "__main__":
    main()
