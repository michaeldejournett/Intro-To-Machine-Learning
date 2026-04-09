"""Bootstrap confidence intervals, confusion matrices, and figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from config import N_BOOTSTRAP, RANDOM_STATE, REPORT_FIG_DIR


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_boot: int = N_BOOTSTRAP,
    random_state: int = RANDOM_STATE,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        scores.append(float(metric_fn(yt, yp)))
    scores = np.sort(np.array(scores))
    lo = scores[int((alpha / 2) * n_boot)]
    hi = scores[int((1 - alpha / 2) * n_boot) - 1]
    point = float(metric_fn(y_true, y_pred))
    return point, lo, hi


def evaluate_and_plot(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
    out_dir: Path | None = None,
) -> dict:
    out_dir = out_dir or REPORT_FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    acc, acc_lo, acc_hi = bootstrap_metric_ci(
        y_true, y_pred, lambda yt, yp: accuracy_score(yt, yp)
    )
    f1m, f1_lo, f1_hi = bootstrap_metric_ci(
        y_true,
        y_pred,
        lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Predicted bin")
    ax.set_ylabel("True bin")
    ax.set_title(f"{name} — row-normalized confusion matrix")
    fig.tight_layout()
    fig_path = out_dir / f"confusion_norm_{name.replace(' ', '_')}.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    report = classification_report(
        y_true, y_pred, labels=labels, digits=4, zero_division=0
    )

    result = {
        "model": name,
        "accuracy": acc,
        "accuracy_ci95": [acc_lo, acc_hi],
        "macro_f1": f1m,
        "macro_f1_ci95": [f1_lo, f1_hi],
        "confusion_matrix_raw": cm.tolist(),
        "confusion_matrix_normalized_rows": cm_norm.tolist(),
        "classification_report": report,
        "figure_pdf": str(fig_path),
    }
    with open(out_dir / f"metrics_{name.replace(' ', '_')}.json", "w") as f:
        json.dump(result, f, indent=2)
    return result
