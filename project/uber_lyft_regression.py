from __future__ import annotations

import warnings
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8")
sns.set_context("talk")

ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(__file__).resolve().parent
PAPER_DIR = ROOT_DIR / "paper"
PAPER_FIG_DIR = PAPER_DIR / "figures"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)


def find_zip_path() -> Path:
    candidates = [ROOT_DIR / "archive.zip", PROJECT_DIR / "archive.zip"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("archive.zip not found in project/ or repository root.")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    zip_path = find_zip_path()
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("cab_rides.csv") as f:
            rides = pd.read_csv(f)
        with z.open("weather.csv") as f:
            weather = pd.read_csv(f)

    rides = rides.sample(n=20_000, random_state=42)
    return rides, weather, zip_path


def prepare_data(rides: pd.DataFrame, weather: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    rides = rides.copy()
    weather = weather.copy()

    rides.dropna(subset=["price"], inplace=True)

    rides["datetime"] = pd.to_datetime(rides["time_stamp"], unit="ms")
    rides["hour"] = rides["datetime"].dt.hour
    rides["day_of_week"] = rides["datetime"].dt.dayofweek
    rides["month"] = rides["datetime"].dt.month

    rides["time_stamp_hr"] = (rides["time_stamp"] // 3600000) * 3600
    weather["time_stamp_hr"] = (weather["time_stamp"].astype(int) // 3600) * 3600

    df = rides.merge(
        weather[["temp", "clouds", "pressure", "rain", "humidity", "wind", "location", "time_stamp_hr"]],
        left_on=["source", "time_stamp_hr"],
        right_on=["location", "time_stamp_hr"],
        how="left",
    )

    matched = df["temp"].notna().sum()
    print(f"Merged shape: {df.shape}")
    print(f"Weather matched: {matched:,} / {len(df):,} rows ({matched / len(df) * 100:.1f}%)")

    for col in ["temp", "clouds", "pressure", "rain", "humidity", "wind"]:
        df[col].fillna(df[col].median(), inplace=True)

    cat_cols = ["cab_type", "name", "source", "destination"]
    for col in cat_cols:
        encoder = LabelEncoder()
        df[col + "_enc"] = encoder.fit_transform(df[col].astype(str))

    features = [
        "distance",
        "hour",
        "day_of_week",
        "month",
        "temp",
        "clouds",
        "pressure",
        "rain",
        "humidity",
        "wind",
        "cab_type_enc",
        "name_enc",
        "source_enc",
        "destination_enc",
    ]

    X = df[features].copy()
    y = df["price"].copy()
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Remaining NaNs: {X.isna().sum().sum()}")
    print("surge_multiplier excluded from training features")

    return X, y, features


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_boot: int = 500,
    random_state: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(metric_fn(y_true[idx], y_pred[idx]))
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(low), float(high)


def evaluate_model(name: str, y_true: pd.Series, preds: np.ndarray) -> dict:
    y_true_arr = np.asarray(y_true)
    mae = mean_absolute_error(y_true_arr, preds)
    rmse = np.sqrt(mean_squared_error(y_true_arr, preds))
    r2 = r2_score(y_true_arr, preds)

    mae_ci = bootstrap_metric_ci(y_true_arr, preds, mean_absolute_error)
    rmse_ci = bootstrap_metric_ci(y_true_arr, preds, lambda a, b: np.sqrt(mean_squared_error(a, b)))
    r2_ci = bootstrap_metric_ci(y_true_arr, preds, r2_score)

    return {
        "Model": name,
        "MAE": mae,
        "MAE_CI": f"[{mae_ci[0]:.3f}, {mae_ci[1]:.3f}]",
        "RMSE": rmse,
        "RMSE_CI": f"[{rmse_ci[0]:.3f}, {rmse_ci[1]:.3f}]",
        "R2": r2,
        "R2_CI": f"[{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]",
    }


def plot_actual_vs_predicted(y_test: pd.Series, preds_lr: np.ndarray, preds_rf: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_test, preds_lr, alpha=0.2, s=5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    axes[0].set_xlabel("Actual Price ($)")
    axes[0].set_ylabel("Predicted Price ($)")
    axes[0].set_title("Linear Regression — Actual vs Predicted")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(y_test, preds_rf, alpha=0.2, s=5, color="orange")
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    axes[1].set_xlabel("Actual Price ($)")
    axes[1].set_ylabel("Predicted Price ($)")
    axes[1].set_title("Random Forest — Actual vs Predicted")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PAPER_FIG_DIR / "actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(features: list[str], X_train_sc: np.ndarray, y_train: pd.Series, rf: RandomForestRegressor) -> None:
    importances_rf = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)

    lr_model = LinearRegression()
    lr_model.fit(X_train_sc, y_train)
    importances_lr = pd.Series(np.abs(lr_model.coef_), index=features).sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    importances_lr.plot(kind="barh", ax=axes[0], color="steelblue")
    axes[0].set_title("Linear Regression — Feature Coefficients (abs)")
    axes[0].set_xlabel("Coefficients (abs)")

    importances_rf.plot(kind="barh", ax=axes[1], color="darkorange")
    axes[1].set_title("Random Forest — Feature Importances")
    axes[1].set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig(PAPER_FIG_DIR / "feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(y_test: pd.Series, preds_lr: np.ndarray, preds_rf: np.ndarray) -> None:
    n_bins = 3
    bin_edges = np.quantile(y_test, np.linspace(0, 1, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 4:
        bin_edges = np.array([y_test.min() - 1e-9, *bin_edges[1:-1], y_test.max() + 1e-9])
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    labels = ["Low", "Medium", "High"]
    if len(bin_edges) != 4:
        labels = [f"Bin {i+1}" for i in range(len(bin_edges) - 1)]

    y_true_binned = pd.cut(y_test, bins=bin_edges, labels=labels, include_lowest=True)

    for model_name, preds in {"lr": preds_lr, "rf": preds_rf}.items():
        y_pred_binned = pd.cut(preds, bins=bin_edges, labels=labels, include_lowest=True)
        cm = pd.crosstab(y_true_binned, y_pred_binned, normalize="index").reindex(index=labels, columns=labels)

        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
        plt.title(f"Normalized Confusion Matrix — {model_name.upper()}")
        plt.xlabel("Predicted Bin")
        plt.ylabel("True Bin")
        plt.tight_layout()
        plt.savefig(PAPER_FIG_DIR / f"normalized_confusion_matrix_{model_name}.png", dpi=300, bbox_inches="tight")
        plt.close()


def main() -> None:
    rides, weather, zip_path = load_data()
    print(f"Using zip file: {zip_path.resolve()}")
    print(f"Rides shape: {rides.shape}")
    print(f"Weather shape: {weather.shape}")

    X, y, features = prepare_data(rides, weather)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    lr_model = LinearRegression()
    lr_model.fit(X_train_sc, y_train)
    preds_lr = lr_model.predict(X_test_sc)

    rf_model = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    preds_rf = rf_model.predict(X_test)

    metrics_lr = evaluate_model("Linear Regression", y_test, preds_lr)
    metrics_rf = evaluate_model("Random Forest", y_test, preds_rf)
    metrics_df = pd.DataFrame([metrics_lr, metrics_rf])
    metrics_df.to_csv(PROJECT_DIR / "paper_metrics_with_ci.csv", index=False)

    best_row = metrics_df.sort_values("MAE").iloc[0]
    print("\n=== METRICS FOR REPORT ===")
    print(metrics_df.to_string(index=False))
    print(f"\nBest Model: {best_row['Model']}")

    plot_actual_vs_predicted(y_test, preds_lr, preds_rf)
    plot_feature_importance(features, X_train_sc, y_train, rf_model)
    plot_confusion_matrices(y_test, preds_lr, preds_rf)

    print(f"\nSaved figures to: {PAPER_FIG_DIR}")
    print(f"Saved metrics CSV to: {PROJECT_DIR / 'paper_metrics_with_ci.csv'}")


if __name__ == "__main__":
    main()
