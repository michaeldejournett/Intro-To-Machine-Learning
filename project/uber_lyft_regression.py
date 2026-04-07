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

N_SPLITS = 20
TEST_SIZE = 0.2
BASE_RANDOM_STATE = 42

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


def evaluate_model(name: str, y_true: pd.Series, preds: np.ndarray) -> dict:
    y_true_arr = np.asarray(y_true)
    mae = mean_absolute_error(y_true_arr, preds)
    rmse = np.sqrt(mean_squared_error(y_true_arr, preds))
    r2 = r2_score(y_true_arr, preds)

    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


def ci_from_std(values: list[float], z_value: float = 1.96) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean_val = float(np.mean(arr))
    if len(arr) < 2:
        return mean_val, mean_val
    std_val = float(np.std(arr, ddof=1))
    se_val = std_val / np.sqrt(len(arr))
    margin = z_value * se_val
    return mean_val - margin, mean_val + margin


def evaluate_repeated_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_SPLITS,
    test_size: float = TEST_SIZE,
    base_random_state: int = BASE_RANDOM_STATE,
) -> tuple[pd.DataFrame, tuple[pd.Series, np.ndarray, np.ndarray, np.ndarray, pd.Series, RandomForestRegressor]]:
    metric_store = {
        "Linear Regression": {"MAE": [], "RMSE": [], "R2": []},
        "Random Forest": {"MAE": [], "RMSE": [], "R2": []},
    }
    representative = None

    for split_idx in range(n_splits):
        split_seed = base_random_state + split_idx
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=split_seed)

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        lr_model = LinearRegression()
        lr_model.fit(X_train_sc, y_train)
        preds_lr = lr_model.predict(X_test_sc)

        rf_model = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=split_seed, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        preds_rf = rf_model.predict(X_test)

        lr_metrics = evaluate_model("Linear Regression", y_test, preds_lr)
        rf_metrics = evaluate_model("Random Forest", y_test, preds_rf)

        for metric_name in ["MAE", "RMSE", "R2"]:
            metric_store["Linear Regression"][metric_name].append(float(lr_metrics[metric_name]))
            metric_store["Random Forest"][metric_name].append(float(rf_metrics[metric_name]))

        if split_idx == 0:
            representative = (y_test, preds_lr, preds_rf, X_train_sc, y_train, rf_model)

    rows = []
    for model_name in ["Linear Regression", "Random Forest"]:
        mae_values = metric_store[model_name]["MAE"]
        rmse_values = metric_store[model_name]["RMSE"]
        r2_values = metric_store[model_name]["R2"]

        mae_ci = ci_from_std(mae_values)
        rmse_ci = ci_from_std(rmse_values)
        r2_ci = ci_from_std(r2_values)

        rows.append(
            {
                "Model": model_name,
                "MAE": float(np.mean(mae_values)),
                "MAE_CI": f"[{mae_ci[0]:.3f}, {mae_ci[1]:.3f}]",
                "RMSE": float(np.mean(rmse_values)),
                "RMSE_CI": f"[{rmse_ci[0]:.3f}, {rmse_ci[1]:.3f}]",
                "R2": float(np.mean(r2_values)),
                "R2_CI": f"[{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]",
            }
        )

    if representative is None:
        raise RuntimeError("No representative split generated; check n_splits setting.")

    return pd.DataFrame(rows), representative


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

    metrics_df, representative = evaluate_repeated_holdout(X, y)
    y_test, preds_lr, preds_rf, X_train_sc, y_train, rf_model = representative
    metrics_df.to_csv(PROJECT_DIR / "paper_metrics_with_ci.csv", index=False)

    best_row = metrics_df.sort_values("MAE").iloc[0]
    print("\n=== METRICS FOR REPORT ===")
    print(f"Evaluation strategy: repeated holdout ({N_SPLITS} splits, {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} train/test)")
    print(metrics_df.to_string(index=False))
    print(f"\nBest Model: {best_row['Model']}")

    plot_actual_vs_predicted(y_test, preds_lr, preds_rf)
    plot_feature_importance(features, X_train_sc, y_train, rf_model)
    plot_confusion_matrices(y_test, preds_lr, preds_rf)

    print(f"\nSaved figures to: {PAPER_FIG_DIR}")
    print(f"Saved metrics CSV to: {PROJECT_DIR / 'paper_metrics_with_ci.csv'}")


if __name__ == "__main__":
    main()
