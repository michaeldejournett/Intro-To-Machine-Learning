# Uber/Lyft Regression Pipeline (`uber_lyft_regression.py`)

This README documents only the regression pipeline in:

- `project/uber_lyft_regression.py`

## What this script does

The script trains and evaluates two regression models to predict ride fare price:

- Linear Regression (with feature scaling)
- Random Forest Regressor

It performs:

1. Data loading from `archive.zip`
2. Preprocessing and feature engineering
3. Repeated holdout evaluation (20 train/test splits)
4. Confidence interval computation for regression metrics using standard deviation
5. Figure export for the paper
6. Metrics CSV export for the paper table

## Data requirements

Place `archive.zip` in either:

- repository root (`Intro-To-Machine-Learning/archive.zip`), or
- `project/archive.zip`

The zip must contain:

- `cab_rides.csv`
- `weather.csv`

## Python environment

Use the project virtual environment (recommended):

```bash
source .venv/bin/activate
```

Install dependencies if needed:

```bash
pip install -r requirements.txt
```

## Run the pipeline

From repository root:

```bash
python project/uber_lyft_regression.py
```

## Outputs generated

### Metrics table

- `project/paper_metrics_with_ci.csv`

Columns:

- `Model`
- `MAE`, `MAE_CI`
- `RMSE`, `RMSE_CI`
- `R2`, `R2_CI`

### Figures

Saved to `paper/figures/`:

- `actual_vs_predicted.png`
- `feature_importance.png`
- `normalized_confusion_matrix_lr.png`
- `normalized_confusion_matrix_rf.png`

## Evaluation details

- Split strategy: repeated holdout (`n=20`) with 80/20 train/test splits
- Models are re-fit on each split
- Reported metric value = mean across splits
- 95% confidence intervals use normal approximation from split-level standard deviation:

\[
\bar{x} \pm 1.96 \cdot \frac{s}{\sqrt{n}}
\]

where:

- \(\bar{x}\) is the mean metric over splits
- \(s\) is sample standard deviation of that metric over splits
- \(n\) is number of splits

## Feature summary

Target variable:

- `price`

Feature groups include:

- Trip distance
- Time features (`hour`, `day_of_week`, `month`)
- Weather (`temp`, `clouds`, `pressure`, `rain`, `humidity`, `wind`)
- Encoded categorical fields (`cab_type`, `name`, `source`, `destination`)

Notes:

- Missing weather values are median-imputed
- `surge_multiplier` is intentionally excluded from training features

## Reproducibility notes

- Random seeds are controlled per split for deterministic reruns
- The script samples 20,000 ride rows before merging
- Main figure generation uses a representative split for plotting consistency

## Troubleshooting

- **`archive.zip not found`**: place zip in root or `project/`
- **Missing packages**: install with `pip install -r requirements.txt`
- **No output figures**: verify write permission to `paper/figures/`
