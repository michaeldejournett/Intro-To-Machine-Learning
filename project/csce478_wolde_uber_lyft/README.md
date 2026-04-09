# Uber / Lyft price bins — CSCE 478

**Ermias Wolde**

Kaggle dataset: [Uber & Lyft Cab prices](https://www.kaggle.com/datasets/ravi72munde/uber-lyft-cab-prices).

**What it does:** Instead of predicting exact fare in dollars, the code puts each ride into **1 of 5 price buckets** (quintiles). That way we can use accuracy, F1, and confusion matrices like the project asked. Features are ride stuff (distance, surge, cab type, etc.) plus **weather** joined by neighborhood and time.

## Files (quick map)

| File | What it does |
|------|----------------|
| `run_project.py` | Main script — run this |
| `data_loader.py` | Loads cab + weather, merges them |
| `preprocess.py` | Cleaning, one-hot, splits |
| `models.py` | Two PyTorch MLPs |
| `train.py` | Training loop |
| `evaluate.py` | Bootstrap CIs + confusion matrix plots |
| `generate_report_snippets.py` | Puts numbers into `report/generated_metrics.tex` |
| `download_data.py` | Downloads from Kaggle if you don’t have the CSVs |
| `build_report.sh` | Runs `pdflatex` on the IEEE report |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Either:

```bash
python download_data.py
```

Or put `cab_rides.csv` and `weather.csv` in a `data/` folder next to the scripts. You can also set `UBER_LYFT_DATA_DIR` to another folder.

## Train + metrics

```bash
python run_project.py
```

Faster debug run: `MAX_SAMPLES=50000 python run_project.py`

That writes `artifacts/summary_for_report.json` and confusion matrix PDFs under `report/figures/`.

Then:

```bash
python generate_report_snippets.py
```

## PDF report (LaTeX)

I used BasicTeX on a Mac:

```bash
brew install --cask basictex
eval "$(/usr/libexec/path_helper)"
sudo /Library/TeX/texbin/tlmgr update --self
sudo /Library/TeX/texbin/tlmgr install ieeetran collection-fontsrecommended
```

Then from this folder:

```bash
python generate_report_snippets.py
chmod +x build_report.sh
./build_report.sh
```

Output: `report/ieee_report.pdf`. If LaTeX is annoying, Overleaf + IEEE template works too.

## Submit

Whatever your syllabus says — usually the `.py` files, `requirements.txt`, the PDF, and maybe `artifacts/summary_for_report.json` so they don’t have to re-train.
