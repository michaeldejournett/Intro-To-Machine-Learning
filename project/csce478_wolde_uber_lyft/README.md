# Uber / Lyft price bins (ML project)

**Ermias Wolde · CSCE 478**

Predicts which **price quintile** a ride is in (5 classes) using ride details plus **weather** matched by neighborhood and time.  
Dataset: [Uber & Lyft Cab prices](https://www.kaggle.com/datasets/ravi72munde/uber-lyft-cab-prices) (Kaggle).

## What’s in the repo

| File | Role |
|------|------|
| `run_project.py` | Full pipeline: load data → preprocess → train 2 neural nets → metrics + figures |
| `data_loader.py` | Merge `cab_rides.csv` + `weather.csv` |
| `preprocess.py` | Cleaning, labels, train/val/test split |
| `models.py` | Shallow MLP vs deeper batch-norm MLP |
| `train.py` | PyTorch training + early stopping |
| `evaluate.py` | Bootstrap CIs + normalized confusion matrices |
| `generate_report_snippets.py` | Writes `report/generated_metrics.tex` for the IEEE write-up |
| `download_data.py` | Pulls the Kaggle dataset (needs internet once) |
| `report/ieee_report.tex` | IEEE-style report (needs `IEEEtran` + `pdflatex`) |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

**Option A — automatic (recommended):**

```bash
python download_data.py
```

**Option B — manual:** unzip Kaggle’s files into a `data/` folder next to these scripts:

```
data/cab_rides.csv
data/weather.csv
```

You can also point elsewhere: `export UBER_LYFT_DATA_DIR=/path/to/folder`

## Run

```bash
python run_project.py
```

Optional: use fewer rows for a quick test: `MAX_SAMPLES=50000 python run_project.py`

Outputs:

- `artifacts/summary_for_report.json` — numbers for your report  
- `report/figures/confusion_norm_*.pdf` — normalized confusion matrices  

Then update the LaTeX metrics file:

```bash
python generate_report_snippets.py
```

Build the PDF (if you have LaTeX):

```bash
pdflatex -output-directory=report report/ieee_report.tex
```

Or upload `report/` to **Overleaf** with the [IEEE conference template](https://www.ieee.org/conferences/publishing/templates) and drop in `IEEEtran.cls` if needed.

## Before you submit

1. **PDF:** Run `python run_project.py`, then `python generate_report_snippets.py`, then compile `report/ieee_report.tex` so tables and text match your latest numbers.  
2. **Course rules:** Add school / semester if your syllabus asks for it (only the report line was set to **Ermias Wolde** and **CSCE 478**).  
3. Zip or push `*.py`, `requirements.txt`, `report/`, and `artifacts/summary_for_report.json` if the grader should see metrics without re-running training.
