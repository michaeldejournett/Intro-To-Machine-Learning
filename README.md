# Intro-To-Machine-Learning

## Repository Layout

- `project/` — machine learning notebooks and project work
- `paper/` — IEEE LaTeX paper files and bibliography

## IEEE LaTeX Paper Template (BibTeX)

This repository includes an IEEE-style LaTeX template and BibTeX bibliography files:

- `paper/ieee_paper_template.tex`
- `paper/references.bib`

### Build commands

Run the following from the project root:

```bash
cd paper
pdflatex ieee_paper_template.tex
bibtex ieee_paper_template
pdflatex ieee_paper_template.tex
pdflatex ieee_paper_template.tex
```

The final PDF will be generated as `paper/ieee_paper_template.pdf`.

## Reproducibility Runbook

Run the ML pipeline and export all paper artifacts from the repository root:

```bash
python project/uber_lyft_regression.py
```

This command generates:
- `project/paper_metrics_with_ci.csv`
- `paper/figures/actual_vs_predicted.png`
- `paper/figures/feature_importance.png`
- `paper/figures/normalized_confusion_matrix_lr.png`
- `paper/figures/normalized_confusion_matrix_rf.png`

Optional wrapper (same canonical pipeline):

```bash
python export_figures.py
```

Data requirement:
- Place `archive.zip` in the repository root or in `project/`.
- The zip must contain `cab_rides.csv` and `weather.csv`.

Suggested full build flow:

```bash
python project/uber_lyft_regression.py
cd paper
pdflatex ieee_paper_template.tex
bibtex ieee_paper_template
pdflatex ieee_paper_template.tex
pdflatex ieee_paper_template.tex
```

### If `IEEEtran.cls` is missing

Install the IEEE LaTeX class package, then re-run the build commands.

For Ubuntu/Debian-based systems:

```bash
sudo apt-get update
sudo apt-get install -y texlive-publishers
```