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

### If `IEEEtran.cls` is missing

Install the IEEE LaTeX class package, then re-run the build commands.

For Ubuntu/Debian-based systems:

```bash
sudo apt-get update
sudo apt-get install -y texlive-publishers
```