#!/usr/bin/env bash
# Build report/ieee_report.pdf (run from project root after BasicTeX + tlmgr packages are installed).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PATH="/Library/TeX/texbin:$PATH"
cd "$ROOT/report"
pdflatex -interaction=nonstopmode -halt-on-error ieee_report.tex
pdflatex -interaction=nonstopmode -halt-on-error ieee_report.tex
echo "Done: $ROOT/report/ieee_report.pdf"
