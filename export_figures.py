#!/usr/bin/env python3
"""Generate paper figures and metrics via the canonical project pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    pipeline_script = root_dir / "project" / "uber_lyft_regression.py"

    if not pipeline_script.exists():
        raise FileNotFoundError(f"Pipeline script not found: {pipeline_script}")

    print(f"Running canonical pipeline: {pipeline_script}")
    subprocess.run([sys.executable, str(pipeline_script)], check=True, cwd=root_dir)
    print("Completed. Figures are in paper/figures and metrics are in project/paper_metrics_with_ci.csv")


if __name__ == "__main__":
    main()
