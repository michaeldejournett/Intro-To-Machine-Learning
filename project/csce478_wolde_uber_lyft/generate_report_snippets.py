"""Reads summary JSON and writes generated_metrics.tex for the paper."""

from __future__ import annotations

import json
from pathlib import Path

import config as cfg


def fmt(x: float) -> str:
    return f"{x:.4f}"


def main() -> None:
    src = cfg.ARTIFACTS_DIR / "summary_for_report.json"
    if not src.exists():
        raise SystemExit(f"Can't find {src} — run python run_project.py first.")
    with open(src) as f:
        data = json.load(f)

    lines = []
    for name, payload in data.items():
        acc = payload["accuracy"]
        lo, hi = payload["accuracy_ci95"]
        f1 = payload["macro_f1"]
        f1_lo, f1_hi = payload["macro_f1_ci95"]
        lines.append(
            f"\\newcommand{{\\{name}Acc}}{{{fmt(acc)}}}"
        )
        lines.append(
            f"\\newcommand{{\\{name}AccLo}}{{{fmt(lo)}}}"
        )
        lines.append(
            f"\\newcommand{{\\{name}AccHi}}{{{fmt(hi)}}}"
        )
        lines.append(
            f"\\newcommand{{\\{name}Fone}}{{{fmt(f1)}}}"
        )
        lines.append(
            f"\\newcommand{{\\{name}FoneLo}}{{{fmt(f1_lo)}}}"
        )
        lines.append(
            f"\\newcommand{{\\{name}FoneHi}}{{{fmt(f1_hi)}}}"
        )

    out = cfg.PROJECT_ROOT / "report" / "generated_metrics.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
