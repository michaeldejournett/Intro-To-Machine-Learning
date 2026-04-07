## Plan: Deep Paper Revision (DRAFT)

Deep revision focused on scientific credibility and consistency while keeping your chosen random split evaluation. This plan prioritizes fixing contradictions first, then improving rigor/communication, then submission polish. It uses your decisions: deep revision scope, keep random split, no strict section/page constraints.

**Steps**
1. Unify experimental narrative and artifact generation around one canonical pipeline in [project/uber_lyft_regression.py](project/uber_lyft_regression.py), then align figure references in [paper/ieee_paper_template.tex](paper/ieee_paper_template.tex).
2. Resolve surge-feature contradiction by either removing surge-effect claims or explicitly adding and documenting surge-related predictors; reflect the final choice consistently in [paper/ieee_paper_template.tex](paper/ieee_paper_template.tex#L63-L75) and [project/uber_lyft_regression.py](project/uber_lyft_regression.py#L101-L107). I found that is too significant and is kind of like using price to predict price, so i remove the surge multiplier variables. 
3. Expand Results reporting to include consistent MAE/RMSE/R2 with confidence intervals across text, tables, and CSV-linked artifacts ([paper/ieee_paper_template.tex](paper/ieee_paper_template.tex#L87-L90), [project/paper_metrics_with_ci.csv](project/paper_metrics_with_ci.csv)).
4. Add citation support for all major claims and insert a short Related Work paragraph using [paper/references.bib](paper/references.bib), including model-choice rationale and limits of random split.
5. Rewrite accuracy/impact claims to dataset-scoped language, correct MAE terminology, and add a Threats to Validity subsection in [paper/ieee_paper_template.tex](paper/ieee_paper_template.tex#L168-L189).
6. Improve reproducibility: remove hardcoded paths in [export_figures.py](export_figures.py#L5-L28) and add a runbook section in [README.md](README.md#L15-L31) for data prep, training, metric export, and figure export.
7. Final submission cleanup: remove template placeholders and ensure IEEE formatting consistency in [paper/ieee_paper_template.tex](paper/ieee_paper_template.tex#L20-L191).

**Verification**
- Rebuild paper and confirm no LaTeX warnings blocking references: run the existing build flow documented in [README.md](README.md#L15-L31).
- Regenerate metrics and verify table values match [project/paper_metrics_with_ci.csv](project/paper_metrics_with_ci.csv).
- Spot-check all claim sentences in Introduction/Conclusion for citation presence and non-overstated language.
- Confirm one-command reproducibility path from README using relative paths only.

**Decisions**
- Scope: deep revision rather than light copyedit.
- Validation: retain random split, but explicitly document its limitations and avoid overgeneralized claims.
- Structure freedom: no strict page/heading limits, so add concise Related Work and Threats to Validity sections.
