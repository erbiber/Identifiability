#!/usr/bin/env bash
# Rebuild every result, figure and check from the supplied data. Run from the repository root.
set -euo pipefail
pip install -r requirements.txt
python scripts/import_author_data.py      # data/*.csv from the supplied workbook
python scripts/run_results.py             # results/*.csv (about 1 h on two cores)
python scripts/synthetic_validation.py    # simulation tests of Section 5.4 and Fig. 7 (about 4 min)
python scripts/paper_figures.py           # Figs. 1-6 and S1-S11
python scripts/toc_graphic.py             # graphical abstract
python tests/test_analytic.py             # analytic limits
python tests/test_synthetic.py            # simulation values
python tests/test_import.py               # PET per-fiber table against the workbook
python tests/test_claims.py               # every value quoted in the article and Supplementary material
