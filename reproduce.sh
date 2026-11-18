#!/usr/bin/env bash
# Regenerate every result file and figure of the article and its Supplementary material, then run the tests.
# Run from the repository root:  bash reproduce.sh
set -euo pipefail
python scripts/run_results.py           # every result file except the simulation tests (about 3.5 h on one core)
python scripts/paper_figures.py         # Figs. 1-6 and the Supplementary figures
python scripts/synthetic_validation.py  # simulation tests of Section 5.4 / S10 and Fig. 7 (about 4 min)
if [ -f scripts/toc_graphic.py ]; then python scripts/toc_graphic.py; fi   # graphical abstract
python -m pytest -q tests               # analytic limits, headline values (claims.csv) and simulation values
