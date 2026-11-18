#!/usr/bin/env bash
# Recompute every result and figure, then check the headline values against the paper. Run from the repository root.
set -euo pipefail
pip install -r requirements.txt
python scripts/run_results.py
python scripts/paper_figures.py
python scripts/toc_graphic.py
python tests/test_analytic.py
python tests/test_claims.py
