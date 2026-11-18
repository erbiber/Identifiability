# Updating the repository to revision R23 (Polymer)

This folder holds every file that is new or changed for R23. Copy it over your local R22 working tree (the full tree
with `data/`, `claims.csv`, `tests/`, `raman_weighting.py`, etc.), then follow the steps below. Files of the R22 tree
that are not here stay as they are.

## 1. Copy

| Path | Status | Note |
|---|---|---|
| `README.md` | replaced | new title, *Polymer*, result map from Table S16, tests, licence |
| `CITATION.cff`, `.zenodo.json` | replaced | new title, version 2.0.0, keywords |
| `requirements.txt` | replaced | versions used for the simulation tests; check against your R22 environment |
| `reproduce.sh` | replaced | now also runs `synthetic_validation.py` and all tests; compare with your R22 file before overwriting |
| `scripts/synthetic_validation.py` | new | simulation tests V1–V4 (Section 5.4, S10); seeds 101, 202, 303, 404 |
| `results/synth_plate_control.csv`, `synth_route3.csv`, `synth_route4.csv`, `synth_robustness.csv` | new | reproduced byte for byte by a clean run (about 4 min) |
| `figures/fig7_synthetic_validation.pdf`, `.png` | new | Fig. 7 of the article |
| `tests/test_synthetic.py` | new | checks the Section 5.4 / S10 values; passes on these files |
| `data/raw/pellerin_compilation_of_data.xlsx` | new | the workbook supplied by C. Pellerin; source of the `data/*.csv` record files |
| `data/README.md` | new or replaced | provenance of the supplied and digitized values; merge with any existing text |
| `tools/update_figure_labels.py` | new | updates figure labels in `scripts/paper_figures.py` (step 2) |

## 2. Figures that still show old wording

The article text was changed, but these labels are drawn by `scripts/paper_figures.py`:

1. Run `python tools/update_figure_labels.py` (dry run), check the report, then `--write`.
   It replaces "flat diameter" → "uniform-arrest diameter", "A1 has content" → "A1 permits radial variation",
   "excluded by the plate" → "excluded under the observed plate ceiling" and "[laboratory values]" →
   "(supplied by the authors)".
2. Edit Fig. 5 (decision tree) by hand:
   - The "no" box that reads "Below the flat diameter, a continued rise excludes a skin of bounded slope (Remark 2)"
     should read: **"Below the uniform-arrest diameter: a rise larger than ΓΔΘ* excludes a bounded-slope skin,
     once Γ and the spread of arrest clocks ΔΘ* are bounded (Remark 2)"**.
   - Separate the two exits that are now merged: **"A1–A3 fail: the record cannot be interpreted (steps 1–3)"** and
     **"Below the uniform-arrest diameter: Remark 2 test"**.
3. Run `python scripts/paper_figures.py` and check Figs. 4, 5 and 6 at journal column width.
4. Check the figure file names against the article numbering (Fig. 4 uniform-arrest diameter and validity envelope;
   Fig. 5 decision tree; Fig. 6 PS and PET skin bounds; Fig. 7 simulation tests).

## 3. Table S18 replicate counts

Five cells of SI Table S18 read `[REPLICATES]`. Take the numbers from the code and replace them in the SI:
PS 5th-percentile scenarios (part `psunc`), PET collector medians (part `pet`/`channels`), PET offset with log
diameter (`channels`), PAN 90% equivalence interval and PAN slope interval (`pan`).

## 4. claims.csv

The simulation values are checked by `tests/test_synthetic.py`, so `claims.csv` does not need new rows. If you prefer
one map, add rows for the Section 5.4 values and update the count "117 headline values" in SI Section S9.

## 5. Run everything

```bash
pip install -r requirements.txt
bash reproduce.sh          # about 3.5 h; or run the three scripts and pytest separately
git status                 # result files should be unchanged except the new synth_*.csv and redrawn figures
```

## 6. Commit, release and DOI

```bash
git add -A
git commit -m "R23: Polymer revision - uniform-arrest diameter, simulation tests, supplied data, metadata"
git push origin main
git tag -a v2.0.0 -m "Version submitted to Polymer (R23)"
git push origin v2.0.0
```

On GitHub, create a release from tag `v2.0.0`. With the Zenodo–GitHub integration enabled, Zenodo archives the release
and mints a DOI. Then:

1. Put the DOI in the article reference [62] and SI reference [S36], replacing `zenodo.[DOI to be assigned]`.
2. Put it in `README.md` (`zenodo.XXXXXXX`) and, if you like, in `CITATION.cff` (`doi:`), and push (no new release
   needed; the concept DOI resolves to the latest version).

## 7. Before submission

- Written agreement from C. Pellerin covering the public archiving of the supplied values and the personal
  communication of the 29.4 m/s disk speed (the article says the data are "shared with his agreement").
- Export the figures as separate files (vector PDF for line art), and the article PDF from Word, not LibreOffice
  (LibreOffice drops the equations).
