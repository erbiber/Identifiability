# Observational Limits on Assigning Molecular Orientation Mechanisms in Electrospun Polymer Fibers — code, data and results

Code, data, computed values and figures for the article

> Biber, E. *Observational Limits on Assigning Molecular Orientation Mechanisms in Electrospun Polymer Fibers*.
> Submitted to *Journal of Polymer Science* (2026).

Archived release: Zenodo, https://doi.org/10.5281/zenodo.XXXXXXX (version 2.1.0, the submitted version of the
article). Development version: https://github.com/erbiber/Identifiability.

Every number quoted in the article and its Supporting Information is computed by the scripts here and written to
`results/*.csv`, and `tests/test_claims.py` checks each quoted value against those files (`claims.csv`). The article
is a theoretical and computational study with reanalysis of published data; no new experiments are reported. The
published polystyrene, poly(ethylene terephthalate) and polyacrylonitrile records enter through the per-fiber and bin
values supplied by the corresponding author of the source articles and shared with his agreement (`data/README.md`).

## Reproducing the results

Tested with Python 3.13, numpy 2.5.3, scipy 1.18.1 and matplotlib 3.11.2 (`requirements.txt`).

```bash
bash reproduce.sh
```

which runs, from the repository root:

```bash
python scripts/import_author_data.py      # data/*.csv from data/raw/pellerin_compilation_of_data.xlsx
python scripts/run_results.py             # every result file except the simulation tests (about 1 h on two cores)
python scripts/synthetic_validation.py    # simulation tests of Section 5.4 / S10 and Fig. 7 (about 4 min)
python scripts/paper_figures.py           # Figs. 1-6 and S1-S11
python scripts/toc_graphic.py             # graphical abstract
python tests/test_analytic.py             # analytic limits and solver checks
python tests/test_synthetic.py            # simulation values
python tests/test_claims.py               # quoted values against the result files
python tests/test_import.py               # PET per-fiber table against the supplied workbook
```

`run_results.py` accepts part names to recompute only those parts (`python scripts/run_results.py pan pet`).
Numbers are written to six significant figures; all random draws use fixed seeds, so a rerun reproduces every file.

## Tests

- `tests/test_claims.py` checks every value listed in `claims.csv` (result file, row filter, column and aggregate)
  to within half a unit of its last quoted digit.
- `tests/test_analytic.py` checks the inversion of the shell fraction, the three cases of Proposition 3, the
  electrostatic limit and surface continuity of the cylinder field, the finite-difference solver against the series
  solution, the Kuhn-Grün relation at small extension, the free-volume parameters, the scaling of Eq. (9), the arrest
  criterion and the exact bootstrap interval of a median.
- `tests/test_synthetic.py` checks the values of Section 5.4 and Section S10.
- `tests/test_import.py` re-reads every workbook cell cited in `data/pet_per_fiber.csv` and checks the PET fibers by
  value and identifier, never by row position.

Run all of them with `python -m pytest -q tests` from the repository root, or each one directly with `python tests/<name>.py`.
The tests check consistency between the text and the code, not the correctness of the models.

## Where each result is computed

Section numbers refer to the article; S-numbers to the Supporting Information. Parts refer to `scripts/run_results.py`.

| Result | Where reported | Part (module) | Output files (`results/`) |
|---|---|---|---|
| Lemma 1 and the arrest order | 2.3; Fig. S2 | lemma1 (october.py, lemma1.py) | lemma1_drying.csv, arrest_order.csv |
| A5: injectivity, conditioning, jet splitting | S3.3–S3.4 | a5, a5cond (october.py, conditioning.py) | a5_injectivity.csv, a5_conditioning.csv, a5_splitting.csv |
| Bayes factors for a committed skin | 3.2; Fig. S6 | bayes (october.py) | bayes_factors_*.csv |
| Fading-memory mimicry, Propositions 3–4 | 3.3; Fig. 3; S2 | kernel, prop3 (october.py) | kernel_*.csv, proposition3*.csv |
| Proposition 4 when stretching ends before the axis arrests | 3.3; S2.3 | prop4_early_end (conditioning.py, separability.py) | proposition4_early_end.csv |
| Biot-number scope of Eq. (9) | 4.1; S5.1 | biot (conditioning.py) | biot_scope.csv |
| Joint collector and readout scenarios for the PS bound | 5.1.2; S8.3 | ps_joint (conditioning.py) | ps_joint_scenarios.csv |
| Finite extensibility | S2.5; Fig. S1 | finite (finite.py) | finite_extensibility_*.csv |
| Functional forms (Proposition S1) | S6.2; Fig. S7 | forms, forms_sampling | functional_forms*.csv |
| Route 3 coupling and sample sizes | S7; Fig. S8 | coupling (october.py) | coupling_*.csv, route3_sample_sizes.csv, sample_sizes.csv |
| Arrest composition | S4.1; Fig. S3 | composition | arrest_composition.csv |
| PS–chloroform diffusivity, Eq. (9), Tables S4–S6 | 4.1; S5; Figs. S12a, S5 | flat (diffusivity.py) | flat_diameter.csv, diffusivity_sensitivity.csv, flat_diameter_joint_worst.csv, chloroform_*.csv, diffusivity_vs_data.csv, a3_inertness.csv |
| Self-consistent uniform-arrest diameter, A3 limit, envelope, case study, budget | 4.1–4.2, 5.1.1; S4.3, S5.3, S8.5; Figs. S12b, S4 | drying (ps_drying.py) | flat_diameter_consistent.csv, a3_post_arrest.csv, a3_flat_dependence.csv, a3_window_estimate.csv, variant_effects.csv, flat_vs_eq9.csv, case_study_consistent.csv, ps_window_global.csv, flat_threshold_sensitivity.csv, flat_diameter_budget.csv |
| Evaporation rate, relaxation law and grid checks | S4.3, S5.3 | drying_checks | a3_post_arrest_checks.csv, flat_diameter_budget_evaporation.csv |
| Arrest front; trend steepening (Remark 1) | 5.1.2, 4.2; S8.3, S4.3 | skinfront, remark3 | ps_arrest_front.csv, a3_trend_steepening.csv |
| PS skin bound and two-channel analysis | 5.1.2–5.1.3; S8.3; Fig. 5a; Tables S10–S11 | records, psunc (records.py) | ps_*.csv |
| PET plate bound, plate control, channels | 5.2; S8.6–S8.7; Figs. 5b, S10 | pet, channels (records.py) | pet_*.csv |
| PAN readout agreement and skin bound | 5.3.1; S8.8; Fig. S11 | pan (records.py) | pan_*.csv |
| Confocal readout weight of a thin skin | 5.1.2; S8.4; Table S12 | raman, raman_check (raman_weighting.py, raman_fd.py) | raman_*.csv, ps_optical_bound*.csv |
| Simulation tests of the roadmap | 5.4; S10; Fig. S13 | scripts/synthetic_validation.py | synth_*.csv |

File and part names beginning `flat` refer to the uniform-arrest diameter.

## Contents

```
scripts/      analysis modules, run_results.py, paper_figures.py, synthetic_validation.py, toc_graphic.py,
              import_author_data.py, digitize_pet_fig3b.py
data/         supplied record values (CSV, and the original workbook in data/raw/), literature tables used by the
              diffusivity model, and data/digitized/ (independent extraction of PET Fig. 3b)
results/      computed values (CSV)
figures/      figures as PDF (vector) and PNG
tests/        test_analytic.py, test_claims.py, test_import.py, test_synthetic.py
claims.csv    map from each quoted value to its result file, rows, column and aggregate
reproduce.sh  one-command reproduction
```

## License

Code: MIT (`LICENSE`). Results, figures and data: CC BY 4.0 (`LICENSE-DATA`). When reusing the record values or the
literature tables, please also cite the source articles listed in `data/README.md`.

## Use of AI tools

The numerical code was developed with the assistance of a generative AI tool (Claude, Anthropic), which was also used
to check the values quoted in the text against the computed result files. The author verified the code against
analytic limits and independent calculations, as described in the Data availability statement of the article.
