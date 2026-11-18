# Observational limits on assigning molecular orientation mechanisms in electrospun polymer fibers — code, data and results

Code, data, computed values and figures for the article

> Biber, E. *Observational limits on assigning molecular orientation mechanisms in electrospun polymer fibers*.
> Submitted to *Polymer* (2026).

Archived release: Zenodo, https://doi.org/10.5281/zenodo.XXXXXXX (version 2.0.0, corresponding to the submitted
version of the article). Development version: https://github.com/erbiber/Identifiability.

Every number quoted in the article and its Supplementary material is computed by the scripts here and written to
`results/*.csv`; every figure is drawn from those files or from the same model functions. The article is a
theoretical and computational study with reanalysis of published data. No new experiments are reported. The
published records it reanalyses (polystyrene, poly(ethylene terephthalate) and polyacrylonitrile) enter through the
per-fiber and bin values in `data/`, supplied by the corresponding author of the source articles and shared with his
agreement (see `data/README.md`).

## Reproducing the results

Tested with Python 3.13.18, numpy 2.5.3, scipy 1.18.1 and matplotlib 3.11.2 (`requirements.txt`).

```bash
pip install -r requirements.txt
bash reproduce.sh
```

`reproduce.sh` runs, from the repository root:

```bash
python scripts/run_results.py           # every result file except the simulation tests (about 3.5 h on one core)
python scripts/paper_figures.py         # Figs. 1–6 and the Supplementary figures
python scripts/synthetic_validation.py  # simulation tests of Section 5.4 / S10 and Fig. 7 (about 4 min)
python -m pytest -q tests               # analytic limits, headline values and simulation values
```

`run_results.py` accepts one or more part names to recompute only those parts (for example
`python scripts/run_results.py pan`). Numbers are written to six significant figures, and a rerun reproduces every
result file byte for byte. All random draws use fixed seeds.

## Tests

- `tests/test_analytic.py` — analytic limits, the inversion of the shell fraction by the skin-thickness floor, the
  three cases of Proposition 3, the internal field of the cylinder and the finite-difference solver against it.
- `tests/test_claims.py` — checks the headline values of the article against the result files, each to within half a
  unit of its last quoted digit, using `claims.csv` (result file, rows, column and aggregate for each value).
- `tests/test_synthetic.py` — checks the values of Section 5.4 and Section S10 against the simulation results.

The tests check consistency between the text and the code, not the correctness of the models.

## Where each result is used

Section numbers refer to the article; S-numbers to the Supplementary material. Part names refer to
`scripts/run_results.py`.

| Result | Where reported | Part | Output files (`results/`) |
|---|---|---|---|
| Uniform-arrest diameter, Eq. (9), and its sensitivity | 4.1; Fig. 4a; Tables S4–S6 | flat | flat_diameter.csv, flat_diameter_joint_worst.csv, diffusivity_sensitivity.csv |
| Contribution of each input; global PS envelope | 4.1, 5.1.1; Table S7 | budget | flat_diameter_budget.csv, ps_window_global.csv |
| Self-consistent uniform-arrest diameter and PS validity envelope | 4.1, 5.1.1, 5.1.4; Fig. 4b | flat | flat_diameter_consistent.csv, case_study_consistent.csv |
| Arrest composition | 4.1; Fig. S3 | composition | arrest_composition.csv |
| Post-arrest inertness (A3) | 4.2; Fig. S4 | a3 | a3_post_arrest.csv, a3_flat_dependence.csv, a3_post_arrest_checks.csv, a3_sensitivity.csv, a3_inertness.csv, diffusivity_vs_data.csv, chloroform_*.csv |
| Numerical checks of Lemma 1 and A5 | 2.3; Fig. S2 | lemma1, a5 | lemma1_drying.csv, arrest_order.csv, a5_injectivity.csv, a5_conditioning.csv, a5_splitting.csv |
| Bayes factors for a committed skin | 3.2; Fig. S6 | bayes | bayes_factors_*.csv |
| Fading-memory mimicry and Propositions 3–4 | 3.3; Fig. 3; Table S3 | kernel, prop3 | kernel_*.csv, proposition3*.csv |
| Finite-extensibility check | 3.3, 3.4; Fig. S1 | finite | finite_extensibility_*.csv |
| Functional forms (Proposition 6) | 3.4; Fig. S7 | forms | functional_forms*.csv |
| Route 3 coupling and sample sizes | 4.3; Fig. S8 | coupling | coupling_*.csv, route3_sample_sizes.csv, sample_sizes.csv |
| PET channels, error model, plate control by diameter, diameter-adjusted offset | 5.2.1, 5.2.3; Fig. S10 | channels | pet_channels.csv, pet_channels_robustness.csv, pet_error_model.csv, pet_plate_check.csv, pet_offset_diameter.csv, pet_record_check.csv |
| PS skin bound and two-channel analysis | 5.1.2, 5.1.3; Fig. 6a; Table S10 | records | ps_skin_bound.csv, ps_bound_collector_share.csv, ps_common_skin.csv, ps_kappa_needed.csv, ps_record_refit.csv, ps_two_channel*.csv |
| Confocal readout weight of a thin skin | 5.1.2; Fig. 6a; Table S12 | raman, raman_check | raman_field_checks.csv, raman_weighting_kappa.csv, ps_optical_bound*.csv, raman_substrate.csv, raman_tilt.csv |
| Arrest front of the PS drying model | 5.1.2 | skinfront | ps_arrest_front.csv |
| Sampling scenarios for the PS skin bound | 5.1.2 | psunc | ps_bound_uncertainty.csv |
| PAN readout agreement, equivalence test, skin bound | 5.3.1; Fig. S11 | pan | pan_readout_agreement.csv, pan_equivalence.csv, pan_skin_bound.csv |
| Uniform-arrest diameter at span thresholds 0.05–0.2 | 4.1; S5.3 | threshold | flat_threshold_sensitivity.csv |
| PET static-plate bound | 5.2.1; Fig. 6b | pet | pet_plate_skin_bound.csv |
| Simulation tests of the protocol (V1–V3) | 5.4; Fig. 7; S10.1–S10.4 | `scripts/synthetic_validation.py` | synth_plate_control.csv, synth_route3.csv, synth_route4.csv |
| Robustness to misspecification and Monte Carlo intervals (V4) | 5.4; S10.5 | `scripts/synthetic_validation.py` | synth_robustness.csv |

File names beginning `flat_` keep the earlier name of the uniform-arrest diameter.

## Contents

```
scripts/      model functions, run_results.py, paper_figures.py, synthetic_validation.py,
              raman_weighting.py, raman_fd.py, import_author_data.py, digitization scripts
data/         supplied record values (CSV and the original workbook in data/raw/), digitized values in data/digitized/
results/      computed values (CSV)
figures/      figures as PDF (vector) and PNG
tests/        test_analytic.py, test_claims.py, test_synthetic.py
claims.csv    map from each headline value to the result file, rows, column and aggregate
reproduce.sh  one-command reproduction
```

## Licence

Code: MIT (`LICENSE`). Results, figures and data: CC BY 4.0 (`LICENSE-DATA`). When reusing the record values, please
also cite the source articles listed in `data/README.md`.

## Use of AI tools

The numerical code was developed with the assistance of a generative AI tool (Claude, Anthropic), which was also used
to cross-check the values quoted in the text against the computed result files. The author verified the code against
analytic limits and independent calculations, as described in the Data availability statement of the article.
