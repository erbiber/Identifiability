# Identifiability of Molecular Orientation Mechanisms in Electrospun Polymer Fibers — code and results

Code, computed values and figures for the paper

> Biber, E. *Identifiability of Molecular Orientation Mechanisms in Electrospun Polymer Fibers*. Submitted to *Macromolecules* (2026).

Every number quoted in the paper and its Supporting Information is computed by the scripts here and written to
`results/*.csv`; every figure is drawn from those files or from the same model functions. The paper is a theory
paper: no experimental data are generated here. The published records it reanalyses (polystyrene, poly(ethylene
terephthalate), poly(2,6-dimethyl-1,4-phenylene oxide), poly(ethylene oxide) and polyoxymethylene) are cited in the
paper, and only the values taken from them enter the scripts.

## Reproducing the results

Tested with Python 3.13, numpy 2.x (the scripts use `np.trapezoid`, which needs numpy ≥ 2.0), scipy 1.18 and
matplotlib 3.11.

```bash
pip install -r requirements.txt
python scripts/run_results.py      # writes all 32 files in results/  (about 5–6 min on one core)
python scripts/paper_figures.py    # draws Figures 1–9 into figures/  (about 15 s)
python scripts/toc_graphic.py      # draws the Table of Contents graphic
```

Run the scripts from the repository root, and run `run_results.py` first: Figure 8 reads
`results/flat_diameter_consistent.csv`. `run_results.py` also accepts one or more result-group names to recompute only those
groups (`flat`, `composition`, `a3`, `lemma1`, `a5`, `bayes`, `kernel`, `prop3`, `forms`, `coupling`, `pet`), for
example `python scripts/run_results.py pet`; an unknown name stops with the list of valid groups. A fresh run reproduces every file in `results/`
byte for byte.

## Contents

```
scripts/
    theta_r_span.py      model functions: free-volume diffusivity, flat diameter, drying solutions (Lemma 1), A3 windows
    separability.py      model functions: fading-memory kernels, Bayes factors, functional forms, accumulation maps
    coupling.py          model functions: Route 3 coupling and the measured jet stretch
    run_results.py       computes every quoted value and writes results/*.csv
    paper_figures.py     draws Figures 1–9
    toc_graphic.py       draws the Table of Contents graphic
results/                 32 CSV files, listed below with the sections that quote them
figures/                 Figures 1–9 and the TOC graphic, as PDF (vector) and PNG (600 dpi)
```

## Where each result is used

Section numbers refer to the main text; S-numbers to the Supporting Information.

| File | Quantity | Sections |
|---|---|---|
| `arrest_order.csv` | arrest order for non-uniform initial composition | 3.2 |
| `lemma1_drying.csv` | numerical check of Lemma 1 (time-varying evaporation, thinning jet) | S3.1 |
| `flat_diameter.csv` | flat diameter, Equation (7), against arrest composition and flight time | 3.3; S1 (Tables S1–S2) |
| `diffusivity_sensitivity.csv` | flat diameter under variation of the transferred diffusivity | 3.3 (Table 3); S1 |
| `flat_diameter_joint_worst.csv` | flat diameter with all inputs varied jointly | 3.3 (Table 3); 7.1.2 |
| `flat_diameter_consistent.csv` | flat diameter from the consistent drying calculation, mode-dependent arrest | 3.3; 6.1.1; Figure 8 |
| `arrest_composition.csv` | arrest composition of polystyrene–chloroform from sorption data | S3.6 |
| `a3_inertness.csv` | upper diameter for post-arrest inertness (A3) | 3.4; 7.1.2; S3.7 |
| `a3_sensitivity.csv` | sensitivity of the A3 bound to its inputs | 3.4 |
| `a5_injectivity.csv` | injectivity of the accumulation map (A5, Lemma 2) | 3.5 |
| `a5_splitting.csv` | loss of injectivity when jets split | 3.5; S3.11 |
| `bayes_factors_commitment_error.csv` | Bayes factor against error in the committed skin level | 4.3; Figure 3b |
| `bayes_factors_prior_width.csv` | Bayes factor against prior width | 4.3; S3.10 |
| `bayes_factors_noise.csv` | Bayes factor against noise and number of points | S3.10 |
| `kernel_skin_match.csv` | fading-memory profiles matched to skin profiles | 4.4; Figure 3a |
| `kernel_spectra.csv` | the same match for multimode and stretched-exponential kernels | 4.4 |
| `kernel_timing.csv` | position of the maximum against stretching and arrest timing | S3.2 (Table S3) |
| `proposition3_kernels.csv` | Proposition 3 checked for five kernels | 4.4; Figure 4 |
| `proposition3b_bound.csv`, `proposition3b_clock.csv`, `proposition3b_scaling.csv` | Proposition 3(b) and 4 with a composition-dependent clock | 4.4 |
| `functional_forms.csv` | shell fraction against best-fitting exponential | 4.6; Figure 5 |
| `functional_forms_crossings.csv` | crossing diameters of the two forms | 4.6; Figure 5a |
| `functional_forms_separation.csv` | separation under monotone maps of increasing steepness | 4.6; Figure 5c |
| `functional_forms_free_delta.csv` | the same fit with free shell thickness | S3.3 |
| `sample_sizes.csv` | fibers needed for a second structural channel | 5.1 |
| `coupling_beta.csv` | Route 3 coupling, Equations (11)–(12) | 5.2; Figure 6 |
| `coupling_check.csv` | Equations (11)–(12) checked by direct integration | 5.2; S3.9 |
| `coupling_jet.csv` | jet stretch L from the measured polyamic-acid jet, Equation (8) | 5.2; S3.9 |
| `route3_sample_sizes.csv` | fibers needed for Route 3 | 5.2 |
| `case_study_consistent.csv` | polystyrene validity windows and test thresholds | 6.1.1; 6.1.4; S3.13; Figure 8 |
| `pet_plate_skin_bound.csv` | bound on a flight-set skin from the poly(ethylene terephthalate) plate control | 6.2.1; S3.17; Figure 9 |

## Licence

Code (`scripts/`): MIT licence, see `LICENSE`. Computed values (`results/`) and figures (`figures/`): Creative
Commons Attribution 4.0 International (CC BY 4.0), see `LICENSE-DATA`.

## Citation

If you use this code or these results, please cite the paper and the archived version of this repository (see
`CITATION.cff`).
