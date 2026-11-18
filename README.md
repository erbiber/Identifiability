# Identifiability of Molecular Orientation Mechanisms in Electrospun Polymer Fibers and Its Practical Implications — code and results

Code, computed values and figures for the paper

> Biber, E. *Identifiability of Molecular Orientation Mechanisms in Electrospun Polymer Fibers and Its Practical Implications*. Submitted to *Polymer* (2026).

Every number quoted in the paper and its Supplementary material is computed by the scripts here and written to
`results/*.csv`; every figure is drawn from those files or from the same model functions. The paper is a theory
paper: no experimental data are generated here. The published records it reanalyses (polystyrene, poly(ethylene
terephthalate), poly(2,6-dimethyl-1,4-phenylene oxide), poly(ethylene oxide) and polyoxymethylene) are cited in the
paper, and only the values taken from them enter the scripts.

All transport-dependent estimates use one working system, polystyrene–chloroform. Its solvent diffusivity is
measured in the glass below a solvent fraction of 0.104 (Crank & Park 1949, `data/crank_park_1949_table2.csv`) and
follows Vrentas–Duda free-volume theory above the glass composition φg = 0.14 (vapor-sorption desorption branch),
with the polystyrene and chloroform parameters tabulated by Zielinski & Duda (AIChE J 1992), converted to mutual
diffusion with the Flory–Huggins factor (χ = 0.085) and evaluated at 18.5 °C (`theta_r_span.make_D_chcl3`,
Section S1). The polymer parameters are checked against toluene self-diffusion (Pickup & Blum 1989,
`data/pickup_blum_1989_table1.csv`). The parameter set is selected by a glass-physics constraint: below φg the
measured glassy diffusivity must lie at or above the equilibrium free-volume curve, which the Zielinski–Duda set
satisfies (15× or more) and a viscosity-only estimate of chloroform's parameters violates from φ = 0.08 upward; the
latter is kept as one of ten diffusivity variants (`theta_r_span.VARIANTS`: ×0.5, ×2, φg = 0.13 and 0.17, dry tail
to 1e-18, viscosity-estimated parameters, jump-volume ratio ξ fitted to the solution self-diffusion data of Waggoner et
al. 1993, no thermodynamic factor, factor frozen above φ = 0.3).

## Reproducing the results

Tested with Python 3.13, numpy 2.x (the scripts use `np.trapezoid`, which needs numpy ≥ 2.0), scipy 1.18 and
matplotlib 3.11.

```bash
pip install -r requirements.txt
python scripts/run_results.py      # writes all 68 files in results/  (about 3.5 h on one core; the flat and a3 parts can run in parallel)
python scripts/paper_figures.py    # draws the figures into figures/  (about 15 s)
python scripts/toc_graphic.py      # draws the Table of Contents graphic
python tests/test_analytic.py      # analytic limits of the skin bound and of Proposition 3
python tests/test_claims.py        # every headline value in claims.csv against results/
```

Or, in one command, `bash reproduce.sh`, which runs all of the above in order and stops at the first failure.

Run the scripts from the repository root, and run `run_results.py` first: Figure 4 reads
`results/flat_diameter_consistent.csv` and Figure 6 reads `results/raman_weighting_kappa.csv`. `run_results.py` also accepts one or more result-group names to recompute only those
groups (`raman`, `skinfront`, `psunc`, `raman_check`, `channels`, `cluster`, `pan`, `threshold`, `flat`, `composition`, `a3`, `lemma1`, `a5`, `bayes`, `kernel`, `prop3`, `forms`, `coupling`, `pet`, `records`, `finite`, `budget`), for
example `python scripts/run_results.py pet`; an unknown name stops with the list of valid groups. A fresh run reproduces every file in `results/`
byte for byte.

## Contents

```
scripts/
    theta_r_span.py      model functions: polystyrene–chloroform diffusivity (measured glass + free volume), flat diameter, drying solutions (Lemma 1), A3 windows
    separability.py      model functions: fading-memory kernels, Bayes factors, functional forms, accumulation maps
    coupling.py          model functions: Route 3 coupling and the measured jet stretch
    raman_weighting.py   internal field of a dielectric cylinder under a focused beam; confocal Raman weight of a thin skin
    raman_fd.py          finite-difference check of that weight with the BaF2 substrate
    run_results.py       computes every quoted value and writes results/*.csv
    paper_figures.py     draws the figures of the article (map below)
    toc_graphic.py       draws the Table of Contents graphic
    digitize_pet_fig3b.py  extracts per-fiber PET values from the published figure (needs the PDF and PyMuPDF)
    digitize_knapp_fig3.py re-measures per-fiber PAN values from the raster Figure 3 of the immersion study (needs the PDF, PyMuPDF and Pillow)
results/                 68 CSV files, listed below with the sections that quote them
data/                    measured diffusion data (Pickup & Blum 1989, Table I; Crank & Park 1949, Table II; Waggoner et al. 1993, Table I)
                         the polystyrene orientation record digitized from Richard-Lacroix & Pellerin 2015, Figures 2A and 5A,
                         and per-fiber values of the PET record extracted from the vector graphics of Laramée, Lanthier &
                         Pellerin 2022, Figure 3b (scripts/digitize_pet_fig3b.py regenerates it from the published PDF, which is
                         not redistributed), and per-fiber PAN orientation in oil and water re-measured from Knapp, Laramée &
                         Pellerin 2026, Figure 3 (scripts/digitize_knapp_fig3.py)
figures/                 the figures of the article and the graphical abstract, as PDF (vector) and PNG (600 dpi)
tests/                   analytic-limit tests and a regression test of the headline values
claims.csv               machine-readable map from each headline value of the paper (Table 5, Table S16) to the result
                         file, rows, column and aggregate that produce it, with the value as quoted
```

## Where each result is used

Section, figure and table numbers refer to the article as submitted to *Polymer*; S-numbers to its Supplementary material.

| File | Quantity | Part | Used in |
|---|---|---|---|
| `arrest_order.csv` | arrest order for non-uniform initial compositions | `lemma1` | 2.3; S3.2; Fig. S2 |
| `lemma1_drying.csv` | numerical check of Lemma 1 (time-varying evaporation, thinning jet) | `lemma1` | 2.3; S3.1; Fig. S2 |
| `flat_diameter.csv` | flat diameter, Eq. (10), against arrest composition and flight time | `flat` | 4.1; S5.1; Tables S4, S6 |
| `diffusivity_sensitivity.csv` | flat diameter under the ten polystyrene–chloroform diffusivity forms | `flat` | 4.1; S5.1; Table S5 |
| `flat_diameter_joint_worst.csv` | flat diameter with all inputs varied jointly | `flat` | 4.1; S5.1; Table S4; Fig. 4a |
| `flat_diameter_consistent.csv` | self-consistent flat diameter with mode-dependent arrest | `flat` | 4.1; 5.1.1; S5.3; Fig. 4b; Table 5 |
| `case_study_consistent.csv` | polystyrene validity windows and test thresholds | `flat` | 5.1.4; S8.5; Fig. 4b |
| `arrest_composition.csv` | arrest composition of polystyrene–chloroform from sorption data | `composition` | 4.1; S4.1; Fig. S3 |
| `a3_inertness.csv` | first estimate of the diameter for post-arrest inertness (A3) | `a3` | S4.2 |
| `a3_post_arrest.csv` | A3 limit from the drying equation integrated past arrest | `a3` | 4.2; 5.1.1; S4.3; Fig. 4b; Fig. S4; Table 5 |
| `diffusivity_vs_data.csv` | toluene self-diffusion in polystyrene against the Vrentas–Duda fit that supplies the polymer free-volume parameters | `a3` | S5.2; Fig. S5 |
| `chloroform_diffusivity.csv` | chloroform free-volume parameters estimated from viscosity and molar volume (the viscosity-estimated diffusivity form) | `a3` | S5.2 |
| `chloroform_crank_park.csv` | chloroform diffusion in glassy polystyrene at 25 °C against the free-volume curves and the diffusivity forms; the glass-constraint check | `a3` | S5.2; Fig. S5 |
| `chloroform_wet_end_check.csv` | chloroform and toluene self-diffusion in polystyrene solutions against the free-volume self-diffusivity | `a3` | S5.2 |
| `a3_flat_dependence.csv` | how closely the A3 limit follows the flat diameter | `a3` | 4.2; S4.3; Fig. S4 |
| `a3_post_arrest_checks.csv` | robustness of the A3 limit: sub-Tg law, evaporation rate, WLF constants, grid | `a3` | S4.3 |
| `a3_sensitivity.csv` | sensitivity of the A3 limit to its inputs | `a3` | S4.3 |
| `a5_injectivity.csv` | injectivity of the accumulation map (A5, Lemma 2) | `a5` | 2.3; S3.3 |
| `a5_conditioning.csv` | conditioning of the accumulation map: Jacobian condition numbers, closest approach of distant elements, inverse Lipschitz bound | `a5` | 3.1; S3.3 |
| `a5_splitting.csv` | loss of injectivity when jets split | `a5` | 2.3; S3.4 |
| `bayes_factors_prior_width.csv` | Bayes factor against prior width | `bayes` | 3.2; S6.1; Fig. S6 |
| `bayes_factors_commitment_error.csv` | Bayes factor against error in the committed skin level | `bayes` | 3.2; S6.1; Fig. S6; Table 5 |
| `bayes_factors_noise.csv` | Bayes factor against noise and number of points | `bayes` | S6.1 |
| `kernel_skin_match.csv` | fading-memory profiles matched to smooth skin profiles | `kernel` | 3.3; Table 2; Table 5 |
| `kernel_spectra.csv` | the same match for multimode and stretched-exponential kernels | `kernel` | 3.3; Table 5 |
| `kernel_timing.csv` | position of the maximum against stretching and arrest timing | `kernel` | S2.2; Table S3 |
| `proposition3_kernels.csv` | Proposition 3 checked for several kernels | `prop3` | 3.3; Fig. 3 |
| `proposition3b_bound.csv`, `proposition3b_clock.csv`, `proposition3b_scaling.csv` | Proposition 4 with a composition-dependent clock (48 cases) | `prop3` | 3.3; S2.3; Table 5 |
| `functional_forms.csv`, `functional_forms_crossings.csv`, `functional_forms_separation.csv`, `functional_forms_sampling.csv` | shell fraction against the best-fitting exponential; crossings; separation under monotone maps; fibers needed | `forms` | 3.4; S6.2; Fig. S7 |
| `functional_forms_free_delta.csv` | the same fit with free shell thickness | `forms` | S6.2; S8.6 |
| `sample_sizes.csv` | fibers needed for a second structural channel (Route 2) | `coupling` | 4.3; S7.1 |
| `coupling_beta.csv`, `coupling_check.csv`, `coupling_jet.csv` | Route 3 coupling, Eqs. (11)–(12), its check by direct integration and the measured jet stretch | `coupling` | 4.3; S7.2; Fig. S8 |
| `route3_sample_sizes.csv` | fibers needed for Route 3 | `coupling` | 4.3; S7.3 |
| `pet_plate_skin_bound.csv` | bound on a flight-set skin from the PET static-plate control | `pet` | 5.2.1; S8.6; Fig. 6b; Table 5 |
| `ps_skin_bound.csv`, `ps_record_refit.csv` | committed thin skin against the digitized polystyrene record; refit of the published exponential | `records` | 5.1.2; S8.3; Table S10; Fig. 6a; Table 5 |
| `ps_common_skin.csv`, `ps_kappa_needed.csv` | one skin thickness common to all fibers of 0.5-1.7 um by core level, skin orientation and readout weighting; weighting needed for a 13.1 or 30 nm skin | `records` | 5.1.2; S8.3; Table 5 |
| `ps_two_channel.csv`, `ps_two_channel_discordant.csv` | joint analysis of the two Raman channels of the polystyrene record (bin means) | `records` | 5.1.3; S8.3 |
| `pet_channels.csv`, `pet_channels_robustness.csv`, `pet_record_check.csv` | per-fiber PET channels: offsets at equal orientation, matched comparison, test of a common function, robustness, check against the source | `channels` | 5.2.3; S8.7; Fig. S10; Table 5 |
| `pet_error_model.csv` | error model of the PET crystallinity offset: band-width equivalent, errors in <P2>, added noise | `channels` | 5.2.3; S8.7 |
| `pan_readout_agreement.csv`, `pan_skin_bound.csv` | PAN immersion record: oil and water four-spectrum against calibrated orientation; thinnest skin supplying the stated median rise | `pan` | 5.3.1; S8.8; Fig. S11; Table 5 |
| `raman_field_checks.csv`, `raman_weighting_kappa.csv`, `ps_optical_bound.csv` | confocal Raman weight of a thin skin (field checks, kappa by optical setting) and the polystyrene skin bound under it | `raman` | 5.1.2; S8.4; Table S12; Fig. 6a; Table 5 |
| `raman_substrate.csv`, `raman_tilt.csv`, `ps_optical_bound_substrate.csv` | checks of the confocal weight: finite differences with and without the BaF2 substrate, beam tilt along the fiber axis, and the PS bound with the substrate | `raman_check` | 5.1.2; S8.4; Table 5 |
| `ps_arrest_front.csv` | arrest front of the PS drying model: time after surface arrest at which 35 and 140 nm have arrested | `skinfront` | 5.1.2; S8.3; Table 5 |
| `ps_bound_uncertainty.csv` | sampling uncertainty of the volume-average PS skin bound (bin means resampled for 6, 9 and 15 fibers per bin) | `psunc` | 5.1.2; S8.3 |
| `pet_cluster_sensitivity.csv` | design effect and within-session correlation at which the PET crystallinity offset loses significance (run after channels) | `cluster` | 5.2.3; S8.7 |
| `pan_equivalence.csv` | two one-sided tests of oil against calibrated PAN readouts; orientation dependence of the difference | `pan` | 5.3.1; S8.8 |
| `flat_threshold_sensitivity.csv` | flat diameter at span thresholds 0.05, 0.1, 0.2 (central inputs) | `threshold` | 4.1; S5.3; Table 5 |
| `finite_extensibility_profiles.csv`, `finite_extensibility_saturation.csv` | Proposition 3 with FENE-P chains and Kuhn–Grün orientation; saturation against Weissenberg number | `finite` | 3.3; 3.4; S2.5; Fig. S1; Table 5 |
| `flat_diameter_budget.csv`, `ps_window_global.csv` | contribution of each input to the flat-diameter range; how often the polystyrene window is open (run after flat and a3) | `budget` | 4.1; 5.1.1; S5.3; S8.5; Table S7 |

Figure files and their numbers in the article: `fig1_clocks_degeneracy` Fig. 1; `fig2_hierarchy` Fig. 2;
`fig3_where_structure_peaks` Fig. 3; `fig4_flat_windows` Fig. 4; `fig5_route_selection` Fig. 5; `fig6_skin_bounds` Fig. 6;
`figS9_finite_extensibility` Fig. S1; `figS3_a3_vs_flat` Fig. S4; `figS4_diffusivity` Fig. S5; `figS5_bayes` Fig. S6;
`figS6_functional_forms` Fig. S7; `figS7_coupling_regimes` Fig. S8; `figS8_analysis_map` Fig. S9; `figS10_pet_channels`
Fig. S10; `figS11_pan_readout` Fig. S11; `graphical_abstract` the graphical abstract. Figs. S2 and S3 are drawn by the manuscript build from the same result files.

## Licence

Code (`scripts/`): MIT licence, see `LICENSE`. Computed values (`results/`) and figures (`figures/`): Creative
Commons Attribution 4.0 International (CC BY 4.0), see `LICENSE-DATA`.

## Citation

If you use this code or these results, please cite the paper and the archived version of this repository (see
`CITATION.cff`).
