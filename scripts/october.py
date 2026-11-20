"""Model checks of Sections 2-3 of the article and S2-S3, S6-S7 of the Supplementary material, carried over unchanged
from the October 2026 release (their outputs are byte-identical): Lemma 1, A5, Bayes factors, fading-memory kernels,
Propositions 3-4, functional forms, Route 3 coupling and the arrest composition."""
import math

import numpy as np
from scipy.optimize import brentq, curve_fit
from scipy.special import expit

import coupling as cpl
import lemma1
import separability as sep
from common import write


def part_composition():
    rows = [('absorption', 0.40, chi, lemma1.flory_solvent_fraction(0.40, chi)) for chi in (0.05, 0.12)]
    rows += [('desorption', a, chi, lemma1.flory_solvent_fraction(a, chi)) for a in (0.33, 0.35, 0.36) for chi in (0.05, 0.12)]
    write('arrest_composition.csv', ['branch', 'solvent_activity', 'chi', 'solvent_volume_fraction'], rows)


def part_lemma1():
    rows = [('uniform start', *lemma1.lemma1_drying_check(), '')]
    rows.append(('polymer-rich core 0.25', *lemma1.lemma1_drying_check(lambda r: np.where(r < 0.3, 0.25, 0.9)), ''))
    rows.append(('solvent-rich annulus', *lemma1.lemma1_drying_check(lambda r: 0.3 + 0.6 * np.exp(-((r - 0.6) / 0.12) ** 2)), ''))
    for lab, kf, rf in (('evaporation on/off', lambda t: 5.0 if int(t / 2) % 2 == 0 else 0.0, lambda t: 1.0),
                        ('oscillating evaporation', lambda t: 5.0 * (1 + 0.95 * np.sin(3 * t)), lambda t: 1.0),
                        ('thinning jet', lambda t: 5.0, lambda t: np.exp(-min(t, 40) / 4))):
        rise_t, tfv, thv, rewet = lemma1.lemma1_time_dependent_law(kf, rf)
        rows.append((lab, '', rise_t, tfv, thv, rewet))
    write('lemma1_drying.csv', ['case', 'max_rise_in_r', 'max_rise_in_t', 't_f_order_violations', 'theta_r_order_violations', 'elements_rewetted'], rows)
    write('arrest_order.csv', ['axial_solvent_deficit_A', 'evaporation_k', 'arrest_order'],
          [(a, k, lemma1.arrest_order_with_rich_centre(a, k)) for a in (0.05, 0.2, 0.4, 0.6, 0.68, 0.72) for k in (0.5, 50.0)])


def part_a5():
    rows = [('common clock', l0, tau, *sep.a5_element_injectivity(l0, tau)) for l0, tau in ((0.5, 0.3), (0.2, 1.0), (1.0, 0.1), (0.05, 3.0))]
    rows.append(('depth-dependent clock', '', '', *sep.a5_depth_dependent_clock()))
    write('a5_injectivity.csv', ['clock', 'lam0', 'tau', 'jacobian_positive', 'jacobian_negative', 'shared_pairs'], rows)
    write('a5_splitting.csv', ['initial_radius_spread', 'median_label_distance', 'p90_label_distance'],
          [(s, *sep.a5_label_ambiguity(s)) for s in (0.0, 0.05, 0.1, 0.2, 0.3)])


def part_bayes():
    widths = (('half', (0.5, 1.25), (-0.25, 0.75)), ('stated', (0.0, 1.5), (-0.5, 1.0)), ('double', (-0.75, 2.25), (-1.25, 1.75)))
    write('bayes_factors_prior_width.csv', ['prior', 'upper_lo', 'upper_hi', 'lower_lo', 'lower_hi', 'range_product', 'bayes_factor_direct', 'bayes_factor_savage_dickey'],
          [(lab, up[0], up[1], lo[0], lo[1], (up[1] - up[0]) * (lo[1] - lo[0]), sep.containment_bayes_factor(upper=up, lower=lo), sep.savage_dickey_bf(up, lo)) for lab, up, lo in widths])
    write('bayes_factors_commitment_error.csv', ['error_in_committed_upper_level', 'bayes_factor'],
          [(round(e, 3), sep.committed_levels_bf(1.0 + e, 0.0)) for e in np.linspace(0, 0.12, 13)])
    write('bayes_factors_noise.csv', ['sigma', 'radial_points', 'bayes_factor'],
          [(sg, n, sep.containment_bayes_factor(sigma=sg, npts=n)) for sg in (0.05, 0.08, 0.1, 0.15) for n in (30, 60, 120)])


def part_kernel():
    write('kernel_skin_match.csv', ['stretching', 't0', 'w', 'lam', 'rms_to_best_skin'],
          [(lab, t0, w, lam, sep.realistic_skin_match(t0, w, lam)) for lab, t0, w in (('concentrated', 0.3, 0.05), ('moderate', 1.0, 1.0), ('sustained', 5.0, 5.0), ('very sustained', 10.0, 20.0)) for lam in (0.1, 1.0)])
    rows = []
    for ratio in (5, 20, 100):
        for lab, t0, w in (('complete before', 0.3, 0.05), ('moderate', 1.0, 1.0), ('continuing after', 5.0, 5.0)):
            s, c = sep.kernel_orientation_fit(ratio, t0, w)
            rows.append((ratio, lab, t0, w, s, c, sep.kernel_max_position(ratio, t0, w)))
    write('kernel_timing.csv', ['core_to_surface_arrest_ratio', 'stretching', 't0', 'w', 'rms_to_skin', 'rms_to_dense_core', 'max_position_r_over_R'], rows)
    rows = []
    for name, mem in sep.MEMORIES.items():
        sb, rb = sep.spectrum_robustness(mem, t0=0.3, w=0.05)
        sa, ra = sep.spectrum_robustness(mem, t0=5.0, w=5.0)
        rows.append((name, sb, rb, sa, ra))
    write('kernel_spectra.csv', ['memory', 'before_rms_to_skin', 'before_max_position', 'after_rms_to_skin', 'after_max_position'], rows)


def part_prop3():
    a = sep.proposition4_check()
    c = sep.proposition4c_check()
    write('proposition3_kernels.csv', ['kernel', 'a_monotone_inward', 'a_max_position', 'b_max_position_skin_decoupled', 'c_max_position_skin_drawn'],
          [(k, a[k][0], a[k][1], a[k][2], c[k][1] if k in c else '') for k in a])
    write('proposition3b_clock.csv', ['r_stop', 'lam0_over_t_end', 'clock_strength_a', 'max_position_r_over_R'],
          [(rs, lf, av, pos) for rs, lf, poss in sep.proposition3b_clock_check() for av, pos in zip((0, 1, 2, 4, 8), poss)])
    write('proposition3b_scaling.csv', ['lam_wet_over_surface_arrest_time', 'clock_strength_a', 'interior_maximum', 'criterion_wet_predicts_interior', 'criterion_arrest_predicts_interior'],
          sep.proposition3b_scaling_check())
    write('proposition3b_bound.csv', ['family', 'r_stop', 'lam_wet_over_surface_arrest_time', 'clock_strength_a', 't_f_of_element', 'M_element', 'lower_bound', 'M_surface', 'upper_bound', 'lower_bound_holds', 'upper_bound_holds', 'theorem_condition_met', 'interior_maximum'],
          sep.proposition3b_bound_check())


def part_forms():
    rows = []
    for delta in (20.0, 30.0, 40.0):
        d = np.linspace(*sep.WINDOW, 400)
        f = sep.shell_fraction(d, delta)
        p, _ = curve_fit(sep.exponential, d, f, p0=[f[0], 500.0, 0.0], maxfev=100000)
        res = f - sep.exponential(d, *p)
        r2 = 1.0 - np.sum(res ** 2) / np.sum((f - f.mean()) ** 2)
        rms = float(np.sqrt(np.mean(res ** 2)))
        rows.append((delta, float(r2), float(p[0]), float(p[1]), float(p[2]), float(np.abs(res).max() / np.ptp(f)), rms,
                     *[sep.Z ** 2 * s ** 2 / rms ** 2 for s in (0.05, 0.10, 0.15)]))
    write('functional_forms.csv', ['delta_nm', 'R2', 'A', 'd_c_nm', 'b', 'max_residual_fraction_of_range', 'rms_residual', 'n_total_dispersion_0.05', 'n_total_dispersion_0.10', 'n_total_dispersion_0.15'], rows)
    r2, delta, dc = sep.free_delta()
    write('functional_forms_free_delta.csv', ['best_R2', 'delta_nm', 'd_c_nm'], [(float(r2), float(delta), float(dc))])
    xs, ys, _ = sep.crossings(30.0)
    write('functional_forms_crossings.csv', ['crossing_diameter_nm', 'common_value'], [(float(x), float(y)) for x, y in zip(xs, ys)])
    d = np.linspace(*sep.WINDOW, 400)
    f = sep.shell_fraction(d, 30.0)
    p, _ = curve_fit(sep.exponential, d, f, p0=[f[0], 500.0, 0.0], maxfev=200000)
    d0 = brentq(lambda x: sep.shell_fraction(x, 30.0) - sep.exponential(x, *p), 300, 700)
    s0 = float(sep.shell_fraction(d0, 30.0))
    dd = np.linspace(*sep.WINDOW, 20001)
    ff, gg, qq = sep.shell_fraction(dd, 30.0), sep.exponential(dd, *p), sep.shell_fraction(1.3 * dd, 30.0)

    def rel(a, b, m):
        ga, gb = m(a), m(b)
        return float(np.sqrt(np.trapezoid((ga - gb) ** 2, dd) / np.ptp(dd)) / np.ptp(ga))
    rows = []
    for k in np.logspace(0, 5, 11):
        cross = rel(ff, gg, lambda z: expit(k * (z - s0)))
        nocross = min(rel(ff, qq, lambda z: expit(k * (z - t))) for t in np.linspace(ff.min() + 0.005, ff.max() - 0.005, 80))
        rows.append((float(k), cross, nocross))
    write('functional_forms_separation.csv', ['map_steepness_k', 'relative_separation_crossing_pair', 'relative_separation_noncrossing_pair'], rows)


def part_coupling():
    rows = [('Eq 11, stretching until arrest', 2.7, 2.0, 'Wi', wi, cpl.beta_stretch_to_arrest(2.7, wi)) for wi in (10, 30, 100, 300, 1000)]
    rows += [('Eq 11, stretching until arrest', L, 2.0, 'Wi', 100, cpl.beta_stretch_to_arrest(L, 100)) for L in (2.0, 4.0)]
    rows += [('Eq 12, relaxation before arrest', 2.7, 2.0, 'dt_over_lambda', x, cpl.beta_relax_before_arrest(2.7, x)) for x in (0.1, 0.2, 0.5, 1.0)]
    write('coupling_beta.csv', ['equation', 'L', 'rho', 'variable', 'value', 'beta'], rows)
    write('coupling_check.csv', ['tau', 'beta_eq11', 'beta_direct_integration'],
          [(tau, cpl.beta_stretch_to_arrest(2.7, 2 * 2.7 / tau), cpl.beta_check_by_integration(2.7, tau)) for tau in (0.054, 0.18, 0.54)])
    L, frac = cpl.stretch_from_measured_jet()
    write('coupling_jet.csv', ['L', 'fraction_of_diameter_reduction_due_to_stretch'], [(float(L), float(frac))])
    rows = []
    for regime, beta in (('stretching until arrest, Wi = 100 (Eq 11)', cpl.beta_stretch_to_arrest(2.7, 100)),
                         ('relaxation before arrest, dt/lambda = 0.1 (Eq 12)', cpl.beta_relax_before_arrest(2.7, 0.1)),
                         ('relaxation before arrest, dt/lambda = 0.3 (Eq 12)', cpl.beta_relax_before_arrest(2.7, 0.3)),
                         ('relaxation before arrest, dt/lambda = 1 (Eq 12)', cpl.beta_relax_before_arrest(2.7, 1.0))):
        delta = beta * math.log(2.0)
        for sd in (0.1, 0.2, 0.3):
            rows.append((regime, beta, delta, sd, 2 * sep.Z ** 2 * (sd / delta) ** 2))
    write('route3_sample_sizes.csv', ['regime', 'beta', 'delta_ln_diameter', 'sigma_ln_diameter_at_fixed_signal', 'n_per_group'], rows)
    z = sep.Z
    write('sample_sizes.csv', ['quantity', 'difference_over_dispersion', 'fibers'],
          [('phase-separation comparison, per group', 1.0, 2 * z ** 2), ('phase-separation comparison, per group', 0.5, 2 * z ** 2 * 4)])
