"""Result parts for the polystyrene-chloroform model (called by scripts/run_results.py):
flat (Eq. 9 and the diffusivity), drying (self-consistent uniform-arrest diameter, A3, envelope, case study, budget),
drying_checks (evaporation rate, relaxation law, grid), skinfront (arrest front), composition (arrest composition)."""
import math
from concurrent.futures import ProcessPoolExecutor
import os

import numpy as np

import diffusivity as dfm
import ps_drying as pd
from common import read, write

CENTRAL_SHIFT = 6.0
WORKERS = max(1, min(2, os.cpu_count() or 1))


# --------------------------------------------------------------------------------------------- Eq. (9)
def part_flat():
    base = dfm.Diffusivity('baseline')
    write('flat_diameter.csv', ['phi_star', 'flight_ms', 'flat_diameter_nm'],
          [(p, ms, dfm.eq9_diameter_nm(float(base(p)), ms / 1000)) for p in (0.14, 0.17, 0.20, 0.23)
           for ms in (1, 10, 30, 100)])
    Ds = {v: dfm.Diffusivity(v) for v in dfm.VARIANTS}
    write('diffusivity_sensitivity.csv', ['variation', 'D_s_at_0.17_m2_per_s', 'flat_nm_0.14', 'flat_nm_0.17',
                                          'flat_nm_0.20'],
          [(v, float(D(0.17)), *[dfm.eq9_diameter_nm(float(D(p)), 0.01) for p in (0.14, 0.17, 0.20)])
           for v, D in Ds.items()])
    write('flat_diameter_joint_worst.csv', ['phi_star', 'joint_worst_flat_diameter_nm', 'variant'],
          [(p, *max((dfm.eq9_diameter_nm(float(D(p)), t), v) for v, D in Ds.items() for t in (0.001, 0.01, 0.03, 0.1)))
           for p in (0.14, 0.17, 0.20, 0.23)])
    grid = np.round(np.linspace(0.0, 0.6, 121), 4)
    write('chloroform_diffusivity.csv', ['phi', *dfm.VARIANTS, 'free_volume_25C', 'free_volume_viscosity_25C',
                                         'free_volume_xi_fitted_25C'],
          [(p, *[float(Ds[v](p)) for v in dfm.VARIANTS], float(dfm.free_volume_mutual(max(p, 1e-6), dfm.T_25)),
            float(dfm.free_volume_mutual(max(p, 1e-6), dfm.T_25, dfm.CHCL3_VISC)),
            float(dfm.free_volume_mutual(max(p, 1e-6), dfm.T_25, xi=dfm.xi_fitted()))) for p in grid])
    # glassy data against the free-volume curves (25 °C) and the join
    phi, D = dfm.crank_park()
    rows = [('measured', p, d, float(dfm.free_volume_mutual(p, dfm.T_25)),
             d / float(dfm.free_volume_mutual(p, dfm.T_25)),
             d / float(dfm.free_volume_mutual(p, dfm.T_25, dfm.CHCL3_VISC)),
             d / float(dfm.free_volume_mutual(p, dfm.T_25, xi=dfm.xi_fitted()))) for p, d in zip(phi, D)]
    rows.append(('Park law slope (ln D per unit phi)', '', dfm.park_slope(), '', '', '', ''))
    rows.append(('xi, Zielinski-Duda', '', dfm.XI_ZD, '', '', '', ''))
    rows.append(('xi, fitted to chloroform self-diffusion', '', dfm.xi_fitted(), '', '', '', ''))
    rows.append(('pure chloroform self-diffusivity 25 C', '', float(dfm.self_diffusivity_mass(1.0, dfm.T_25)), '', '',
                 '', ''))
    rows.append(('hole free volume ratio viscosity/ZD at 18.5 C', '',
                 dfm.CHCL3_VISC['K11'] * (dfm.CHCL3_VISC['K21mTg1'] + dfm.T_JET) /
                 (dfm.CHCL3['K11'] * (dfm.CHCL3['K21mTg1'] + dfm.T_JET)), '', '', '', ''))
    write('chloroform_crank_park.csv', ['row', 'solvent_volume_fraction', 'D_m2_s', 'D_free_volume_25C',
                                        'ratio_to_free_volume', 'ratio_to_viscosity_variant', 'ratio_to_xi_fitted'],
          rows)
    # the solution end: NMR self-diffusion at 25 °C [S16]
    rows = []
    xi_f = dfm.xi_fitted()
    for x in read('waggoner_1993_table1.csv'):
        w1 = 1 - float(x['wt_percent_PS']) / 100
        d = float(x['D_self_1e-10_m2_per_s']) * 1e-10
        if x['solvent'] == 'chloroform':
            rows.append(('chloroform', 1 - w1, float(dfm.volume_fraction_solvent(w1)), d,
                         float(dfm.self_diffusivity_mass(w1, dfm.T_25)) / d,
                         float(dfm.self_diffusivity_mass(w1, dfm.T_25, dfm.CHCL3_VISC)) / d,
                         float(dfm.self_diffusivity_mass(w1, dfm.T_25, xi=xi_f)) / d))
        else:
            rows.append(('toluene', 1 - w1, '', d,
                         float(dfm.self_diffusivity_mass(w1, dfm.T_25, dfm.TOLUENE, dfm.TOLUENE['xi'])) / d, '', ''))
    write('chloroform_wet_end_check.csv', ['solvent', 'w_PS', 'solvent_volume_fraction', 'D_self_measured',
                                           'model_over_data_ZD', 'model_over_data_viscosity', 'model_over_data_xi_fit'],
          rows)
    rows = []
    for x in read('pickup_blum_1989_table1.csv'):
        w1 = 1 - float(x['w_PS'])
        T = 273.15 + float(x['T_C'])
        d = float(x['D_self_1e-10_m2_per_s']) * 1e-10
        rows.append((float(x['w_PS']), float(x['T_C']), d,
                     float(dfm.self_diffusivity_mass(w1, T, dfm.TOLUENE, dfm.TOLUENE['xi']))))
    rel = np.array([m / d - 1 for _, _, d, m in rows])
    rows.append(('rms relative deviation', '', '', float(np.sqrt(np.mean(rel ** 2)))))
    write('diffusivity_vs_data.csv', ['w_PS', 'T_C', 'D_measured', 'D_free_volume'], rows)
    # first estimate of inertness (Section S4.2)
    core = 1.0 / 2.405 ** 2
    first = lambda tau, p, sk, D: 2e6 * math.sqrt(0.1 * tau * float(D(p * sk)) / core)
    write('a3_inertness.csv', ['phi_star', 'skin_fraction_of_phi_star', 'alpha_channel_upper_diameter_um',
                               'network_orientation_upper_diameter_um'],
          [(p, sk, first(pd.TAU_ALPHA_TG, p, sk, Ds['baseline']), first(pd.MODES['network'] * pd.TAU_ALPHA_TG, p, sk,
                                                                          Ds['baseline']))
           for p in (0.14, 0.17, 0.20) for sk in (0.25, 0.5)])


# --------------------------------------------------------------------------------- self-consistent model
def _scan_variant(variant):
    combos = [(t, m, pe, s) for t in pd.FLIGHTS for m in pd.MAIN_MODES for pe in pd.PHI_EQ for s in pd.SHIFTS]
    combos += [(t, 'network', pe, s) for t in pd.CASE_FLIGHTS for pe in pd.PHI_EQ for s in pd.SHIFTS]
    combos = list(dict.fromkeys(combos))
    return variant, pd.scan(variant, combos=combos)


def all_scans():
    with ProcessPoolExecutor(WORKERS) as ex:
        return dict(ex.map(_scan_variant, dfm.VARIANTS))


def d_ua(scan, t, m, pe, s, thr=pd.THRESHOLD):
    return pd.crossing_diameter_nm(t, scan[(t, m, pe, s)][1], thr)


def d_a3(scan, t, m, pe, s):
    return pd.crossing_diameter_nm(t, scan[(t, m, pe, s)][2], pd.KEEP, increasing_with_d=False)


def part_drying():
    S = all_scans()
    base = S['baseline']
    # self-consistent uniform-arrest diameter (alpha and network) and A3 for every mode
    rows, a3rows = [], []
    for v in dfm.VARIANTS:
        for t in pd.FLIGHTS:
            for m in pd.MAIN_MODES:
                for pe in pd.PHI_EQ:
                    for s in pd.SHIFTS:
                        ps = S[v][(t, m, pe, s)][0]
                        du, da = d_ua(S[v], t, m, pe, s), d_a3(S[v], t, m, pe, s)
                        if m in ('alpha', 'network'):
                            rows.append((v, t * 1000, m, pe, s, ps, du))
                        kept = [pd.value_at_diameter(t, S[v][(t, m, pe, s)][2], dd) for dd in (500.0, 1000.0, 2500.0)]
                        a3rows.append((v, t * 1000, m, pe, s, ps, du, da, da / du, *kept))
    write('flat_diameter_consistent.csv', ['diffusivity_variant', 'flight_ms', 'mode', 'phi_eq',
                                           'Tg_shift_K_per_vol_percent', 'phi_star_mode', 'flat_diameter_nm'], rows)
    write('a3_post_arrest.csv', ['diffusivity_variant', 'flight_ms', 'mode', 'phi_eq', 'Tg_shift_K_per_vol_percent',
                                 'phi_star_mode', 'flat_diameter_nm', 'd_nm_10pct_volume_loss', 'ratio_to_flat',
                                 'kept_at_0.5um', 'kept_at_1um', 'kept_at_2.5um'], a3rows)
    # comparison with Eq. (9) at the same arrest composition
    rows = []
    for v in dfm.VARIANTS:
        D = dfm.Diffusivity(v)
        for t in pd.FLIGHTS:
            for m in ('alpha', 'network'):
                for pe in pd.PHI_EQ:
                    for s in pd.SHIFTS:
                        ps = S[v][(t, m, pe, s)][0]
                        du = d_ua(S[v], t, m, pe, s)
                        rows.append((v, t * 1000, m, pe, s, ps, du, dfm.eq9_diameter_nm(float(D(ps)), t)))
    write('flat_vs_eq9.csv', ['diffusivity_variant', 'flight_ms', 'mode', 'phi_eq', 'Tg_shift_K_per_vol_percent',
                              'phi_star_mode', 'flat_diameter_nm', 'eq9_diameter_nm'], rows)
    # dependence of the A3 limit on the uniform-arrest diameter (alpha and network, 480 cases)
    pts = [(r[0], r[1], r[2], r[3], r[4], r[6], r[7]) for r in a3rows if r[2] in ('alpha', 'network')]
    ld = lambda sel: (np.log([p[5] for p in sel]), np.log([p[6] for p in sel]))
    rows = []
    x, y = ld(pts)
    sl, ic = np.polyfit(x, y, 1)
    ratio = np.exp(y - x)
    rows.append(('pooled', 'all', len(pts), sl, float(np.corrcoef(x, y)[0, 1]), float(np.exp(np.mean(y - x))),
                 float(ratio.min()), float(ratio.max()), float(np.exp(x.max() - x.min()))))
    for gi, name in ((0, 'diffusivity'), (1, 'flight_ms'), (2, 'mode'), (3, 'phi_eq'), (4, 'Tg_shift')):
        for lev in dict.fromkeys(p[gi] for p in pts):
            sel = [p for p in pts if p[gi] == lev]
            x, y = ld(sel)
            sl = np.polyfit(x, y, 1)[0]
            ratio = np.exp(y - x)
            rows.append((name, lev, len(sel), sl, float(np.corrcoef(x, y)[0, 1]), float(np.exp(np.mean(y - x))),
                         float(ratio.min()), float(ratio.max()), float(np.exp(x.max() - x.min()))))
    write('a3_flat_dependence.csv', ['grouping', 'level', 'n', 'log_log_slope', 'log_log_correlation',
                                     'ratio_geomean', 'ratio_min', 'ratio_max', 'flat_fold_range'], rows)
    # the scaling estimate sqrt(Pi phi*) of the window width (Section S4.3)
    rows = []
    for m in ('alpha', 'network'):
        for pe in pd.PHI_EQ:
            for s in pd.SHIFTS:
                t = 0.1
                ps = pd.phi_star(pe, s, pd.MODES[m], t)
                h = 1e-5
                Pi = (pd.log10_lambda(ps + h, pe, s, pd.MODES[m]) - pd.log10_lambda(ps - h, pe, s, pd.MODES[m])) \
                    / (2 * h) * math.log(10)
                rows.append((m, pe, s, t * 1000, ps, -Pi, -Pi / math.log(10) / 100, math.sqrt(-Pi * ps)))
    write('a3_window_estimate.csv', ['mode', 'phi_eq', 'Tg_shift', 'flight_ms', 'phi_star', 'Pi_per_unit_fraction',
                                     'decades_per_vol_percent', 'sqrt_Pi_phi_star'], rows)
    # variants relative to the baseline at the same inputs
    rows = []
    for v in dfm.VARIANTS[1:]:
        for m in pd.MAIN_MODES:
            rf = [d_ua(S[v], t, m, pe, s) / d_ua(base, t, m, pe, s) - 1 for t in pd.FLIGHTS for pe in pd.PHI_EQ
                  for s in pd.SHIFTS]
            ra = [d_a3(S[v], t, m, pe, s) / d_a3(base, t, m, pe, s) - 1 for t in pd.FLIGHTS for pe in pd.PHI_EQ
                  for s in pd.SHIFTS]
            rows.append((v, m, min(rf), max(rf), min(ra), max(ra)))
    write('variant_effects.csv', ['variant', 'mode', 'flat_change_min', 'flat_change_max', 'a3_change_min',
                                  'a3_change_max'], rows)
    # case study: minimum flight for a readout floor (network)
    rows = []
    for v in dfm.VARIANTS:
        for pe in pd.PHI_EQ:
            for s in pd.SHIFTS:
                sel = [(t, d_ua(S[v], t, 'network', pe, s)) for t in pd.CASE_FLIGHTS]
                for floor in (140.0, 500.0):
                    tmin = float('nan')
                    if np.isfinite(sel[0][1]) and sel[0][1] >= floor:
                        tmin = sel[0][0]
                    else:
                        for (t0, d0), (t1, d1) in zip(sel[:-1], sel[1:]):
                            if np.isfinite(d0) and np.isfinite(d1) and d0 < floor <= d1:
                                f = (math.log(floor) - math.log(d0)) / (math.log(d1) - math.log(d0))
                                tmin = math.exp(math.log(t0) + f * (math.log(t1) - math.log(t0)))
                                break
                    rows.append((v, pe, s, floor, tmin * 1000))
    write('case_study_consistent.csv', ['diffusivity_variant', 'phi_eq', 'Tg_shift_K_per_vol_percent',
                                        'readout_floor_nm', 'min_flight_ms'], rows)
    # global window: from max(network uniform-arrest diameter, 0.5 um) to min(network A3 limit, 2.5 um)
    rows = []
    for t in pd.FLIGHTS:
        widths = []
        for v in dfm.VARIANTS:
            for pe in pd.PHI_EQ:
                for s in pd.SHIFTS:
                    lo = max(d_ua(S[v], t, 'network', pe, s), 500.0)
                    hi = min(d_a3(S[v], t, 'network', pe, s), 2500.0)
                    if hi > lo:
                        widths.append(hi / lo)
        w = np.array(widths)
        rows.append((t * 1000, 60, len(w), w.min() if len(w) else '', float(np.median(w)) if len(w) else '',
                     w.max() if len(w) else ''))
    write('ps_window_global.csv', ['flight_ms', 'cases', 'cases_window_open', 'width_ratio_min', 'width_ratio_median',
                                   'width_ratio_max'], rows)
    # threshold sensitivity at central inputs (network)
    rows = []
    for thr in (0.05, 0.1, 0.2):
        for t in (0.04, 0.1):
            for pe in pd.PHI_EQ:
                for s in pd.SHIFTS:
                    rows.append((thr, t * 1000, pe, s, d_ua(base, t, 'network', pe, s, thr)))
    write('flat_threshold_sensitivity.csv', ['span_threshold', 'flight_ms', 'phi_eq', 'Tg_shift_K_per_vol_percent',
                                             'flat_diameter_nm'], rows)
    # budget (Table S7), network, central: baseline, 6 K, phi_eq 0.13 and 0.17, 40 and 100 ms
    rows = []
    f = lambda vals: max(vals) / min(vals)
    fl = [f([d_ua(base, t, 'network', pe, CENTRAL_SHIFT) for t in (0.01, 0.1)]) for pe in pd.PHI_EQ]
    rows.append(('flight time 10-100 ms', min(fl), max(fl)))
    dv = [f([d_ua(S[v], t, 'network', pe, CENTRAL_SHIFT) for v in dfm.VARIANTS]) for pe in pd.PHI_EQ for t in (0.04, 0.1)]
    rows.append(('diffusivity form (ten variants)', min(dv), max(dv)))
    pe_ = [f([d_ua(base, t, 'network', pe, CENTRAL_SHIFT) for pe in pd.PHI_EQ]) for t in (0.04, 0.1)]
    rows.append(('equilibrium glass composition 0.13-0.17', min(pe_), max(pe_)))
    sh = [f([d_ua(base, t, 'network', pe, s) for s in pd.SHIFTS]) for pe in pd.PHI_EQ for t in (0.04, 0.1)]
    rows.append(('glass-transition shift 5-7 K per vol%', min(sh), max(sh)))
    allv = [d_ua(S[v], t, 'network', pe, s) for v in dfm.VARIANTS for t in (0.01, 0.04, 0.1) for pe in pd.PHI_EQ
            for s in pd.SHIFTS]
    rows.append(('all inputs jointly, 10-100 ms', f(allv), f(allv)))
    rows.append(('all inputs jointly, 10-100 ms, min nm', min(allv), min(allv)))
    rows.append(('all inputs jointly, 10-100 ms, max nm', max(allv), max(allv)))
    write('flat_diameter_budget.csv', ['input', 'factor_min', 'factor_max'], rows)


# ------------------------------------------------------------------------------------------- checks
CHECK_PE = 0.15


def _check(args):
    label, kw = args
    combos = [(0.1, m, pe, CENTRAL_SHIFT) for m in ('alpha', 'network') for pe in (*pd.PHI_EQ, CHECK_PE)]
    return label, pd.scan('baseline', combos=combos, **kw)


def part_drying_checks():
    cases = [('reference', {}), ('evaporation x0.3', {'evap': 0.3}), ('evaporation x3', {'evap': 3.0}),
             ('evaporation x10', {'evap': 10.0}), ('equilibrium WLF below Tg', {'relax': {'x_glass': None}}),
             ('glass law x = 0.2', {'relax': {'x_glass': 0.2}}), ('glass law x = 0.4', {'relax': {'x_glass': 0.4}}),
             ('PS WLF constants', {'relax': {'wlf': pd.WLF_PS}}),
             ('grid doubled', {'faces': pd.surface_grid(120, 1e-4)}),
             ('uniform grid, same size', {'faces': pd.uniform_grid(60)})]
    with ProcessPoolExecutor(WORKERS) as ex:
        R = dict(ex.map(_check, cases))
    rows = []
    ref = R['reference']
    for label, sc in R.items():
        for m in ('alpha', 'network'):
            for pe in (*pd.PHI_EQ, CHECK_PE):
                k = (0.1, m, pe, CENTRAL_SHIFT)
                du = pd.crossing_diameter_nm(0.1, sc[k][1], pd.THRESHOLD)
                da = pd.crossing_diameter_nm(0.1, sc[k][2], pd.KEEP, increasing_with_d=False)
                du0 = pd.crossing_diameter_nm(0.1, ref[k][1], pd.THRESHOLD)
                da0 = pd.crossing_diameter_nm(0.1, ref[k][2], pd.KEEP, increasing_with_d=False)
                rows.append((label, m, pe, CENTRAL_SHIFT, 100, du, da, du / du0 - 1, da / da0 - 1))
    write('a3_post_arrest_checks.csv', ['check', 'mode', 'phi_eq', 'Tg_shift', 'flight_ms', 'flat_diameter_nm',
                                        'a3_limit_nm', 'flat_change', 'a3_change'], rows)
    # evaporation factor for the budget (network, phi_eq 0.15, 6 K, 100 ms)
    d = [pd.crossing_diameter_nm(0.1, R[c][(0.1, 'network', CHECK_PE, CENTRAL_SHIFT)][1], pd.THRESHOLD)
         for c in ('evaporation x0.3', 'reference', 'evaporation x3', 'evaporation x10')]
    write('flat_diameter_budget_evaporation.csv', ['input', 'factor'], [('evaporation rate 0.3-10x', max(d) / min(d))])


# ------------------------------------------------------------------------------------------ arrest front
def part_skinfront():
    D = dfm.Diffusivity('baseline')
    rows = []
    for d_nm in (500.0, 1000.0, 1700.0):
        for t in (0.04, 0.1):
            R = d_nm * 1e-9 / 2
            sol = pd.Drying(D, t / R ** 2)
            for ps in (0.14, 0.17, 0.20):
                tf = sol.crossing_times(ps)
                x = sol.x
                ts = tf[-1]
                out = []
                for depth in (35.0, 140.0):
                    xd = 1 - depth * 1e-9 / R
                    out.append((float(np.interp(xd, x, tf)) - ts) * t * 1000)
                rows.append((d_nm, t * 1000, ps, ts * t * 1000, *out))
    write('ps_arrest_front.csv', ['diameter_nm', 'flight_ms', 'phi_star', 'surface_arrest_ms',
                                  'front_35nm_after_surface_ms', 'front_140nm_after_surface_ms'], rows)


# ------------------------------------------------------------------------------------- Remark 3: post-arrest loss
def part_remark3():
    """How much differential post-arrest loss of an alpha-timescale channel (central inputs, 100 ms) can steepen the
    published PS trend: decay constant of an exponential fitted (log-linear, bin diameters 0.5-2.3 um) to the
    published fit times the fraction kept, against that of the fit itself."""
    combos = [(0.1, 'alpha', pe, s) for pe in pd.PHI_EQ for s in pd.SHIFTS]
    sc = pd.scan('baseline', combos=combos)
    d = np.arange(500.0, 2301.0, 200.0)
    rows = []
    for k in combos:
        kept = np.array([pd.value_at_diameter(0.1, sc[k][2], x) for x in d])
        base = np.polyfit(d, np.log(0.4282 * np.exp(-1e-3 * d)), 1)[0]
        steep = np.polyfit(d, np.log(0.4282 * np.exp(-1e-3 * d) * kept), 1)[0]
        rows.append((k[2], k[3], kept[0], kept[-1], steep / base - 1))
    write('a3_trend_steepening.csv', ['phi_eq', 'Tg_shift', 'kept_at_0.5um', 'kept_at_2.3um', 'steepening'], rows)


# ------------------------------------------------------------------------- the post-arrest span against total clocks
def part_clockspan():
    """Eq. (S1), the registration time the axis accumulates after the surface arrests, against the difference of the
    total clocks Theta_r(0) - Theta_r(R), which also contains the wet-stage clocks (baseline, central inputs, 100 ms)."""
    D = dfm.Diffusivity('baseline')
    rows = []
    t = 0.1
    for d_nm in (100.0, 200.0, 300.0, 600.0):
        R = d_nm * 1e-9 / 2
        sol = pd.Drying(D, t / R ** 2)
        for m in ('alpha', 'network'):
            for pe in pd.PHI_EQ:
                for s in pd.SHIFTS:
                    f = pd.MODES[m]
                    ps, span, _ = sol.span_and_keep(pe, s, f, t, keep=False)
                    tf = sol.crossing_times(ps)
                    cum = sol._cumulative(sol.clock_rate(pe, s, f, t), sol.tau)
                    th = [float(np.interp(tf[i], sol.tau, cum[i])) for i in (0, -1)]
                    rows.append((d_nm, m, pe, s, span, th[0] - th[1], th[1]))
    write('clock_span_comparison.csv', ['diameter_nm', 'mode', 'phi_eq', 'Tg_shift', 'span_after_surface_arrest_S1',
                                        'total_clock_difference', 'surface_clock_at_arrest'], rows)

# ---------------------------------------------------------------------------------- fast-evaporation limit
FAST_EVAP = (1.0, 10.0, 100.0, 1000.0, 10000.0)


def _fast(args):
    label, kw = args
    combos = [(t, m, pe, CENTRAL_SHIFT) for t in (0.01, 0.1) for m in ('alpha', 'network') for pe in (*pd.PHI_EQ, CHECK_PE)]
    return label, combos, pd.scan('baseline', combos=combos, **kw)


def part_fastevap():
    """Uniform-arrest diameter and A3 limit as the evaporation rate rises toward the surface-equilibrium limit.
    The reference law k = R ln(c0/0.2)/(2 t_flight) is a convention; gas-side mass transfer for a sub-micrometre jet is
    about 10^3-10^4 times faster, which makes the surface condition nearly Dirichlet (surface at c = 0). Factors up to
    10^4 approach that limit; the last two cases refine the grid at the largest factor."""
    cases = [('evaporation x%g' % e, {'evap': e}) for e in FAST_EVAP]
    cases += [('evaporation x10000, grid doubled', {'evap': 1e4, 'faces': pd.surface_grid(120, 1e-4)}),
              ('evaporation x10000, grid x4, h/10', {'evap': 1e4, 'faces': pd.surface_grid(240, 2e-5)})]
    with ProcessPoolExecutor(WORKERS) as ex:
        R = {lab: (combos, sc) for lab, combos, sc in ex.map(_fast, cases)}
    rows = []
    ref_combos, ref = R['evaporation x1']
    for label, (combos, sc) in R.items():
        for k in combos:
            t, m, pe, sh = k
            du = pd.crossing_diameter_nm(t, sc[k][1], pd.THRESHOLD)
            da = pd.crossing_diameter_nm(t, sc[k][2], pd.KEEP, increasing_with_d=False)
            du0 = pd.crossing_diameter_nm(t, ref[k][1], pd.THRESHOLD)
            da0 = pd.crossing_diameter_nm(t, ref[k][2], pd.KEEP, increasing_with_d=False)
            rows.append((label, m, pe, sh, t * 1000, du, da, da / du, du / du0 - 1, da / da0 - 1))
    write('fast_evaporation.csv', ['check', 'mode', 'phi_eq', 'Tg_shift', 'flight_ms', 'flat_diameter_nm',
                                   'a3_limit_nm', 'a3_over_flat', 'flat_change', 'a3_change'], rows)
