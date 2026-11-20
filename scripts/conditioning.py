"""Conditioning of the accumulation map behind A5 (Section S3.3) and the sampling design of the functional-form
comparison (Section S6.2). Parts (called by scripts/run_results.py): a5cond, forms_sampling."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

import separability as sep
from common import write


def common_clock_grid(lam0=0.5, tau=0.3, nx=40, ne=40, D=0.01):
    """Labels (stretch-rate scale, initial depth) and accumulations (log stretch, log clock) under a common clock;
    the same elements as separability.a5_element_injectivity."""
    def element(x, eps, n=4000):
        tt = np.linspace(0.0, 60.0, n)
        depth = (1 - x) * np.exp(-eps * tt / 2)
        front = np.sqrt(D * tt)
        k = int(np.argmax(front >= depth)) if np.any(front >= depth) else n - 1
        k = max(k, 1)
        seg = tt[:k + 1]
        return eps * tt[k], np.trapezoid(1.0 / (lam0 * np.exp(-seg / tau)), seg)
    xs = np.linspace(0.05, 0.95, nx)
    es = np.linspace(0.3, 3.0, ne)
    G = np.array([[element(x, e) for x in xs] for e in es])
    return es, xs, np.log(G[..., 0]), np.log(G[..., 1])


def depth_clock_grid(n=100, phi_star=0.2, k=5.0):
    """The same elements as separability.a5_depth_dependent_clock (clock from the drying solution of Lemma 1)."""
    x = np.linspace(0, 1, n)
    dx = x[1] - x[0]
    xf = 0.5 * (x[1:] + x[:-1])
    D = lambda c: np.exp(9.2 * (np.clip(c, 0, 1) - 0.9))

    def rhs(t, c):
        cf = 0.5 * (c[1:] + c[:-1])
        fl = -D(cf) * (c[1:] - c[:-1]) / dx
        dc = np.zeros_like(c)
        i = np.arange(1, n - 1)
        dc[i] = -(xf[i] * fl[i] - xf[i - 1] * fl[i - 1]) / (x[i] * dx)
        dc[0] = -4 * fl[0] / dx
        dc[-1] = -(k * c[-1] - xf[-1] * fl[-1]) / (dx / 2)
        return dc
    TAU = np.geomspace(1e-05, 400, 6000)
    C = solve_ivp(rhs, (0, 400), np.full(n, 0.9), method='BDF', t_eval=TAU, rtol=1e-08, atol=1e-11).y
    lam = lambda c: np.exp(8 * (0.9 - c))

    def element(ix, s):
        c = C[ix]
        j = int(np.argmax(c <= phi_star))
        tau_f = TAU[j - 1] + (c[j - 1] - phi_star) / (c[j - 1] - c[j]) * (TAU[j] - TAU[j - 1])
        t_f = np.log1p(s * tau_f) / s
        tt = np.linspace(0, t_f, 3000)
        cc = np.interp(np.expm1(s * tt) / s, TAU, c)
        return s * t_f, np.trapezoid(1 / lam(cc), tt)
    xs = np.arange(3, n - 3, 3)
    ss = np.linspace(0.3, 3.0, 30)
    G = np.array([[element(i, s) for i in xs] for s in ss])
    return ss, x[xs], np.log(G[..., 0]), np.log(G[..., 1])


def conditioning(a, b, Tp, Tr, label_gap=0.15):
    """Labels and accumulations rescaled to their sample ranges; Jacobian condition numbers over the interior grid,
    closest approach in accumulation space of elements whose labels differ by more than label_gap, and the largest
    ratio of label distance to accumulation distance over such pairs (a discrete inverse-Lipschitz bound)."""
    ua = (a - a.min()) / np.ptp(a)
    ub = (b - b.min()) / np.ptp(b)
    P1 = (Tp - Tp.min()) / np.ptp(Tp)
    P2 = (Tr - Tr.min()) / np.ptp(Tr)
    J = np.empty(Tp.shape + (2, 2))
    J[..., 0, 0] = np.gradient(P1, ua, axis=0)
    J[..., 0, 1] = np.gradient(P1, ub, axis=1)
    J[..., 1, 0] = np.gradient(P2, ua, axis=0)
    J[..., 1, 1] = np.gradient(P2, ub, axis=1)
    sv = np.linalg.svd(J[1:-1, 1:-1].reshape(-1, 2, 2), compute_uv=False)
    cond = sv[:, 0] / sv[:, 1]
    L = np.stack(np.meshgrid(ua, ub, indexing='ij'), -1).reshape(-1, 2)
    P = np.stack([P1.ravel(), P2.ravel()], 1)
    dl = np.linalg.norm(L[:, None] - L[None], axis=-1)
    dp = np.linalg.norm(P[:, None] - P[None], axis=-1)
    far = dl > label_gap
    return float(np.median(cond)), float(cond.max()), float(dp[far].min()), float((dl[far] / dp[far]).max())


def part_a5cond():
    rows = [('common clock', *conditioning(*common_clock_grid())),
            ('depth-dependent clock', *conditioning(*depth_clock_grid()))]
    write('a5_conditioning.csv', ['clock', 'median_jacobian_condition', 'max_jacobian_condition',
                                  'min_accumulation_distance_far_labels', 'max_inverse_lipschitz'], rows)


def part_forms_sampling():
    """Total fibers to reject one form for the other (two-sided 5%, 80% power) when diameters are drawn uniformly
    in diameter or uniformly in log diameter over 100-2000 nm: the 30 nm shell fraction against the exponential
    fitted under the same sampling (400 diameters, as in functional_forms.csv)."""
    rows = []
    for label, d in (('uniform in diameter', np.linspace(*sep.WINDOW, 400)),
                     ('uniform in log diameter', np.geomspace(*sep.WINDOW, 400))):
        f = sep.shell_fraction(d, 30.0)
        p, _ = curve_fit(sep.exponential, d, f, p0=[f[0], 500.0, 0.0], maxfev=100000)
        ms = float(np.mean((f - sep.exponential(d, *p)) ** 2))
        rows.append((label, *p, ms ** 0.5, *[sep.Z ** 2 * s ** 2 / ms for s in (0.05, 0.10, 0.15)]))
    write('functional_forms_sampling.csv', ['diameter_sampling', 'A', 'd_c_nm', 'b', 'rms_residual',
                                            'n_total_dispersion_0.05', 'n_total_dispersion_0.10',
                                            'n_total_dispersion_0.15'], rows)


def part_route4_excess():
    """Size of the interior maximum of Proposition 3(b): constant stretching rate from t = 0, single exponential kernel
    of relaxation time lam, decoupled skin. An element arresting at t_f retains e lam (1 - exp(-t_f/lam)); the surface
    arrests at t_f(R), the element arresting when stretching ends at t_dag retains the most. Relative excess
    (interior maximum - surface) / surface = [exp(-a) - exp(-a b)] / (1 - exp(-a)), a = t_f(R)/lam, b = t_dag/t_f(R).
    The partial-coupling threshold of Section S11.10 is exp(-a)."""
    rows = []
    for r in (0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0):
        a = 1.0 / r
        for b in (1.5, 2.0, 5.0, np.inf):
            ex = (np.exp(-a) - (0.0 if np.isinf(b) else np.exp(-a * b))) / (1.0 - np.exp(-a))
            rows.append((r, b, ex, np.exp(-a)))
    write('route4_interior_excess.csv', ['lambda_over_tfR', 'tdag_over_tfR', 'relative_interior_excess',
                                         'coupling_threshold_alpha'], rows)


def part_p4_bias():
    """Bias of an orientation estimator that assumes the most probable distribution (MPD) for <P4> when the fiber is a
    perfectly aligned skin over an isotropic core. The mixture has <P4> = <P2> = m; the MPD at <P2> = m has a much smaller
    <P4>. For a readout linear in <P2> and <P4> with relative <P4> sensitivity r, the estimated mean is off by about
    r (m - P4_MPD(m)) / (1 + r dP4_MPD/dP2)."""
    from scipy.optimize import brentq
    x = np.linspace(-1.0, 1.0, 20001)
    p2 = (3 * x ** 2 - 1) / 2
    p4 = (35 * x ** 4 - 30 * x ** 2 + 3) / 8

    def mpd_p4(m):
        lam = brentq(lambda L: np.trapezoid(np.exp(L * p2) * p2, x) / np.trapezoid(np.exp(L * p2), x) - m, -20, 50)
        w = np.exp(lam * p2)
        return np.trapezoid(w * p4, x) / np.trapezoid(w, x)

    rows = []
    for m in (0.09, 0.10, 0.15, 0.20, 0.24):
        g = mpd_p4(m)
        for r in (0.25, 0.5):
            # exact: the estimator returns e with e + r P4_MPD(e) = m + r m (readout linear in <P2> and <P4>)
            e = brentq(lambda z: z + r * mpd_p4(z) - (m + r * m), 1e-4, 0.9)
            rows.append((m, g, r, (e - m) / m))
    write('ps_p4_bias.csv', ['mean_P2', 'P4_most_probable', 'relative_P4_sensitivity', 'relative_bias_of_mean'], rows)


def part_gas_side():
    """Ratio of an ideal gas-side evaporation rate to the reference rate of the drying model (Section S5.3), and the
    nonparametric tolerance level of a sample maximum used for the PET plate ceiling (Section 5.2.1)."""
    R, t = 0.25e-6, 0.1                                  # 0.5 um fiber after a 100 ms flight
    k_ref = R * np.log(0.9 / 0.2) / (2 * t)              # m/s, reference law of ps_drying.py
    Sh, Dg = 0.5, 1.0e-5                                 # Sherwood number on the diameter, vapor diffusivity in air (m2/s)
    p_sat, M, T, Rg, rho_l = 21.2e3, 0.11938, 293.15, 8.314, 1480.0
    rho_v = p_sat * M / (Rg * T)                         # saturated chloroform vapor density (kg/m3)
    k_gas = Sh * Dg / (2 * R) * rho_v / rho_l            # liquid-volume velocity (m/s)
    n = 16
    write('gas_side_and_tolerance.csv', ['quantity', 'value'],
          [('reference evaporation velocity (m/s)', k_ref), ('saturated vapor density (kg/m3)', rho_v),
           ('ideal gas-side velocity (m/s)', k_gas), ('gas-side over reference', k_gas / k_ref),
           ('population quantile exceeded by the largest of 16 with 95% confidence', 0.05 ** (1 / n))])


def part_biot():
    """Biot-number scope of Eq. (9) (Section 4.1, S5.1): the premise Bi = kR/Ds(phi*) < 2/(Pi phi*), against the Biot
    number of the reference evaporation law, Bi = (ln 4.5 / 2) R^2/(Ds t) = 0.752 x Eq. (9), at the uniform-arrest
    threshold (Eq. (9) = 0.1)."""
    import csv
    import os
    rows = []
    with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'a3_window_estimate.csv')) as f:
        for r in csv.DictReader(f):
            rows.append((r['mode'], float(r['phi_eq']), float(r['Tg_shift']),
                         2.0 / (float(r['Pi_per_unit_fraction']) * float(r['phi_star']))))
    coef = np.log(0.9 / 0.2) / 2
    write('biot_scope.csv', ['mode', 'phi_eq', 'Tg_shift', 'biot_limit_2_over_Pi_phi_star', 'reference_biot_per_dTheta',
                             'reference_biot_at_threshold'], [r + (coef, 0.1 * coef) for r in rows])


def part_prop4_early_end():
    """Proposition 4 when stretching ends before the axis arrests (t-dagger = end of stretching < tf(0)), with long
    relaxation afterwards: stretching ends when the arrest front reaches 0.95 or 0.9 of the radius, i.e. just after the
    surface arrests, so inner elements relax for most of the drying time before they arrest. Reports where the maximum
    lies, its excess over the surface value, and whether the sufficient condition of Proposition 4 holds
    (constant stretching, single exponential: lam_wet (1 - exp(-t_dag/lam_wet)) > tf(R))."""
    import separability as sep
    rows = []
    for rs, lf, a, pos, excess, tR, t_end, tf0 in sep.proposition4_early_end_check():
        lam_wet = lf * t_end
        cond = lam_wet * (1 - np.exp(-t_end / lam_wet)) > tR
        rows.append((rs, lf, a, pos, excess, int(cond), t_end / tf0))
    write('proposition4_early_end.csv', ['r_stop', 'lam0_over_t_end', 'clock_strength_a', 'max_position_r_over_R',
                                         'max_excess_over_surface', 'prop4_condition_holds', 't_end_over_tf_axis'], rows)


def part_ps_joint():
    """Joint scenarios for the polystyrene skin bound (Section 5.1.2, S8.3): the disk adds a share c of each bin mean and
    the readout misreports the mean by a factor (1 + b) (b > 0: over-reports, as for an estimator that assumes the most
    probable distribution, Section S8.3; b < 0: under-reports). True skin-induced mean = P2 (1 - c) / (1 + b), core 0,
    skin perfectly aligned; minimum thickness per bin (d/2)[1 - sqrt(1 - f)], and the single thickness that every bin
    needs (largest per-bin minimum), over the bins 0.5-1.7 um. Volume-average weighting."""
    import csv
    import os
    bins = []
    with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'ps_skin_bound.csv')) as f:
        for r in csv.DictReader(f):
            d = float(r['diameter_nm'])
            if 500 <= d <= 1700:
                bins.append((d, float(r['P2'])))
    rows = []
    for c in (0.0, 0.5, 0.7):
        for b in (-0.2, 0.0, 0.2, 0.4):
            mins = [(d / 2) * (1 - np.sqrt(1 - p * (1 - c) / (1 + b))) for d, p in bins]
            rows.append((c, b, min(mins), max(mins)))
    write('ps_joint_scenarios.csv', ['collector_share', 'readout_bias', 'per_bin_min_nm_low', 'single_thickness_min_nm'], rows)
