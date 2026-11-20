"""Synthetic-data validation of the measurement protocol (Section 5.4 and Fig. 7 of the article; Section S10).

Three protocol steps are tested on fibers simulated under a known mechanism, with Raman-like errors:

V1  Static-plate control (protocol step 2): can a plate-versus-collector comparison tell whether a collector
    supplied most of the orientation, and are fiber-level confidence intervals honest when fibers share
    spinning sessions?
V2  Route 3 (relaxation perturbation): how many fibers detect the diameter shift that flow history predicts
    for a twofold change in relaxation time, and how often does a skin (no shift) or a small readout offset
    between solutions produce a false detection?
V3  Route 4 (radial maximum): how often is an interior maximum detected in a cross-section line profile,
    with a point-spread function, pixel noise and a saturating map from retained deformation to orientation,
    when the truth is fading memory (interior maximum) or a smooth skin (surface maximum)?

Every random draw uses a fixed seed. Run:  python scripts/synthetic_validation.py  (about 2-4 min).
Outputs: results/synth_plate_control.csv, results/synth_route3.csv, results/synth_route4.csv,
figures/fig7_synthetic_validation.(pdf|png).
"""
import os
import sys
import numpy as np

RES = 'results'
FIG = 'figures'

# ---------------------------------------------------------------- error model shared by V1 and V2
# Fiber-to-fiber spread of true orientation at fixed diameter and per-fiber measurement error of polarized
# Raman. The PS bin spreads reach 0.076 (Table S10) and include measurement error, so the total used here is conservative; oil-immersed and calibrated PAN readouts
# differ by 0.09 rms per fiber (two estimators), i.e. about 0.065 for one.
SD_FIBER = 0.06
SD_MEAS = 0.065


def session_sd(icc, sd_within):
    """Between-session standard deviation giving intraclass correlation icc."""
    return np.sqrt(icc / (1.0 - icc)) * sd_within


# ================================================================ V1: static-plate control
def _ratio_interval(plate, coll, sess_p, sess_c, cluster):
    """95% interval of mean(plate)/mean(collector) by the delta method with Student t quantiles, using fibers as
    the units (fiber-level analysis) or session means as the units (session-level analysis, 2G - 2 degrees of
    freedom). With few sessions the session-level analysis is the honest one when fibers share sessions."""
    from scipy.stats import t as student
    if cluster:
        a = np.array([plate[sess_p == s].mean() for s in np.unique(sess_p)])
        b = np.array([coll[sess_c == s].mean() for s in np.unique(sess_c)])
    else:
        a, b = plate, coll
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    R = ma / mb
    se = abs(R) * np.sqrt(va / ma ** 2 + vb / mb ** 2) if ma != 0 else np.sqrt(va) / mb
    q = student.ppf(0.975, len(a) + len(b) - 2)
    return np.array([R - q * se, R + q * se])


def v1_plate_control(n_sim=2000, seed=101):
    """Share of the collector-fiber mean that a flight-set state could supply, estimated as
    mean(plate)/mean(collector) for fibers below the readout limit, PET-like levels: collector mean 0.24, plate
    mean set by the true flight share f (0.1, 0.3 or 0.5 of the collector mean)."""
    rng = np.random.default_rng(seed)
    rows = []
    coll_mean = 0.24
    for f in (0.1, 0.3, 0.5):
        for n in (16, 32, 64):
            for n_sess in (2, 4, 8):
                for icc in (0.0, 0.1, 0.2):
                    sd_w = np.hypot(SD_FIBER, SD_MEAS)
                    tau = session_sd(icc, sd_w)
                    cover = {False: 0, True: 0}; width = {False: [], True: []}; majority = {False: 0, True: 0}
                    for _ in range(n_sim):
                        sp = np.repeat(np.arange(n_sess), n // n_sess)
                        sc = sp
                        plate = f * coll_mean + rng.normal(0, tau, n_sess)[sp] + rng.normal(0, sd_w, len(sp))
                        coll = coll_mean + rng.normal(0, tau, n_sess)[sc] + rng.normal(0, sd_w, len(sc))
                        for cl in (False, True):
                            lo, hi = _ratio_interval(plate, coll, sp, sc, cl)
                            cover[cl] += (lo <= f <= hi)
                            width[cl].append(hi - lo)
                            majority[cl] += (hi < 0.5)   # collector supplies most of the orientation
                    for cl in (False, True):
                        rows.append(dict(flight_share=f, fibers_per_collector=n, sessions=n_sess, icc=icc,
                                         resampling='session' if cl else 'fiber',
                                         coverage_95=cover[cl] / n_sim, median_ci_width=float(np.median(width[cl])),
                                         p_conclude_collector_majority=majority[cl] / n_sim))
    return rows


# ================================================================ V2: Route 3
def _trend(d_nm, L):
    """PS-like diameter-orientation trend: the published fit 0.428 exp(-d/L) with L = 1000 nm (Section 5.1.2)."""
    return 0.428 * np.exp(-d_nm / L)


def _fit_shift(d, y, g, sess):
    """Fit y = a exp(-d/(L(1+s g))) + b by bounded least squares; return s and its session-cluster-robust
    (sandwich) standard error."""
    from scipy.optimize import least_squares
    def model(th):
        a, lnL, s, b = th
        return a * np.exp(-d / (np.exp(lnL) * (1 + s * g))) + b
    res = least_squares(lambda th: model(th) - y, x0=[0.43, np.log(1000.0), 0.0, 0.0],
                        bounds=([0.0, np.log(100.0), -0.5, -0.2], [2.0, np.log(20000.0), 1.0, 0.5]))
    J = res.jac; r = -res.fun
    bread = np.linalg.pinv(J.T @ J)
    meat = np.zeros((4, 4))
    for c in np.unique(sess):
        m = sess == c
        u = J[m].T @ r[m]
        meat += np.outer(u, u)
    G = len(np.unique(sess)); n = len(y)
    meat *= G / max(G - 1, 1) * (n - 1) / (n - 4)
    V = bread @ meat @ bread
    return res.x[2], np.sqrt(max(V[2, 2], 0))


def v2_route3(n_sim=400, seed=202):
    """Two solutions differing twofold in relaxation time. Flow history shifts the trend in diameter by a
    factor 1+s (s = 0.05-0.14 of the 3-14% of Section 4.3); a committed skin predicts s = 0. Diameters log-uniform over
    0.4-2.5 um. Fibers come from 4 or 16 spinning sessions per solution with intraclass correlation 0 or 0.1.
    In the separate design each session spins one solution; in the blocked design both solutions are spun in
    each of 8 sessions, so a session effect is common to both. A readout offset of 0.01 between the solutions
    tests A2 for the comparison. Detection: two-sided test of
    s = 0 at 5% with session-cluster-robust errors and Student t quantiles on G - 1 degrees of freedom."""
    from scipy.stats import t as student
    rng = np.random.default_rng(seed)
    rows = []
    sd_w = np.hypot(SD_FIBER, SD_MEAS)
    for truth, s_true, offset in (('flow history', 0.05, 0.0), ('flow history', 0.10, 0.0), ('flow history', 0.14, 0.0),
                                  ('skin', 0.0, 0.0), ('skin', 0.0, 0.01)):
        for icc, n_sess, design in ((0.0, 4, 'separate'), (0.1, 4, 'separate'), (0.1, 16, 'separate'), (0.1, 8, 'blocked')):
            tau = session_sd(icc, sd_w)
            for n in (64, 128, 256, 512, 1024):
                det = 0
                G = 2 * n_sess if design == 'separate' else n_sess
                q = student.ppf(0.975, G - 1)
                for _ in range(n_sim):
                    d = np.exp(rng.uniform(np.log(400), np.log(2500), 2 * n))
                    g = np.repeat([0.0, 1.0], n)
                    if design == 'separate':      # each session spins one solution
                        sess = np.repeat(np.arange(G), n // n_sess)
                    else:                         # both solutions spun in every session, alternating
                        sess = np.tile(np.repeat(np.arange(G), n // G), 2)
                    y = _trend(d, 1000.0 * (1 + s_true * g)) + offset * g
                    y = y + rng.normal(0, tau, G)[sess] + rng.normal(0, sd_w, 2 * n)
                    s_hat, se = _fit_shift(d, y, g, sess)
                    det += abs(s_hat) > q * se
                rows.append(dict(truth=truth, true_shift=s_true, readout_offset=offset, design=design, sessions=n_sess,
                                 icc=icc, fibers_per_solution=n, p_detect_shift=det / n_sim))
    return rows


# ================================================================ V3: Route 4
def _arrest_time(r, R, ts, tc):
    """Diffusive arrest front: the arrested depth grows as the square root of time after surface arrest."""
    return ts + (tc - ts) * ((R - r) / R) ** 2


def retained(r, R, ts, tc, t_end, lam):
    """Retained deformation under constant stretching up to t_end and single-mode fading memory
    (Proposition 3; unit stretching rate); the skin is decoupled, so arrested material is not drawn further."""
    tf = _arrest_time(r, R, ts, tc)
    during = tf <= t_end
    M = np.where(during, lam * (1 - np.exp(-tf / lam)),
                 lam * (np.exp(-(tf - t_end) / lam) - np.exp(-tf / lam)))
    return M


def q_saturating(M, Mref, P_mean_target, r, P_max=1.0):
    """Saturating map S = P_max[1 - exp(-M/M0)] from retained deformation to <P2>, with M0 chosen so the
    cross-section mean of S equals P_mean_target (0.25 PS-like, 0.75 PET-like)."""
    w = r / r.sum()
    lo, hi = 1e-6, 1e6
    for _ in range(200):
        M0 = np.sqrt(lo * hi)
        mean = np.sum(w * P_max * (1 - np.exp(-M / M0)))
        if mean > P_mean_target:
            lo = M0
        else:
            hi = M0
    return P_max * (1 - np.exp(-M / M0))


def _measure(x, prof_abs, R, psf, noise, rng):
    """Line profile across a cross-section: normalized Gaussian blur restricted to the fiber, pixel noise."""
    P = prof_abs(np.abs(x))
    K = np.exp(-0.5 * ((x[:, None] - x[None, :]) / psf) ** 2)
    meas = (K @ P) / K.sum(axis=1)
    return meas + rng.normal(0, noise, len(x))


def _interior_max(x, y, R, psf, noise, npix_bin):
    """Fold the line profile about the axis, bin, and call an interior maximum when the largest bin lies deeper
    than one PSF width and exceeds the outermost bin by more than 2.5 standard errors of a difference."""
    depth = R - np.abs(x)
    edges = np.arange(0, R + 1e-9, npix_bin)
    idx = np.digitize(depth, edges) - 1
    nb = len(edges) - 1
    means = np.array([y[idx == k].mean() if np.any(idx == k) else np.nan for k in range(nb)])
    counts = np.array([np.sum(idx == k) for k in range(nb)])
    k = int(np.nanargmax(means))
    centre = edges[k] + npix_bin / 2
    se = noise * np.sqrt(1 / counts[k] + 1 / counts[0])
    return centre > psf and means[k] - means[0] > 2.5 * se


def v3_route4(n_sim=300, seed=303):
    """Fiber 1 um across; surface arrest at ts = 1, axis at tc = 20 (core arresting 20 times later); relaxation
    time lambda = 1 or 3 surface-arrest times (the slowly relaxing wet material that Proposition 4 requires);
    stretching ends at t_end = ts + phi (tc - ts) (phi = 0.05, 0.15, 0.3: interior maximum at depth R sqrt(phi)).
    The skin comparator has the same cross-section mean and surface value as the flow profile's maximum but
    decreases monotonically inward with a diffusive width. Pixel 10 nm; PSF sigma 20-150 nm; pixel noise 0.03 or
    0.06; mean orientation 0.25 or 0.75 after the saturating map."""
    rng = np.random.default_rng(seed)
    R = 500.0; ts, tc = 1.0, 20.0
    x = np.arange(-R + 5, R, 10.0)
    rr = np.linspace(0, R, 2001)
    rows = []
    for lam, phi in [(l, f) for l in (1.0, 3.0) for f in (0.05, 0.15, 0.3)]:
        t_end = ts + phi * (tc - ts)
        M = retained(rr, R, ts, tc, t_end, lam)
        for P_mean in (0.25, 0.75):
            S_flow = q_saturating(M, None, P_mean, rr)
            peak = S_flow.max(); surf = S_flow[-1]
            depth_peak = R - rr[np.argmax(S_flow)]
            # smooth skin with the flow profile's peak value at the surface and the same area-weighted mean
            w = rr / rr.sum()
            core = None
            lo, hi = 0.0, peak
            for _ in range(100):
                core = 0.5 * (lo + hi)
                width = 60.0
                S_skin = core + (peak - core) * np.exp(-(R - rr) / width)
                if np.sum(w * S_skin) > np.sum(w * S_flow):
                    hi = core
                else:
                    lo = core
            f_flow = lambda a, S=S_flow: np.interp(a, rr, S)
            f_skin = lambda a, S=S_skin: np.interp(a, rr, S)
            for psf in (20.0, 50.0, 100.0, 150.0):
                for noise in (0.03, 0.06):
                    hit = {'fading memory': 0, 'skin': 0}
                    for _ in range(n_sim):
                        for name, f in (('fading memory', f_flow), ('skin', f_skin)):
                            y = _measure(x, f, R, psf, noise, rng)
                            hit[name] += _interior_max(x, y, R, psf, noise, 20.0)
                    rows.append(dict(lambda_over_ts=lam, phi=phi, peak_depth_nm=round(depth_peak, 1), mean_P2=P_mean,
                                     peak_minus_surface=round(peak - surf, 4), psf_sigma_nm=psf, pixel_noise=noise,
                                     p_detect_interior_given_fading_memory=hit['fading memory'] / n_sim,
                                     p_false_interior_given_skin=hit['skin'] / n_sim))
    return rows


# ================================================================ V4: robustness to misspecification
def _wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def v4_robustness(n_sim=2000, n_sim_r4=600, seed=404):
    """Departures from the error models of V1-V3. Plate control (true share 30%, 64 fibers, 4 or 8 sessions per
    collector, intraclass correlation 0.1): Gaussian baseline; heavy-tailed session effects (Student t, 3 degrees of
    freedom, scaled to the same standard deviation); measurement error proportional to the true value (equal mean
    variance); measured values truncated to the physical range [-0.5, 1]; per-readout error 0.04 or 0.09 (the range
    allowed by the PAN discrepancy of 0.092 rms for correlated or unequal errors). Route 4 (maximum 190 nm deep,
    relaxation time 3 surface-arrest times, mean 0.25, PSF sigma 20 or 50 nm, noise 0.03): baseline; pixel noise
    correlated along the profile (AR(1), correlation length 30 nm); true blur 1.5 times the nominal width used in the
    decision rule; a sectioning-damaged outer layer 30 nm thick whose orientation is lowered by 0.05, 0.10 or 0.20."""
    rng = np.random.default_rng(seed)
    rows = []
    f, coll_mean, n = 0.3, 0.24, 64
    for n_sess in (4, 8):
        for case in ('baseline', 'heavy-tailed sessions', 'proportional error', 'truncated to physical range',
                     'readout error 0.04', 'readout error 0.09'):
            sd_meas = {'readout error 0.04': 0.04, 'readout error 0.09': 0.09}.get(case, SD_MEAS)
            sd_w = np.hypot(SD_FIBER, sd_meas)
            tau = session_sd(0.1, sd_w)
            cov = maj = 0
            for _ in range(n_sim):
                sp = np.repeat(np.arange(n_sess), n // n_sess)
                def draw(mu):
                    if case == 'heavy-tailed sessions':
                        se = rng.standard_t(3, n_sess) * tau / np.sqrt(3.0)
                    else:
                        se = rng.normal(0, tau, n_sess)
                    true = mu + se[sp] + rng.normal(0, SD_FIBER, len(sp))
                    if case == 'proportional error':
                        sd = sd_meas * np.clip(true, 0.0, None) / mu * 1.0
                        sd = sd * np.sqrt(sd_meas ** 2 / max(np.mean(sd ** 2), 1e-12))
                        y = true + rng.normal(0, 1, len(sp)) * sd
                    else:
                        y = true + rng.normal(0, sd_meas, len(sp))
                    if case == 'truncated to physical range':
                        y = np.clip(y, -0.5, 1.0)
                    return y
                plate, col = draw(f * coll_mean), draw(coll_mean)
                lo, hi = _ratio_interval(plate, col, sp, sp, True)
                cov += lo <= f <= hi; maj += hi < 0.5
            rows.append(dict(test='plate control', setting=f'{n_sess} sessions', case=case, runs=n_sim,
                             coverage_95=cov / n_sim, p_collector_majority=maj / n_sim,
                             p_false_interior_given_skin='', p_detect_interior_given_fading_memory=''))
    # Route 4
    R = 500.0; ts, tc = 1.0, 20.0
    x = np.arange(-R + 5, R, 10.0); rr = np.linspace(0, R, 2001)
    t_end = ts + 0.15 * (tc - ts)
    M = retained(rr, R, ts, tc, t_end, 3.0)
    S_flow = q_saturating(M, None, 0.25, rr)
    peak = S_flow.max(); w = rr / rr.sum(); lo_, hi_ = 0.0, peak
    for _ in range(100):
        core = 0.5 * (lo_ + hi_); S_skin = core + (peak - core) * np.exp(-(R - rr) / 60.0)
        if np.sum(w * S_skin) > np.sum(w * S_flow): hi_ = core
        else: lo_ = core
    noise = 0.03
    def measure(S, case, psf):
        prof = np.interp(np.abs(x), rr, S)
        if case.startswith('sectioning damage'):
            drop = float(case.split()[-1])
            prof = prof - drop * (np.abs(x) > R - 30.0)
        sig = psf * (1.5 if case == 'blur 1.5x nominal' else 1.0)
        K = np.exp(-0.5 * ((x[:, None] - x[None, :]) / sig) ** 2)
        m = (K @ prof) / K.sum(axis=1)
        if case == 'correlated noise':
            phi = np.exp(-10.0 / 30.0); e = np.empty(len(x)); e[0] = rng.normal(0, noise)
            for i in range(1, len(x)):
                e[i] = phi * e[i - 1] + rng.normal(0, noise * np.sqrt(1 - phi ** 2))
            return m + e
        return m + rng.normal(0, noise, len(x))
    for psf in (20.0, 50.0):
      for case in ('baseline', 'correlated noise', 'blur 1.5x nominal', 'sectioning damage 0.05',
                   'sectioning damage 0.10', 'sectioning damage 0.20'):
        hit_f = hit_s = 0
        for _ in range(n_sim_r4):
            hit_f += _interior_max(x, measure(S_flow, case, psf), R, psf, noise, 20.0)
            hit_s += _interior_max(x, measure(S_skin, case, psf), R, psf, noise, 20.0)
        rows.append(dict(test='Route 4', setting=f'depth 190 nm; PSF {psf:g} nm; noise 0.03', case=case, runs=n_sim_r4,
                           coverage_95='', p_collector_majority='',
                           p_false_interior_given_skin=hit_s / n_sim_r4,
                           p_detect_interior_given_fading_memory=hit_f / n_sim_r4))
    return rows


# ================================================================ output
def write(name, rows):
    os.makedirs(RES, exist_ok=True)
    keys = list(rows[0].keys())
    with open(os.path.join(RES, name), 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join(f'{r[k]:.6g}' if isinstance(r[k], float) else str(r[k]) for k in keys) + '\n')
    print('wrote', os.path.join(RES, name))


def figure(r1, r2, r3):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 3.3))
    # (a) coverage of nominal 95% intervals, flight share 0.3, 4 sessions
    for cl, ls in (('fiber', '--'), ('session', '-')):
        for icc, c in ((0.0, '#7f7f7f'), (0.1, '#2b6cb0'), (0.2, '#b03a2e')):
            pts = [(r['fibers_per_collector'], r['coverage_95']) for r in r1 if r['resampling'] == cl and r['icc'] == icc
                   and r['flight_share'] == 0.3 and r['sessions'] == 4]
            pts.sort()
            ax[0].plot(*zip(*pts), ls, color=c, marker='o', ms=3, label=f'ICC {icc:g}' if cl == 'session' else None)
    ax[0].plot([], [], '-', color='k', label='over session means')
    ax[0].plot([], [], '--', color='k', label='over fibers')
    ax[0].axhline(0.95, color='k', lw=0.6, ls=':')
    ax[0].set_xlabel('fibers per collector'); ax[0].set_ylabel('coverage of nominal 95% interval')
    ax[0].set_ylim(0.4, 1.0); ax[0].legend(fontsize=7, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, columnspacing=0.8)
    # (b) Route 3 power
    for (truth, s, off), c in ((('flow history', 0.05, 0.0), '#9ecae1'), (('flow history', 0.10, 0.0), '#4292c6'),
                               (('flow history', 0.14, 0.0), '#08519c'), (('skin', 0.0, 0.0), '#7f7f7f'),
                               (('skin', 0.0, 0.01), '#b03a2e')):
        pts = sorted((r['fibers_per_solution'], r['p_detect_shift']) for r in r2
                     if r['truth'] == truth and r['true_shift'] == s and r['readout_offset'] == off and r['icc'] == 0.1
                     and r['design'] == 'blocked')
        lab = f'shift {s:.0%}' if truth == 'flow history' else ('skin' if off == 0 else 'skin, readout offset 0.01')
        ax[1].plot(*zip(*pts), '-o', color=c, ms=3, label=lab)
    ax[1].set_xscale('log'); ax[1].set_xlabel('fibers per solution'); ax[1].set_ylabel('probability of detecting a shift')
    ax[1].axhline(0.8, color='k', lw=0.6, ls=':'); ax[1].set_ylim(0, 1.02); ax[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[1].legend(fontsize=7, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, columnspacing=0.8, handlelength=1.5)
    # (c) Route 4 detection vs PSF, phi 0.15, noise 0.03
    for P_mean, c in ((0.25, '#2b6cb0'), (0.75, '#b03a2e')):
        pts = sorted((r['psf_sigma_nm'], r['p_detect_interior_given_fading_memory']) for r in r3
                     if r['phi'] == 0.15 and r['lambda_over_ts'] == 3.0 and r['pixel_noise'] == 0.03 and r['mean_P2'] == P_mean)
        fp = sorted((r['psf_sigma_nm'], r['p_false_interior_given_skin']) for r in r3
                    if r['phi'] == 0.15 and r['lambda_over_ts'] == 3.0 and r['pixel_noise'] == 0.03 and r['mean_P2'] == P_mean)
        ax[2].plot(*zip(*pts), '-o', color=c, ms=3, label=f'memory, {P_mean}')
        ax[2].plot(*zip(*fp), ':s', color=c, ms=3, label=f'skin, {P_mean}')
    pts = sorted((r['psf_sigma_nm'], r['p_detect_interior_given_fading_memory']) for r in r3
                 if r['phi'] == 0.15 and r['lambda_over_ts'] == 1.0 and r['pixel_noise'] == 0.03 and r['mean_P2'] == 0.75)
    ax[2].plot(*zip(*pts), '-.^', color='#b03a2e', ms=3, label='memory, 0.75, fast')
    ax[2].set_xlabel('PSF standard deviation (nm)'); ax[2].set_ylabel('probability of an interior call')
    ax[2].set_ylim(-0.02, 1.02); ax[2].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[2].legend(fontsize=7, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, columnspacing=0.8, handlelength=1.8)
    for a, t in zip(ax, 'abc'):
        a.text(-0.28, 1.02, t, transform=a.transAxes, fontweight='bold', fontsize=9)
    fig.tight_layout(w_pad=2.5)
    os.makedirs(FIG, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'fig7_synthetic_validation.{ext}'), dpi=400)
    print('wrote figures/fig7_synthetic_validation')


def _load(name):
    import csv

    def num(v):
        try:
            return float(v)
        except ValueError:
            return v
    return [{k: num(v) for k, v in r.items()} for r in csv.DictReader(open(os.path.join(RES, name)))]


if __name__ == '__main__' and sys.argv[1:] == ['--figure-only']:
    # redraw Fig. 7 from the result files without rerunning the simulations
    figure(_load('synth_plate_control.csv'), _load('synth_route3.csv'), _load('synth_route4.csv'))
elif __name__ == '__main__':
    r1 = v1_plate_control(); write('synth_plate_control.csv', r1)
    r2 = v2_route3(); write('synth_route3.csv', r2)
    r3 = v3_route4(); write('synth_route4.csv', r3)
    write('synth_robustness.csv', v4_robustness())
    figure(r1, r2, r3)
