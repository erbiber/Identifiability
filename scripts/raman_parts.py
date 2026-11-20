"""Result parts for the confocal readout weight (called by scripts/run_results.py):
raman (series solution, optical bound, tilt, checks) and raman_check (finite differences with the BaF2 window)."""
from concurrent.futures import ProcessPoolExecutor
import os

import numpy as np

import raman_fd as fd
import raman_weighting as rw
import records
from common import write

D_TABLE = (500.0, 700.0, 1000.0, 1700.0)
D_FIT = tuple(np.arange(500.0, 1701.0, 10.0))
DELTAS = (13.1, 30.0, 60.0)
WORKERS = max(1, min(2, os.cpu_count() or 1))


def thinnest(share, target, a):
    """Smallest delta with share(delta) >= target (bisection; nan if even the whole cross-section is too little)."""
    if share(a) < target:
        return float('nan')
    lo, hi = 0.0, a
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if share(mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def series_share(d_nm, beam):
    r, wz, wx = rw.weights(d_nm * 1e-9, **beam)
    a = d_nm * 1e-9 / 2
    return lambda delta: max(rw.skin_share(r, wz, a, delta), rw.skin_share(r, wx, a, delta)), r, wz, wx


def optical_bound(share_of, ds, targets, a_of):
    """Thinnest single skin (nm) supplying every target value at the listed diameters."""
    out = []
    for d, t in zip(ds, targets):
        out.append(thinnest(share_of[d], t, a_of(d)) * 1e9)
    return float(np.max(out)) if np.all(np.isfinite(out)) else float('nan')


def part_raman():
    dbins, pbins, _ = records.ps_bins()
    w = (dbins >= 500) & (dbins <= 1700)
    dbins, pbins = tuple(dbins[w]), pbins[w]
    krows, brows, scan = [], [], []
    for label, beam in rw.SETTINGS.items():
        shares = {}
        for d in sorted(set(D_TABLE) | set(D_FIT) | set(dbins)):
            s, r, wz, wx = series_share(d, beam)
            shares[d] = s
            f30 = rw.area_fraction(d * 1e-9 / 2, 30e-9)
            scan.append((label, int(beam.get('na', 0) > 0), d, s(30e-9) / f30))
            if d in D_TABLE or d in dbins:
                a = d * 1e-9 / 2
                for delta in DELTAS:
                    f = rw.area_fraction(a, delta * 1e-9)
                    kz = rw.skin_share(r, wz, a, delta * 1e-9) / f
                    kx = rw.skin_share(r, wx, a, delta * 1e-9) / f
                    for pol, k in (('Z', kz), ('X', kx), ('max', max(kz, kx))):
                        krows.append((label, int(beam.get('na', 0) > 0), pol, delta, d, k))
        for s_skin in (1.0, 0.3):
            fit = optical_bound(shares, D_FIT, records.published_fit(np.array(D_FIT)) / s_skin, lambda d: d * 1e-9 / 2)
            bins = optical_bound(shares, dbins, pbins / s_skin, lambda d: d * 1e-9 / 2)
            brows.append((label, int(beam.get('na', 0) > 0), s_skin, fit, bins))
    write('raman_weighting_kappa.csv', ['setting', 'focused', 'polarization', 'delta_nm', 'd_nm', 'kappa'], krows)
    write('raman_kappa_scan.csv', ['setting', 'focused', 'd_nm', 'kappa30_max'], scan)
    write('ps_optical_bound.csv', ['setting', 'focused', 's_skin', 'fit_max_nm', 'per_bin_max_nm'], brows)
    # tilt along the fiber axis (NA 0.8, centred), 30 nm shell
    rows = []
    for d in np.arange(500.0, 1701.0, 100.0):
        best = (0.0, 0.0)
        for alpha in np.radians(np.arange(0.0, 53.01, 1.0)):
            dd, mm = rw.tilted(d * 1e-9, alpha)
            k = max(rw.kappa(dd, 30e-9 * np.cos(alpha), m=mm, na=0.8))
            best = max(best, (k, np.degrees(alpha)))
        rows.append((d, best[0], best[1]))
    write('raman_tilt.csv', ['d_nm', 'kappa30_max', 'alpha_deg_at_max'], rows)
    # checks
    z, x, x_exp = rw.small_cylinder_check()
    ctm, cte = rw.surface_continuity()
    base = max(rw.kappa(1000e-9, 30e-9, na=0.8))
    # convergence: radial and angular grids doubled
    r1, p1, z1, x1 = rw.internal_fields(1000e-9, rw.LAMBDA_EXC, na=0.8, nr=480, nphi=512)
    r2, p2, z2, x2 = rw.internal_fields(1000e-9, rw.LAMBDA_STOKES, na=0.8, nr=480, nphi=512)
    a = 500e-9
    f = rw.area_fraction(a, 30e-9)
    fine_k = max(rw.skin_share(r1, (z1 * z2).mean(1), a, 30e-9), rw.skin_share(r1, (x1 * x2).mean(1), a, 30e-9)) / f
    write('raman_field_checks.csv', ['check', 'value'],
          [('small_Z', z), ('small_X', x), ('small_X_expected', x_exp), ('surface_continuity_TM', ctm),
           ('surface_continuity_TE', cte), ('kappa30_1um_NA0.8', base), ('kappa30_1um_NA0.8_doubled_grid', fine_k),
           ('relative_change_doubled_grid', fine_k / base - 1)])


def _fd_case(args):
    d, na, n_sub = args
    G, frac, rr, W = fd.weight_map(d * 1e-9, na, n_sub=n_sub)
    out = {}
    for delta in np.arange(5.0, 400.0, 5.0):
        out[delta] = (fd.shell_share(G, frac, rr, W['Z'], delta * 1e-9), fd.shell_share(G, frac, rr, W['X'], delta * 1e-9))
    return args, out


def part_raman_check():
    dbins, pbins, _ = records.ps_bins()
    w = (dbins >= 500) & (dbins <= 1700)
    dbins, pbins = dbins[w], pbins[w]
    ds = sorted(set(D_TABLE) | set(dbins))
    cases = [(d, na, fd.N_SUB) for na in (0.8, 0.5) for d in ds] + [(d, 0.8, 1.0) for d in D_TABLE]
    with ProcessPoolExecutor(WORKERS) as ex:
        R = dict(ex.map(_fd_case, cases))
    rows = []
    for (d, na, ns), out in R.items():
        f = rw.area_fraction(d / 2, 30.0)
        kz, kx = out[30.0][0] / f, out[30.0][1] / f
        ser = rw.kappa(d * 1e-9, 30e-9, na=na)
        rows.append(('substrate' if ns > 1 else 'no substrate', na, d, kz, kx, max(kz, kx), max(ser)))
    write('raman_substrate.csv', ['case', 'NA', 'd_nm', 'kappa30_fd_Z', 'kappa30_fd_X', 'kappa30_fd_substrate',
                                  'kappa30_series_no_substrate'], rows)
    brows = []
    for na in (0.8, 0.5):
        def share_of(d):
            out = R[(d, na, fd.N_SUB)]
            deltas = np.array(sorted(out))
            vals = np.array([max(out[x]) for x in deltas])
            return lambda delta: float(np.interp(delta * 1e9, deltas, vals))
        shares = {d: share_of(d) for d in dbins}
        fit = optical_bound(shares, list(dbins), records.published_fit(dbins), lambda d: d * 1e-9 / 2)
        bins = optical_bound(shares, list(dbins), pbins, lambda d: d * 1e-9 / 2)
        brows.append((na, 1.0, fit, bins))
    write('ps_optical_bound_substrate.csv', ['NA', 's_skin', 'fit_max_nm', 'per_bin_max_nm'], brows)
