"""Reanalysis of the published records: polystyrene (Richard-Lacroix and Pellerin 2015), poly(ethylene terephthalate)
(Laramée, Lanthier and Pellerin 2022) and polyacrylonitrile (Knapp, Laramée and Pellerin 2026), from the values
supplied by the source authors (data/*.csv, built by scripts/import_author_data.py).
Parts (called by scripts/run_results.py): records, psunc, pet, channels, pan."""
import math

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit, isotonic_regression

from common import read, shell_fraction, thinnest_shell, write

# ----------------------------------------------------------------------------------------------- polystyrene
FIT_A, FIT_K = 0.4282, 1.0e-3          # published fit, 0.4282 exp(-0.0010 d), d in nm
WIN = (500.0, 1700.0)                  # diameters below the onset where the bins are tested
D_FIT = np.linspace(*WIN, 1201)        # continuous diameters for the bound against a curve


def ps_bins():
    r = read('richard_lacroix_pellerin_2015_fig2a.csv')
    return (np.array([float(x['diameter_nm']) for x in r]), np.array([float(x['P2']) for x in r]),
            np.array([float(x['half_width']) for x in r]))


def published_fit(d):
    return FIT_A * np.exp(-FIT_K * d)


def expfit(d, p):
    f = lambda x, a, k: a * np.exp(-k * x)
    (a, k), _ = curve_fit(f, d, p, p0=[0.4, 1e-3], maxfev=20000)
    return a, k


def min_skin(d, mean, skin=1.0, core=0.0, kappa=1.0):
    """Thinnest shell supplying mean = core + kappa f (skin - core) (linear mixing)."""
    return thinnest_shell(d, (np.asarray(mean) - core) / (kappa * (skin - core)))


def common_skin(d, mean, **kw):
    """One thickness supplying every value: the largest per-diameter minimum (nan if some value is unreachable)."""
    m = min_skin(d, mean, **kw)
    return float(np.max(m)) if np.all(np.isfinite(m)) else float('nan')


def part_records():
    d, p, hw = ps_bins()
    a, k = expfit(d, p)
    pred = a * np.exp(-k * d)
    r2 = 1 - np.sum((p - pred) ** 2) / np.sum((p - p.mean()) ** 2)
    write('ps_record_refit.csv', ['quantity', 'value'],
          [('prefactor', a), ('decay_constant_per_nm', k), ('R2', r2), ('published_prefactor', FIT_A),
           ('published_decay_constant_per_nm', FIT_K)])
    rows = []
    for di, pi, hi in zip(d, p, hw):
        need = [pi / float(shell_fraction(di, dl)) for dl in (13.1, 30.0, 60.0)]
        rows.append((di, pi, hi, *need, float(min_skin(di, pi)), float(min_skin(di, pi, skin=0.3)),
                     float(min_skin(di, pi / 2)), float(min_skin(di, 0.3 * pi)), float(min_skin(di, 0.2 * pi)),
                     float(min_skin(di, max(pi - hi, 0.0)))))
    write('ps_skin_bound.csv', ['diameter_nm', 'P2', 'half_width', 'skin_P2_needed_13.1nm', 'skin_P2_needed_30nm',
                                'skin_P2_needed_60nm', 'min_skin_nm_if_skin_P2_le_1', 'min_skin_nm_if_skin_P2_le_0.3',
                                'min_skin_nm_half_from_collector_P2_le_1', 'min_skin_nm_70pct_from_collector_P2_le_1',
                                'min_skin_nm_80pct_from_collector_P2_le_1', 'min_skin_nm_mean_minus_sd_P2_le_1'], rows)
    w = (d >= WIN[0]) & (d <= WIN[1])
    fit = published_fit(D_FIT)
    rows = []
    for share in (0.0, 0.5, 0.7, 0.8):
        kept = 1 - share
        per = min_skin(d[w], kept * p[w])
        rows.append((share, float(per.min()), float(per.max()), common_skin(d[w], kept * p[w]),
                     common_skin(D_FIT, kept * fit)))
    write('ps_bound_collector_share.csv', ['collector_share', 'per_bin_min_nm_low', 'per_bin_min_nm_high',
                                           'common_skin_min_nm_bins', 'common_skin_min_nm_fit'], rows)
    rows = []
    for core, skin, kappa in ((0, 1, 1), (0.01, 1, 1), (0.02, 1, 1), (0, 0.3, 1), (0, 1, 0.5), (0, 1, 1.5), (0, 1, 2),
                              (0, 1, 3), (0, 0.3, 3)):
        per = min_skin(d[w], p[w], skin=skin, core=core, kappa=kappa)
        rows.append((core, skin, kappa, float(per.min()), float(per.max()),
                     common_skin(D_FIT, fit, skin=skin, core=core, kappa=kappa)))
    write('ps_common_skin.csv', ['core_P2', 'skin_P2', 'weighting_kappa', 'per_bin_min_nm_low', 'per_bin_min_nm_high',
                                 'common_skin_min_nm_fit'], rows)
    rows = []
    for delta in (13.1, 30.0, 60.0):
        rows.append((delta, float(np.max(fit / shell_fraction(D_FIT, delta))),
                     float(np.max(p[w] / shell_fraction(d[w], delta)))))
    write('ps_kappa_needed.csv', ['skin_nm', 'kappa_needed_fit', 'kappa_needed_bins'], rows)
    part_two_channel(d, p, hw)


def part_two_channel(d, p, hw):
    e = read('richard_lacroix_pellerin_2015_fig5a.csv')
    eid = np.array([float(x['EID_percent']) for x in e])
    ehw = np.array([float(x['half_width']) for x in e])
    assert np.allclose([float(x['diameter_nm']) for x in e], d)
    lin = stats.linregress(p, eid)
    tau = stats.kendalltau(p, eid)
    rho = stats.spearmanr(p, eid)
    disc = []
    n = len(d)
    for i in range(n):
        for j in range(i + 1, n):
            if (p[i] - p[j]) * (eid[i] - eid[j]) < 0:
                disc.append((d[i], d[j], abs(p[i] - p[j]), abs(eid[i] - eid[j]),
                             abs(eid[i] - eid[j]) / math.hypot(ehw[i], ehw[j])))
    write('ps_two_channel.csv', ['quantity', 'value'],
          [('bins', n), ('linear_intercept_percent', lin.intercept), ('linear_slope_percent', lin.slope),
           ('linear_R2', lin.rvalue ** 2), ('kendall_tau', tau.statistic), ('kendall_p', tau.pvalue),
           ('spearman_rho', rho.statistic), ('pairs', n * (n - 1) // 2), ('discordant_pairs', len(disc))])
    write('ps_two_channel_discordant.csv', ['diameter_a_nm', 'diameter_b_nm', 'P2_difference', 'EID_difference',
                                            'EID_difference_over_half_widths_in_quadrature'], disc)


def part_psunc(reps=4000, seed=7):
    """Sampling scenarios: each bin mean drawn as normal with standard error half_width / sqrt(n)."""
    d, p, hw = ps_bins()
    w = (d >= WIN[0]) & (d <= WIN[1])
    rng = np.random.default_rng(seed)
    rows = []
    for n in (6, 9, 15):
        fitb, binb = [], []
        for _ in range(reps):
            q = p + rng.standard_normal(len(p)) * hw / math.sqrt(n)
            try:
                a, k = expfit(d, q)
            except RuntimeError:
                continue
            fitb.append(common_skin(D_FIT, a * np.exp(-k * D_FIT)))
            binb.append(common_skin(d[w], q[w]))
        fitb, binb = np.array(fitb), np.array(binb)
        rows.append((n, reps, len(fitb), np.nanpercentile(fitb, 5), np.nanpercentile(fitb, 50),
                     np.nanpercentile(binb, 5), np.nanpercentile(binb, 50), float(np.mean(~np.isfinite(binb)))))
    write('ps_bound_uncertainty.csv', ['fibers_per_bin', 'replicates', 'successful_fits', 'fit_p05_nm', 'fit_p50_nm',
                                       'bins_p05_nm', 'bins_p50_nm', 'fraction_bins_unreachable'], rows)


# ---------------------------------------------------------------------------------- poly(ethylene terephthalate)
PET_SPOT_NM = 550.0
PET_CEILING, PET_MEDIAN, PET_GAP, PET_DISK = 0.04, 0.02, 0.18, 0.24   # stated in the source (Table S9)
FWHM_PER_XC = 209 * (1.445 - 1.335)   # X_c = ((305 - FWHM)/209 - 1.335)/0.110  ->  dFWHM = 22.99 dX_c


def part_pet():
    rows = []
    for delta in (13.1, 30.0, 60.0):
        phi = float(shell_fraction(PET_SPOT_NM, delta))
        for label, ceiling in (('every fiber (maximum 0.04)', PET_CEILING), ('typical fiber (median 0.02)', PET_MEDIAN)):
            s_max = ceiling / phi
            rows.append((delta, PET_SPOT_NM, phi, label, ceiling, s_max, s_max * phi / PET_GAP, s_max * phi / PET_DISK))
    write('pet_plate_skin_bound.csv', ['skin_thickness_nm', 'diameter_nm', 'shell_volume_fraction', 'plate_statistic',
                                       'plate_orientation_ceiling', 'max_skin_orientation', 'share_of_gap_median',
                                       'share_of_disk_median'], rows)


def pet_fibers():
    r = read('laramee_lanthier_pellerin_2022_fig3b.csv')
    col = np.array([x['collector'] for x in r])
    p2 = np.array([float(x['P2']) for x in r])
    xc = np.array([float(x['X_c']) for x in r])
    ft = np.array([float(x['F_T']) for x in r])
    dia = np.array([float(x['diameter_nm']) if x['diameter_nm'] else np.nan for x in r])
    return col, p2, xc, ft, dia


def exact_bootstrap_median_ci(v, level=0.95):
    """Percentile interval of the bootstrap distribution of the median of an odd sample, computed exactly:
    the resampled median is at most the k-th smallest value when at least (n+1)/2 draws fall at or below it."""
    v = np.sort(v)
    n = len(v)
    cdf = stats.binom.sf((n - 1) // 2, n, np.arange(1, n + 1) / n)
    a = (1 - level) / 2
    return v[np.argmax(cdf >= a)], v[np.argmax(cdf >= 1 - a)]


def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = len(y) - X.shape[1]
    cov = np.linalg.inv(X.T @ X) * (res @ res) / dof
    return beta, np.sqrt(np.diag(cov))


def ols_hc3(y, X):
    """OLS with HC3 heteroskedasticity-consistent standard errors."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    xtxi = np.linalg.inv(X.T @ X)
    h = np.einsum('ij,jk,ik->i', X, xtxi, X)
    u = X * (e / (1 - h))[:, None]
    return beta, np.sqrt(np.diag(xtxi @ (u.T @ u) @ xtxi))


def design(col, p2, extra=()):
    """Common slope in P2 and one intercept per collector; coefficient 2 is disk minus gap."""
    return np.column_stack([p2, np.ones_like(p2), (col == 'disk').astype(float), (col == 'plate').astype(float), *extra])


def linear_offset(col, p2, y, extra=()):
    beta, se = ols(y, design(col, p2, extra))
    return beta[2], se[2]


def matched_offset(col, p2, y, window=0.03, lo=0.05, hi=0.55):
    g = col == 'gap'
    diffs = []
    for i in np.where((col == 'disk') & (p2 >= lo) & (p2 <= hi))[0]:
        m = g & (np.abs(p2 - p2[i]) <= window)
        if m.any():
            diffs.append(y[i] - y[m].mean())
    return np.array(diffs)


def iso_rss(x, y):
    o = np.argsort(x, kind='stable')
    fit = isotonic_regression(y[o]).x
    return float(np.sum((y[o] - fit) ** 2))


def common_function_test(col, p2, y, perms, rng):
    def stat(c):
        pooled = iso_rss(p2, y)
        sep = sum(iso_rss(p2[c == k], y[c == k]) for k in ('plate', 'gap', 'disk'))
        return pooled - sep
    obs = stat(col)
    hits = sum(stat(rng.permutation(col)) >= obs for _ in range(perms))
    return obs, (hits + 1) / (perms + 1)


def common_function_test_stratified(col, p2, y, perms, rng, strata=5):
    """As common_function_test, but collector labels are permuted only within quantile strata of <P2>, so that the
    null distribution keeps each collector's range of orientation."""
    def stat(c):
        return iso_rss(p2, y) - sum(iso_rss(p2[c == k], y[c == k]) for k in ('plate', 'gap', 'disk'))
    edges = np.quantile(p2, np.linspace(0, 1, strata + 1)[1:-1])
    bins = np.searchsorted(edges, p2, side='right')
    obs = stat(col)
    hits = 0
    for _ in range(perms):
        c = col.copy()
        for b in range(strata):
            ix = np.where(bins == b)[0]
            c[ix] = rng.permutation(col[ix])
        hits += stat(c) >= obs
    return obs, (hits + 1) / (perms + 1)


def boot_within(col, rng, reps):
    idx = {k: np.where(col == k)[0] for k in ('plate', 'gap', 'disk')}
    for _ in range(reps):
        yield np.concatenate([rng.choice(v, len(v)) for v in idx.values()])


def part_channels():
    col, p2, xc, ft, dia = pet_fibers()
    fm = ft - xc
    # record check against the statements of the source
    def r2(y):
        return stats.linregress(p2, y).rvalue ** 2
    weak = np.abs(p2) <= 0.01
    write('pet_record_check.csv', ['quantity', 'value'],
          [('fibers, total', len(col)), ('fibers, plate', int((col == 'plate').sum())),
           ('fibers, gap', int((col == 'gap').sum())), ('fibers, disk', int((col == 'disk').sum())),
           ('fibers with a diameter', int(np.isfinite(dia).sum())),
           ('pooled linear R2, F_T against <P2>', r2(ft)), ('pooled linear R2, F_M against <P2>', r2(fm)),
           ('largest gap <P2>', p2[col == 'gap'].max()), ('median X_c', np.median(xc)), ('maximum X_c', xc.max()),
           ('mean X_c, disk', xc[col == 'disk'].mean()), ('fibers with |<P2>| <= 0.01', int(weak.sum())), ('F_T of fibers with |<P2>| <= 0.01, min', ft[weak].min()),
           ('F_T of fibers with |<P2>| <= 0.01, max', ft[weak].max())])
    # plate control by diameter
    rng = np.random.default_rng(11)
    rows = []
    for k in ('plate', 'gap', 'disk'):
        m = (col == k) & (dia < PET_SPOT_NM)
        v = p2[m]
        lo, hi = exact_bootstrap_median_ci(v) if len(v) % 2 else \
            np.percentile([np.median(rng.choice(v, len(v))) for _ in range(20000)], [2.5, 97.5])
        rows.append((k, int(m.sum()), np.median(v), v.max(), lo, hi, PET_CEILING / np.median(v), PET_CEILING / lo))
    # the plate bound at the supplied (nominal) diameters of the collector fibers below 550 nm, volume average:
    # s_skin <= 0.04 / f(largest plate diameter); collector skin share = median over fibers of s_skin f(d_i) / median P2
    dpl = dia[(col == 'plate') & (dia < PET_SPOT_NM)].max()
    for k in ('gap', 'disk'):
        m = (col == k) & (dia < PET_SPOT_NM)
        for delta in (13.1, 30.0, 60.0):
            smax = PET_CEILING / float(shell_fraction(dpl, delta))
            share = float(np.median(smax * shell_fraction(dia[m], delta)) / np.median(p2[m]))
            rows.append((k + ' at nominal diameters, delta %g nm' % delta, int(m.sum()), '', '', '', '', share, ''))
    nodia = [(c, x) for c, x, dd in zip(col, p2, dia) if not np.isfinite(dd)]
    write('pet_plate_check.csv', ['collector', 'fibers_below_550nm', 'median_P2', 'max_P2', 'median_ci_low',
                                  'median_ci_high', 'stated_ceiling_share_of_median', 'stated_ceiling_share_at_ci_low'],
          rows)
    write('pet_no_diameter.csv', ['collector', 'P2'], nodia)
    # sensitivity of the plate bound to the diameter cut. The supplied diameters are nominal widths that overestimate
    # true diameters below the spot, so the 550 nm cut is not exact. For each cut: largest plate value below it (the
    # data ceiling) and its share of the gap and disk medians below the same cut, at matched diameters.
    rows = []
    for cut in (500.0, 525.0, 550.0, 575.0, 600.0, 625.0, 650.0):
        pl, g, dk = [(col == k) & (dia < cut) for k in ('plate', 'gap', 'disk')]
        mx = p2[pl].max()
        rows.append((cut, int(pl.sum()), mx, float(dia[pl][np.argmax(p2[pl])]), int(g.sum()), np.median(p2[g]),
                     int(dk.sum()), np.median(p2[dk]), mx / np.median(p2[g]), mx / np.median(p2[dk])))
    write('pet_plate_cut_sensitivity.csv', ['cut_nm', 'plate_fibers', 'plate_max_P2', 'diameter_of_plate_max_nm',
                                            'gap_fibers', 'gap_median_P2', 'disk_fibers', 'disk_median_P2',
                                            'share_of_gap_median', 'share_of_disk_median'], rows)
    # frequentist coverage of the order-statistic intervals for the population median: X(a) <= m <= X(b) holds when
    # a <= B <= b - 1, B ~ binomial(n, 1/2); the next wider pair (a - 1, b + 1) is reported alongside
    rows = []
    for k in ('gap', 'disk'):
        v = np.sort(p2[(col == k) & (dia < PET_SPOT_NM)])
        n = len(v)
        lo, hi = exact_bootstrap_median_ci(v)
        a, b = int(np.searchsorted(v, lo)) + 1, int(np.searchsorted(v, hi)) + 1
        for aa, bb in ((a, b), (a - 1, b + 1)):
            cov = float(stats.binom.cdf(bb - 1, n, 0.5) - stats.binom.cdf(aa - 1, n, 0.5))
            rows.append((k, n, aa, bb, v[aa - 1], v[bb - 1], cov, PET_CEILING / v[aa - 1]))
    write('pet_median_interval_coverage.csv', ['collector', 'fibers', 'lower_order', 'upper_order', 'lower_P2',
                                               'upper_P2', 'coverage_of_population_median',
                                               'stated_ceiling_share_at_lower'], rows)
    # chance that at least one of the plate fibers without a diameter lies below the cut, if they are exchangeable
    # with the plate fibers that have one (hypergeometric, sampling without replacement)
    sized = dia[(col == 'plate') & np.isfinite(dia)]
    n, k, u = len(sized), int((sized < PET_SPOT_NM).sum()), int(((col == 'plate') & ~np.isfinite(dia)).sum())
    p_none = float(np.prod([(n - k - i) / (n - i) for i in range(u)]))
    write('pet_unsized_plate.csv', ['quantity', 'value'],
          [('plate fibers with a diameter', n), ('of which below 550 nm', k), ('plate fibers without a diameter', u),
           ('P(at least one unsized fiber below 550 nm | exchangeable)', 1 - p_none)])
    # channels
    rng = np.random.default_rng(12)
    rows = []
    for name, y in (('F_T', ft), ('X_c', xc), ('F_M', fm)):
        off, se = linear_offset(col, p2, y)
        bl = np.array([linear_offset(col[i], p2[i], y[i])[0] for i in boot_within(col, rng, 4000)])
        md = matched_offset(col, p2, y)
        bm = []
        for i in boot_within(col, rng, 2000):
            v = matched_offset(col[i], p2[i], y[i])
            bm.append(v.mean())
        stat, pval = common_function_test(col, p2, y, 1000, rng)
        rows.append((name, off, se, *np.percentile(bl, [2.5, 97.5]), md.mean(), len(md),
                     *np.percentile(bm, [2.5, 97.5]), stat, pval))
    # heteroskedasticity-consistent (HC3) intervals for the same linear offsets, with and without the plate fibers;
    # an independent check of the bootstrap intervals that, like them, treats fibers as independent
    rows_h = []
    for setname, sub in (('all collectors', np.ones(len(col), bool)), ('disk and gap only', col != 'plate')):
        X = design(col[sub], p2[sub])
        X = X[:, np.any(X != 0, axis=0)]                  # drop the plate column when no plate fibers are present
        for name, y in (('F_T', ft), ('X_c', xc), ('F_M', fm)):
            b, se = ols_hc3(y[sub], X)
            q = stats.t.ppf(0.975, int(sub.sum()) - X.shape[1])
            rows_h.append((setname, name, int(sub.sum()), b[2], se[2], b[2] - q * se[2], b[2] + q * se[2]))
    rng_s = np.random.default_rng(31)
    rows_s = [(name, *common_function_test_stratified(col, p2, y, 1000, rng_s)) for name, y in
              (('F_T', ft), ('X_c', xc), ('F_M', fm))]
    write('pet_channels_stratified.csv', ['channel', 'rss_reduction_separate_isotonic',
                                          'permutation_p_within_P2_quintiles'], rows_s)
    write('pet_channels_hc3.csv', ['fibers', 'channel', 'n', 'disk_minus_gap_offset', 'hc3_se', 'ci_low', 'ci_high'],
          rows_h)
    write('pet_channels.csv', ['channel', 'disk_minus_gap_offset_equal_P2', 'offset_ols_se', 'offset_ci_low',
                               'offset_ci_high', 'matched_disk_minus_gap', 'matched_disk_fibers', 'matched_ci_low',
                               'matched_ci_high', 'rss_reduction_separate_isotonic', 'permutation_p_common_function'],
          rows)
    rows = []
    for win in (0.02, 0.03, 0.04, 0.05):
        md = matched_offset(col, p2, xc, window=win)
        rows.append(('all disk fibers', win, len(md), float(np.mean(md > 0)), float(np.median(md)), float(md.mean())))
    keep = ~((col == 'disk') & (xc > 0.08))
    for win in (0.02, 0.03, 0.04, 0.05):
        md = matched_offset(col[keep], p2[keep], xc[keep], window=win)
        rows.append(('disk fibers with X_c <= 0.08', win, len(md), float(np.mean(md > 0)), float(np.median(md)),
                     float(md.mean())))
    write('pet_channels_robustness.csv', ['sample', 'window', 'disk_fibers', 'fraction_positive', 'median_excess',
                                          'mean_excess'], rows)
    # diameter-adjusted offset
    m = np.isin(col, ('gap', 'disk')) & np.isfinite(dia)
    rng = np.random.default_rng(13)
    rows = []
    for label, extra in (('without diameter', ()), ('with log diameter', (np.log(np.where(m, dia, 1.0)),))):
        def est(i):
            X = np.column_stack([p2[i], np.ones(len(i)), (col[i] == 'disk').astype(float), *[e[i] for e in extra]])
            return ols(xc[i], X)[0][2]
        idx = np.where(m)[0]
        b = []
        gi, di = idx[col[idx] == 'gap'], idx[col[idx] == 'disk']
        for _ in range(4000):
            b.append(est(np.concatenate([rng.choice(gi, len(gi)), rng.choice(di, len(di))])))
        rows.append((label, len(idx), est(idx), *np.percentile(b, [2.5, 97.5])))
    write('pet_offset_diameter.csv', ['model', 'fibers', 'X_c_offset', 'ci_low', 'ci_high'], rows)
    # session clustering
    rows = []
    n_gd = int(np.isin(col, ('gap', 'disk')).sum())
    off_l = linear_offset(col, p2, xc)[0]
    md = matched_offset(col, p2, xc)
    rngm = np.random.default_rng(14)
    se_l = float(np.std([linear_offset(col[i], p2[i], xc[i])[0] for i in boot_within(col, rngm, 4000)], ddof=1))
    se_m = float(np.std([matched_offset(col[i], p2[i], xc[i]).mean() for i in boot_within(col, rngm, 2000)], ddof=1))
    for est_name, est, se in (('linear model', off_l, se_l), ('matched comparison', md.mean(), se_m)):
        deff = (est / (1.959964 * se)) ** 2
        for k in (2, 4):
            mm = n_gd / (2 * k)
            tq = stats.t.ppf(0.975, 2 * k - 2)                 # sessions as units: t on 2G - 2 degrees of freedom
            deff_t = (est / (tq * se)) ** 2
            rows.append((est_name, est, se, deff, k, mm, (deff - 1) / (mm - 1), 2 * k - 2, deff_t,
                         (deff_t - 1) / (mm - 1)))
    write('pet_cluster_sensitivity.csv', ['estimator', 'offset', 'bootstrap_standard_error', 'design_effect_to_lose_significance',
                                          'sessions_per_collector', 'fibers_per_session', 'icc_needed', 'session_t_dof',
                                          'design_effect_to_lose_significance_t', 'icc_needed_t'], rows)
    # error model
    rng = np.random.default_rng(15)
    rows = [('FWHM_1725 per unit X_c (cm-1)', FWHM_PER_XC),
            ('FWHM_1725 equivalent of linear offset (cm-1)', FWHM_PER_XC * off_l),
            ('FWHM_1725 equivalent of matched offset (cm-1)', FWHM_PER_XC * md.mean())]
    for sd in (0.005, 0.01):
        v = [linear_offset(col, p2, xc + rng.normal(0, sd, len(xc)))[0] for _ in range(500)]
        rows += [('X_c noise %g: median offset' % sd, np.median(v)), ('X_c noise %g: 2.5th percentile' % sd,
                  np.percentile(v, 2.5)), ('X_c noise %g: 97.5th percentile' % sd, np.percentile(v, 97.5))]
    for sd in (0.02, 0.05):
        v = [linear_offset(col, p2 + rng.normal(0, sd, len(p2)), xc)[0] for _ in range(500)]
        rows += [('P2 noise %g: median offset' % sd, np.median(v))]
    # attenuation: with reliability R the true slope is b/R and the offset becomes off - b (1/R - 1) dP2
    b = ols(xc, design(col, p2))[0][0]
    dp = p2[col == 'disk'].mean() - p2[col == 'gap'].mean()
    rel = 1.0 / (1.0 + off_l / (b * dp))
    rows += [('slope of X_c on P2', b), ('disk minus gap mean P2', dp),
             ('reliability of <P2> at which the offset vanishes', rel),
             ('R2 of F_T on P2 (lower bound on reliability)', r2(ft))]
    write('pet_error_model.csv', ['quantity', 'value'], rows)


# ------------------------------------------------------------------------------------------- polyacrylonitrile
PAN_STATED = {'oil': (0.37, 0.21), 'calibrated': (0.40, 0.19)}   # medians below 600 nm and above the onset [S32]
PAN_MARGIN = 0.05


def part_pan():
    r = read('knapp_laramee_pellerin_2026_fig3.csv')
    rng = np.random.default_rng(21)
    rows = []
    for med in ('oil', 'water'):
        sm = np.array([float(x['P2_four_spectrum']) for x in r if x['medium'] == med])
        ca = np.array([float(x['P2_calibrated']) for x in r if x['medium'] == med])
        dlt = sm - ca
        bs = np.array([rng.choice(dlt, len(dlt)).mean() for _ in range(4000)])
        lin = stats.linregress(ca, sm)
        rows.append((med, len(dlt), dlt.mean(), *np.percentile(bs, [2.5, 97.5]), dlt.std(ddof=1),
                     math.sqrt(np.mean(dlt ** 2)), stats.pearsonr(ca, sm).statistic, stats.spearmanr(ca, sm).statistic,
                     lin.slope, lin.intercept, float(np.mean(np.abs(dlt) <= 0.10)), sm.min(), sm.max(), ca.min(),
                     ca.max()))
        if med == 'oil':
            lo90, hi90 = np.percentile(bs, [5, 95])
            sl = stats.linregress(ca, dlt)
            bsl = []
            for _ in range(4000):
                i = rng.integers(0, len(ca), len(ca))
                bsl.append(stats.linregress(ca[i], dlt[i]).slope)
            eq = [('fibers', len(dlt)), ('mean difference', dlt.mean()), ('mean difference, 90% interval low', lo90),
                  ('mean difference, 90% interval high', hi90), ('equivalence margin', PAN_MARGIN),
                  ('equivalent within margin', bool(-PAN_MARGIN < lo90 and hi90 < PAN_MARGIN)),
                  ('slope of difference on calibrated', sl.slope),
                  ('slope 95% interval low', np.percentile(bsl, 2.5)), ('slope 95% interval high', np.percentile(bsl, 97.5))]
    write('pan_readout_agreement.csv', ['medium', 'fibers', 'mean_SM_minus_calib', 'mean_ci_low', 'mean_ci_high',
                                        'sd_difference', 'rms_difference', 'pearson_r', 'spearman_rho', 'slope',
                                        'intercept', 'fraction_within_0.10', 'SM_min', 'SM_max', 'calib_min',
                                        'calib_max'], rows)
    write('pan_equivalence.csv', ['quantity', 'value'], eq)
    air = read('knapp_laramee_pellerin_2026_fig3_air.csv')
    sm = np.array([float(x['P2_four_spectrum']) for x in air])
    ca = np.array([float(x['P2_calibrated']) for x in air])
    write('pan_air_check.csv', ['quantity', 'value'],
          [('fibers', len(sm)), ('four-spectrum negative', int((sm < 0).sum())), ('four-spectrum below -0.5',
           int((sm < -0.5).sum())), ('calibrated min', ca.min()), ('calibrated max', ca.max())])
    ca_oil = np.array([float(x['P2_calibrated']) for x in r if x['medium'] == 'oil'])
    rows = []
    for src, (m, c) in PAN_STATED.items():
        for s in (1.0, round(float(ca_oil.max()), 2)):   # most oriented calibrated fiber, 0.57
            f = (m - c) / (s - c)
            for d in (400.0, 500.0, 600.0):
                rows.append((src, m, c, s, f, d, float(thinnest_shell(d, f))))
    write('pan_skin_bound.csv', ['medians', 'median_below_600nm', 'core_median_above_onset', 'skin_orientation',
                                 'min_shell_fraction', 'diameter_nm', 'min_skin_nm'], rows)
