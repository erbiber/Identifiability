import numpy as np
from scipy.optimize import curve_fit
WINDOW = (100.0, 2000.0)
Z = 1.959964 + 0.841621

def shell_fraction(d, delta):
    return np.clip(1.0 - (1.0 - 2.0 * delta / d) ** 2, 0.0, 1.0)

# Poly(ethylene terephthalate) static-plate record (Laramee, Lanthier & Pellerin 2022): no fiber thinner than the
# 550 nm laser spot exceeds <P2> = 0.04 (median 0.02); medians below 550 nm are 0.18 (gap) and 0.24 (disk).
PET_SPOT_NM, PET_CEILING, PET_MEDIAN, PET_GAP, PET_DISK = 550.0, 0.04, 0.02, 0.18, 0.24


def pet_skin_bound(delta, ceiling=PET_CEILING, d=PET_SPOT_NM):
    """Largest orientation a flight-set skin of thickness delta can carry under the plate ceiling.
    The shell fraction is evaluated at the laser spot, an underestimate for thinner fibers, so the bound is conservative."""
    return np.minimum(1.0, ceiling / shell_fraction(d, delta))


def exponential(d, A, d0, c):
    return A * np.exp(-d / d0) + c

def report(delta, sigmas=(0.05, 0.1, 0.15), n=400):
    d = np.linspace(*WINDOW, n)
    f = shell_fraction(d, delta)
    p, _ = curve_fit(exponential, d, f, p0=[f[0], 500.0, 0.0], maxfev=100000)
    res = f - exponential(d, *p)
    r2 = 1.0 - np.sum(res ** 2) / np.sum((f - f.mean()) ** 2)
    rms = np.sqrt(np.mean(res ** 2))
    print('delta = %2.0f nm | R^2 = %.4f | max|res| = %.4f (%.1f%% of range) | rms = %.4f' % (delta, r2, np.abs(res).max(), 100 * np.abs(res).max() / np.ptp(f), rms))
    for s in sigmas:
        print('    dispersion %.2f -> total N = %.0f fibers' % (s, Z ** 2 * s ** 2 / rms ** 2))

def free_delta(lo=10.0, hi=80.0, n=71):
    d = np.linspace(*WINDOW, 400)
    best = (0.0, None, None)
    for delta in np.linspace(lo, hi, n):
        f = shell_fraction(d, delta)
        try:
            p, _ = curve_fit(exponential, d, f, p0=[f[0], 500.0, 0.0], maxfev=200000)
        except Exception:
            continue
        r2 = 1.0 - np.sum((f - exponential(d, *p)) ** 2) / np.sum((f - f.mean()) ** 2)
        if r2 > best[0]:
            best = (r2, delta, p[1])
    return best

def crossings(delta=30.0, n=400):
    d = np.linspace(*WINDOW, n)
    f = shell_fraction(d, delta)
    p, _ = curve_fit(exponential, d, f, p0=[f[0], 500.0, 0.0], maxfev=200000)
    g = exponential(d, *p)
    k = np.where(np.diff(np.sign(f - g)) != 0)[0]
    from scipy.optimize import brentq
    h = lambda x: shell_fraction(x, delta) - exponential(x, *p)
    xs = np.array([brentq(h, d[i], d[i + 1]) for i in k])
    return (xs, shell_fraction(xs, delta), np.abs(f - g).max() / np.ptp(f))

def straddles(delta=30.0, n=400, window=3):
    d = np.linspace(*WINDOW, n)
    f = shell_fraction(d, delta)
    p, _ = curve_fit(exponential, d, f, p0=[f[0], 500.0, 0.0], maxfev=200000)
    g = exponential(d, *p)
    out = []
    for k in np.where(np.diff(np.sign(f - g)) != 0)[0]:
        s0 = f[k]
        lo, hi = (max(0, k - window), min(n, k + window + 1))
        out.append((s0, bool(np.all((f[lo:k + 1] >= s0) & (g[lo:k + 1] >= s0)) and np.all((f[k + 1:hi] <= s0) & (g[k + 1:hi] <= s0)))))
    return out
from scipy.optimize import minimize as _minimize

def kernel_profile(r, tf_exp, lam_ratio, eps=1.0, n=240):
    out = np.empty_like(r)
    for i, ri in enumerate(r):
        tf = ri ** (-tf_exp)
        t = np.linspace(0.0, tf, n)
        lam = np.exp(np.log(lam_ratio) * (t / tf))
        th_r = np.concatenate([[0.0], np.cumsum(np.diff(t) / lam[:-1])])
        out[i] = np.trapezoid(np.exp(-(th_r[-1] - th_r)), eps * t)
    return out

def kernel_vs_shell_free_history(lam_v, t0=0.5, w=0.05, npts=300):
    r = np.linspace(0.05, 1.0, npts)
    burst = lambda tt: np.exp(-0.5 * ((tt - t0) / w) ** 2) / (w * np.sqrt(2 * np.pi))
    p = np.empty_like(r)
    for i, ri in enumerate(r):
        tt = np.linspace(0.0, 1.0 / ri, 4000)
        p[i] = np.trapezoid(burst(tt) * np.exp(-(tt[-1] - tt) / lam_v), tt)
    rng = np.max(p) - np.min(p)
    if rng < 1e-300:
        return float('nan')
    p = (p - np.min(p)) / rng
    return min((float(np.sqrt(np.mean((p - (r > 1 - d).astype(float)) ** 2))) for d in np.linspace(0.01, 0.6, 120)))

def kernel_vs_shell(delta, npts=160):
    r = np.linspace(0.02, 1.0, npts)
    target = (r > 1.0 - delta).astype(float)

    def loss(p):
        m = kernel_profile(r, abs(p[0]), np.exp(p[1]))
        rng = np.max(m) - np.min(m)
        m = (m - np.min(m)) / (rng + 1e-12)
        return float(np.mean((p[2] * m + p[3] - target) ** 2))
    best = None
    for s in ([0.5, -2.0, 1.0, 0.0], [1.5, -4.0, 1.0, 0.0], [0.2, -1.0, 1.0, 0.0], [3.0, -6.0, 1.0, 0.0]):
        res = _minimize(loss, s, method='Nelder-Mead', options=dict(maxiter=500, fatol=1e-07))
        if best is None or res.fun < best.fun:
            best = res
    return float(np.sqrt(best.fun))

def shell_map(ws=(0.05, 0.2, 1.0, 3.0, 10.0), lams=(0.03, 0.1, 0.3, 1.0, 3.0), diverge_p=None, npts=240):
    r = np.linspace(0.05, 1.0, npts)
    tfs = 1.0 / r
    grid = {}
    for w in ws:
        for lam0 in lams:
            best = np.inf
            for t0 in (0.3, 0.6):
                prof = np.empty_like(r)
                for i, T in enumerate(tfs):
                    tt = np.linspace(0.0, T * (1 - 1e-06), 3000)
                    if diverge_p is None:
                        th = tt / lam0
                    else:
                        lam = lam0 / np.power(np.clip(1 - tt / T, 1e-09, None), diverge_p)
                        th = np.concatenate([[0.0], np.cumsum(np.diff(tt) / lam[:-1])])
                    e = np.exp(-0.5 * ((tt - t0) / w) ** 2)
                    prof[i] = np.trapezoid(e * np.exp(-(th[-1] - th)), tt)
                rng = np.max(prof) - np.min(prof)
                if rng < 1e-300:
                    continue
                prof = (prof - np.min(prof)) / rng
                best = min(best, min((float(np.sqrt(np.mean((prof - (r > 1 - d).astype(float)) ** 2))) for d in np.linspace(0.01, 0.6, 120))))
            grid[w, lam0] = best
    return grid

def containment_bayes_factor(delta_true=0.15, sigma=0.08, npts=60, seed=4, upper=(0.0, 1.5), lower=(-0.5, 1.0), nd=1200):
    from scipy.special import logsumexp
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    r = np.linspace(0.02, 1.0, npts)
    data = (r > 1 - delta_true).astype(float) + rng.normal(0.0, sigma, npts)
    const = -npts * np.log(sigma * np.sqrt(2 * np.pi))

    def ll(pred):
        return -0.5 * np.sum((data - pred) ** 2) / sigma ** 2 + const

    def level(y, lo, hi):
        if y.size == 0:
            return 0.0
        m, S, s = (y.mean(), np.sum((y - y.mean()) ** 2), sigma / np.sqrt(y.size))
        mass = norm.cdf((hi - m) / s) - norm.cdf((lo - m) / s)
        return -0.5 * S / sigma ** 2 + np.log(s * np.sqrt(2 * np.pi)) + np.log(max(mass, 1e-300)) - np.log(hi - lo)
    ds = np.linspace(0.02, 0.6, nd)
    lzg = logsumexp([ll((r > 1 - d).astype(float)) for d in ds]) - np.log(nd)
    lzf = logsumexp([level(data[r > 1 - d], *upper) + level(data[r <= 1 - d], *lower) + const for d in ds]) - np.log(nd)
    return float(np.exp(lzg - lzf))

def a5_element_injectivity(lam0=0.5, tau=0.3, nx=40, ne=40, D=0.01):
    from scipy.spatial import cKDTree

    def element(x, eps, n=4000):
        tt = np.linspace(0.0, 60.0, n)
        depth = (1 - x) * np.exp(-eps * tt / 2)
        front = np.sqrt(D * tt)
        k = int(np.argmax(front >= depth)) if np.any(front >= depth) else n - 1
        k = max(k, 1)
        seg = tt[:k + 1]
        return (eps * tt[k], np.trapezoid(1.0 / (lam0 * np.exp(-seg / tau)), seg))
    xs = np.linspace(0.05, 0.95, nx)
    es = np.linspace(0.3, 3.0, ne)
    G = np.array([[element(x, e) for x in xs] for e in es])
    Tp, Tr = (np.log(G[..., 0]), np.log(G[..., 1]))
    J = (np.gradient(Tp, xs, axis=1) * np.gradient(Tr, es, axis=0) - np.gradient(Tp, es, axis=0) * np.gradient(Tr, xs, axis=1))[1:-1, 1:-1]
    P = np.stack([Tp.ravel(), Tr.ravel()], 1)
    P = (P - P.min(0)) / np.ptp(P, 0)
    pts = np.stack(np.meshgrid(es, xs, indexing='ij'), -1).reshape(-1, 2)
    pts = (pts - pts.min(0)) / np.ptp(pts, 0)
    d, idx = cKDTree(P).query(P, k=2)
    coll = int(((d[:, 1] < 0.01) & (np.linalg.norm(pts[idx[:, 1]] - pts, axis=1) > 0.15)).sum())
    return (int((J > 1e-06).sum()), int((J < -1e-06).sum()), coll)

def committed_levels_bf(hi, lo, delta_true=0.15, sigma=0.08, npts=60, seed=4, nd=1200):
    from scipy.special import logsumexp
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    r = np.linspace(0.02, 1.0, npts)
    data = (r > 1 - delta_true).astype(float) + rng.normal(0.0, sigma, npts)
    const = -npts * np.log(sigma * np.sqrt(2 * np.pi))

    def level(y, a, b):
        m, S, s_ = (y.mean(), np.sum((y - y.mean()) ** 2), sigma / np.sqrt(y.size))
        return -0.5 * S / sigma ** 2 + np.log(s_ * np.sqrt(2 * np.pi)) + np.log(max(norm.cdf((b - m) / s_) - norm.cdf((a - m) / s_), 1e-300)) - np.log(b - a)
    ds = np.linspace(0.02, 0.6, nd)
    lzf = logsumexp([level(data[r > 1 - d], 0.0, 1.5) + level(data[r <= 1 - d], -0.5, 1.0) + const for d in ds]) - np.log(nd)
    lzg = logsumexp([-0.5 * np.sum((data - np.where(r > 1 - d, hi, lo)) ** 2) / sigma ** 2 + const for d in ds]) - np.log(nd)
    return float(np.exp(lzg - lzf))

def realistic_skin_match(t0, w, lam, npts=240):
    r = np.linspace(0.05, 1.0, npts)
    tf = 1.0 + 19.0 * (1 - r) ** 2 / np.max((1 - r) ** 2)
    p = np.empty_like(r)
    for i, T in enumerate(tf):
        tt = np.linspace(0.0, T, 3000)
        p[i] = np.trapezoid(np.exp(-0.5 * ((tt - t0) / w) ** 2) * np.exp(-(T - tt) / lam), tt)
    rng = np.max(p) - np.min(p)
    if rng < 1e-12:
        return float('nan')
    p = (p - np.min(p)) / rng
    return min((float(np.sqrt(np.mean((p - 0.5 * (1 + np.tanh((r - (1 - d)) / e))) ** 2))) for d in np.linspace(0.01, 0.6, 60) for e in (0.005, 0.02, 0.05, 0.1, 0.2)))

def a5_label_ambiguity(r0_spread, npts=3000, D=0.01, seed=3, pair_tol=0.01):
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)

    def el(x, eps, R0, lam0=0.5, tau=0.3, n=3000):
        tt = np.linspace(0.0, 60.0, n)
        depth = (1 - x) * R0 * np.exp(-eps * tt / 2)
        k = int(np.argmax(np.sqrt(D * tt) >= depth)) if np.any(np.sqrt(D * tt) >= depth) else n - 1
        k = max(k, 1)
        seg = tt[:k + 1]
        return (eps * tt[k], np.trapezoid(1.0 / (lam0 * np.exp(-seg / tau)), seg), R0 * np.exp(-eps * tt[k] / 2))
    X = rng.uniform(0.05, 0.95, npts)
    E = rng.uniform(0.3, 3.0, npts)
    R0 = np.exp(rng.normal(0.0, r0_spread, npts))
    o = np.array([el(x, e, r) for x, e, r in zip(X, E, R0)])
    P = np.log(o[:, :2])
    P = (P - P.min(0)) / np.ptp(P, 0)
    L = np.stack([(X - 0.05) / 0.9, np.log(o[:, 2]) / 6.0], 1)
    d, idx = cKDTree(P).query(P, k=2)
    close = d[:, 1] < pair_tol
    ld = np.linalg.norm(L[idx[:, 1]] - L, axis=1)[close]
    return (float(np.median(ld)), float(np.percentile(ld, 90)))

def a5_under_splitting(r0_spread, npts=2500, D=0.01, seed=1):
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)

    def el(x, eps, R0, lam0=0.5, tau=0.3, n=3000):
        tt = np.linspace(0.0, 60.0, n)
        depth = (1 - x) * R0 * np.exp(-eps * tt / 2)
        k = int(np.argmax(np.sqrt(D * tt) >= depth)) if np.any(np.sqrt(D * tt) >= depth) else n - 1
        k = max(k, 1)
        seg = tt[:k + 1]
        return (eps * tt[k], np.trapezoid(1.0 / (lam0 * np.exp(-seg / tau)), seg), R0 * np.exp(-eps * tt[k] / 2))
    X = rng.uniform(0.05, 0.95, npts)
    E = rng.uniform(0.3, 3.0, npts)
    R0 = np.exp(rng.normal(0.0, r0_spread, npts))
    out = np.array([el(x, e, r) for x, e, r in zip(X, E, R0)])
    P = np.log(out[:, :2])
    P = (P - P.min(0)) / np.ptp(P, 0)
    lab = np.stack([X, np.log(out[:, 2])], 1)
    lab = (lab - lab.min(0)) / np.ptp(lab, 0)
    d, idx = cKDTree(P).query(P, k=2)
    return int(((d[:, 1] < 0.01) & (np.linalg.norm(lab[idx[:, 1]] - lab, axis=1) > 0.15)).sum()) / npts

def kernel_orientation_fit(ratio, t0, w, lam=0.3, npts=240):
    r = np.linspace(0.05, 1.0, npts)
    tf = 1.0 + (ratio - 1.0) * (1 - r) ** 2 / np.max((1 - r) ** 2)
    p = np.empty_like(r)
    for i, T in enumerate(tf):
        tt = np.linspace(0.0, T, 3000)
        p[i] = np.trapezoid(np.exp(-0.5 * ((tt - t0) / w) ** 2) * np.exp(-(T - tt) / lam), tt)
    rng = np.max(p) - np.min(p)
    if rng < 1e-12:
        return (float('nan'), float('nan'))
    p = (p - np.min(p)) / rng

    def best(sign):
        out = np.inf
        for d in np.linspace(0.01, 0.6, 60):
            for e in (0.005, 0.02, 0.05, 0.1, 0.2):
                s = 0.5 * (1 + np.tanh((r - (1 - d)) / e))
                out = min(out, float(np.sqrt(np.mean((p - (s if sign > 0 else 1 - s)) ** 2))))
        return out
    return (best(+1), best(-1))

def kernel_max_position(ratio, t0, w, lam=0.3, npts=400):
    r = np.linspace(0.05, 1.0, npts)
    tf = 1.0 + (ratio - 1.0) * (1 - r) ** 2 / np.max((1 - r) ** 2)
    p = np.empty_like(r)
    for i, T in enumerate(tf):
        tt = np.linspace(0.0, T, 3000)
        p[i] = np.trapezoid(np.exp(-0.5 * ((tt - t0) / w) ** 2) * np.exp(-(T - tt) / lam), tt)
    return float(r[int(np.argmax(p))])
MEMORIES = {'single mode 0.3': lambda a: np.exp(-a / 0.3), 'two modes 0.03 & 3': lambda a: 0.5 * np.exp(-a / 0.03) + 0.5 * np.exp(-a / 3.0), 'two modes 0.1 & 10': lambda a: 0.5 * np.exp(-a / 0.1) + 0.5 * np.exp(-a / 10.0), 'stretched exp 0.5': lambda a: np.exp(-np.sqrt(a / 0.3)), 'stretched exp 0.3': lambda a: np.exp(-(a / 0.3) ** 0.3)}

def spectrum_robustness(memory, ratio=20, t0=0.3, w=0.05, npts=300):
    r = np.linspace(0.05, 1.0, npts)
    tf = 1.0 + (ratio - 1.0) * (1 - r) ** 2 / np.max((1 - r) ** 2)
    p = np.empty_like(r)
    for i, T in enumerate(tf):
        tt = np.linspace(0.0, T, 3000)
        p[i] = np.trapezoid(np.exp(-0.5 * ((tt - t0) / w) ** 2) * memory(T - tt), tt)
    p = (p - np.min(p)) / (np.max(p) - np.min(p))
    skin = min((float(np.sqrt(np.mean((p - 0.5 * (1 + np.tanh((r - (1 - d)) / e))) ** 2))) for d in np.linspace(0.01, 0.6, 60) for e in (0.005, 0.02, 0.05, 0.1, 0.2)))
    return (skin, float(r[int(np.argmax(p))]))
PROP4_KERNELS = {'single exponential': lambda a: np.exp(-a / 0.3), 'two modes': lambda a: 0.5 * np.exp(-a / 0.03) + 0.5 * np.exp(-a / 3.0), 'stretched exp 0.3': lambda a: np.exp(-(a / 0.3) ** 0.3), 'linear ramp (not a mixture)': lambda a: np.clip(1 - a / 1.5, 0, None), 'plateau-drop (not a mixture)': lambda a: np.where(a < 0.5, 1.0, np.exp(-(a - 0.5) / 0.2))}

def proposition4_check(npts=300):
    r = np.linspace(0.05, 1.0, npts)
    tf = 1.0 + 19.0 * (1 - r) ** 2 / np.max((1 - r) ** 2)
    before = lambda t: np.where(t < 0.8, 1.0, 0.0)
    ongoing = lambda t: np.exp(-0.5 * ((t - 5.0) / 5.0) ** 2)
    out = {}
    for name, K in PROP4_KERNELS.items():
        pa = np.array([np.trapezoid(before(np.linspace(0, T, 4000)) * K(T - np.linspace(0, T, 4000)), np.linspace(0, T, 4000)) for T in tf])
        pb = np.array([np.trapezoid(ongoing(np.linspace(0, T, 4000)) * K(T - np.linspace(0, T, 4000)), np.linspace(0, T, 4000)) for T in tf])
        out[name] = (bool(np.all(np.diff(pa[::-1]) <= 1e-12)), float(r[np.argmax(pa)]), float(r[np.argmax(pb)]))
    return out

def savage_dickey_bf(upper=(0.0, 1.5), lower=(-0.5, 1.0), delta_true=0.15, sigma=0.08, npts=60, seed=4, nd=1200):
    from scipy.special import logsumexp, log_ndtr
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    r = np.linspace(0.02, 1.0, npts)
    data = (r > 1 - delta_true).astype(float) + rng.normal(0.0, sigma, npts)

    def logmass(m, sd, lo, hi):
        a, b = ((lo - m) / sd, (hi - m) / sd)
        x, y = (log_ndtr(b), log_ndtr(a)) if b <= 0 else (log_ndtr(-a), log_ndtr(-b))
        return x + np.log1p(-np.exp(y - x)) if x > y else -np.inf
    const = -npts * np.log(sigma * np.sqrt(2 * np.pi))
    lw, ld = ([], [])
    for d in np.linspace(0.02, 0.6, nd):
        sh = r > 1 - d
        tl, td, ok = (const, 0.0, True)
        for y, (lo, hi), val in ((data[sh], upper, 1.0), (data[~sh], lower, 0.0)):
            m, S, sd = (y.mean(), np.sum((y - y.mean()) ** 2), sigma / np.sqrt(y.size))
            lm = logmass(m, sd, lo, hi)
            if not np.isfinite(lm):
                ok = False
                break
            tl += -0.5 * S / sigma ** 2 + np.log(sd * np.sqrt(2 * np.pi)) + lm - np.log(hi - lo)
            td += norm.logpdf((val - m) / sd) - np.log(sd) - lm
        if ok:
            lw.append(tl)
            ld.append(td)
    lw, ld = (np.array(lw), np.array(ld))
    return float(np.exp(logsumexp(lw + ld) - logsumexp(lw) + np.log((upper[1] - upper[0]) * (lower[1] - lower[0]))))

def proposition4c_check(T_end=25.0, npts=300):
    r = np.linspace(0.05, 1.0, npts)
    tf = 1.0 + 19.0 * (1 - r) ** 2 / np.max((1 - r) ** 2)
    ongoing = lambda t: np.exp(-0.5 * ((t - 5.0) / 5.0) ** 2)
    out = {}
    for name in ('single exponential', 'two modes', 'linear ramp (not a mixture)'):
        K = PROP4_KERNELS[name]
        dec = np.empty_like(r)
        drawn = np.empty_like(r)
        for i, T in enumerate(tf):
            tt = np.linspace(0, T, 4000)
            pre = np.trapezoid(ongoing(tt) * K(T - tt), tt)
            ss = np.linspace(T, T_end, 4000)
            dec[i] = pre
            drawn[i] = pre + np.trapezoid(ongoing(ss), ss)
        out[name] = (float(r[np.argmax(dec)]), float(r[np.argmax(drawn)]))
    return out

def a5_depth_dependent_clock(n=100, phi_star=0.2, k=5.0):
    from scipy.integrate import solve_ivp
    from scipy.spatial import cKDTree
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
        return (s * t_f, np.trapezoid(1 / lam(cc), tt))
    xs = np.arange(3, n - 3, 3)
    ss = np.linspace(0.3, 3.0, 30)
    G = np.array([[element(i, s) for i in xs] for s in ss])
    Tp, Tr = (np.log(G[..., 0]), np.log(G[..., 1]))
    J = (np.gradient(Tp, x[xs], axis=1) * np.gradient(Tr, ss, axis=0) - np.gradient(Tp, ss, axis=0) * np.gradient(Tr, x[xs], axis=1))[1:-1, 1:-1]
    P = np.stack([Tp.ravel(), Tr.ravel()], 1)
    P = (P - P.min(0)) / np.ptp(P, 0)
    L = np.stack(np.meshgrid(ss, x[xs], indexing='ij'), -1).reshape(-1, 2)
    L = (L - L.min(0)) / np.ptp(L, 0)
    d, idx = cKDTree(P).query(P, k=2)
    coll = int(((d[:, 1] < 0.01) & (np.linalg.norm(L[idx[:, 1]] - L, axis=1) > 0.15)).sum())
    return (int((J > 0).sum()), int((J < 0).sum()), coll)

def proposition3b_clock_check(a_values=(0, 1, 2, 4, 8), lam_fracs=(0.03, 0.1, 0.3, 1.0), r_stops=(0.8, 0.5), n=100):
    from scipy.integrate import solve_ivp
    r = np.linspace(0, 1, n)
    dr = r[1] - r[0]
    rf = 0.5 * (r[1:] + r[:-1])
    D = lambda c: np.exp(9.2 * (np.clip(c, 0, 1) - 0.9))

    def rhs(t, c):
        cf = 0.5 * (c[1:] + c[:-1])
        fl = -D(cf) * (c[1:] - c[:-1]) / dr
        dc = np.zeros_like(c)
        i = np.arange(1, n - 1)
        dc[i] = -(rf[i] * fl[i] - rf[i - 1] * fl[i - 1]) / (r[i] * dr)
        dc[0] = -4 * fl[0] / dr
        dc[-1] = -(5.0 * c[-1] - rf[-1] * fl[-1]) / (dr / 2)
        return dc
    T = np.concatenate([[0.0], np.geomspace(1e-05, 400, 60000)])
    C = solve_ivp(rhs, (0, 400), np.full(n, 0.9), method='BDF', t_eval=T, rtol=1e-07, atol=1e-10).y
    tf = np.array([T[np.argmax(C[i] <= 0.2)] for i in range(n)])
    assert np.all(np.diff(tf) <= 0) and tf.min() > 0

    def profile(a, lam0, t_end):
        M = np.zeros(n)
        out = np.full(n, np.nan)
        dt = np.diff(T)
        for k in range(len(T) - 1):
            live = T[k] < tf
            lam = lam0 * np.exp(a * (0.9 - C[:, k]))
            e = 1.0 if T[k] < t_end else 0.0
            M = np.where(live, M * np.exp(-dt[k] / lam) + e * lam * (1 - np.exp(-dt[k] / lam)), M)
            just = (T[k + 1] >= tf) & np.isnan(out)
            out[just] = M[just]
        assert np.nanmax(out) > 0
        return out
    rows = []
    for rs in r_stops:
        t_end = float(np.interp(rs, r, tf))
        for lf in lam_fracs:
            rows.append((rs, lf, [float(r[np.nanargmax(profile(a, lf * t_end, t_end))]) for a in a_values]))
    return rows

def proposition3b_scaling_check(n=100):
    from scipy.integrate import solve_ivp
    r = np.linspace(0, 1, n)
    dr = r[1] - r[0]
    rf = 0.5 * (r[1:] + r[:-1])
    D = lambda c: np.exp(9.2 * (np.clip(c, 0, 1) - 0.9))

    def rhs(t, c):
        cf = 0.5 * (c[1:] + c[:-1])
        fl = -D(cf) * (c[1:] - c[:-1]) / dr
        dc = np.zeros_like(c)
        i = np.arange(1, n - 1)
        dc[i] = -(rf[i] * fl[i] - rf[i - 1] * fl[i - 1]) / (r[i] * dr)
        dc[0] = -4 * fl[0] / dr
        dc[-1] = -(5.0 * c[-1] - rf[-1] * fl[-1]) / (dr / 2)
        return dc
    T = np.concatenate([[0.0], np.geomspace(1e-06, 400, 70000)])
    C = solve_ivp(rhs, (0, 400), np.full(n, 0.9), method='BDF', t_eval=T, rtol=1e-07, atol=1e-10).y
    tf = np.array([T[np.argmax(C[i] <= 0.2)] for i in range(n)])
    tR = tf[-1]
    t_end = float(np.interp(0.5, r, tf))

    def profile(a, lam0):
        M = np.zeros(n)
        out = np.full(n, np.nan)
        dt = np.diff(T)
        for k in range(len(T) - 1):
            live = T[k] < tf
            lam = lam0 * np.exp(a * (0.9 - C[:, k]))
            e = 1.0 if T[k] < t_end else 0.0
            M = np.where(live, M * np.exp(-dt[k] / lam) + e * lam * (1 - np.exp(-dt[k] / lam)), M)
            just = (T[k + 1] >= tf) & np.isnan(out)
            out[just] = M[just]
        return out
    rows = []
    for lw in (100, 10, 1, 0.1):
        for a in (0, 8):
            p = profile(a, lw * tR)
            interior = r[np.nanargmax(p)] < 0.98
            lam_wet = lw * tR
            lam_arrest = lam_wet * np.exp(a * (0.9 - 0.2))
            rows.append((lw, a, interior, lam_wet > p[-1], lam_arrest > p[-1]))
    return rows


def proposition3b_bound_check(n=100):
    from scipy.integrate import solve_ivp
    r = np.linspace(0, 1, n)
    dr = r[1] - r[0]
    rf = 0.5 * (r[1:] + r[:-1])
    D = lambda c: np.exp(9.2 * (np.clip(c, 0, 1) - 0.9))

    def rhs(t, c):
        cf = 0.5 * (c[1:] + c[:-1])
        fl = -D(cf) * (c[1:] - c[:-1]) / dr
        dc = np.zeros_like(c)
        i = np.arange(1, n - 1)
        dc[i] = -(rf[i] * fl[i] - rf[i - 1] * fl[i - 1]) / (r[i] * dr)
        dc[0] = -4 * fl[0] / dr
        dc[-1] = -(5.0 * c[-1] - rf[-1] * fl[-1]) / (dr / 2)
        return dc
    T = np.concatenate([[0.0], np.geomspace(1e-06, 400, 70000)])
    C = solve_ivp(rhs, (0, 400), np.full(n, 0.9), method='BDF', t_eval=T, rtol=1e-07, atol=1e-10).y
    assert C.max() <= 0.9 + 1e-06
    tf = np.array([T[np.argmax(C[i] <= 0.2)] for i in range(n)])
    tR = tf[-1]

    def profile(a, lam0, t_end):
        M = np.zeros(n)
        out = np.full(n, np.nan)
        dt = np.diff(T)
        for k in range(len(T) - 1):
            live = T[k] < tf
            lam = lam0 * np.exp(a * (0.9 - C[:, k]))
            e = 1.0 if T[k] < t_end else 0.0
            M = np.where(live, M * np.exp(-dt[k] / lam) + e * lam * (1 - np.exp(-dt[k] / lam)), M)
            just = (T[k + 1] >= tf) & np.isnan(out)
            out[just] = M[just]
        return out
    cases = []
    for rs in (0.8, 0.5):
        t_end = float(np.interp(rs, r, tf))
        for lf in (0.03, 0.1, 0.3, 1.0):
            for a in (0, 1, 2, 4, 8):
                cases.append(('clock', rs, t_end, lf * t_end, a))
    t_end = float(np.interp(0.5, r, tf))
    for lw in (100, 10, 1, 0.1):
        for a in (0, 8):
            cases.append(('scaling', 0.5, t_end, lw * tR, a))
    rows = []
    for family, rs, t_end, lam_wet, a in cases:
        p = profile(a, lam_wet, t_end)
        j = int(np.argmin(np.where(tf <= t_end, t_end - tf, np.inf)))
        lower = lam_wet * (1 - np.exp(-tf[j] / lam_wet))
        upper = tR
        rows.append((family, rs, lam_wet / tR, a, float(tf[j]), float(p[j]), float(lower), float(p[-1]), float(upper),
                     bool(p[j] >= lower * (1 - 1e-06)), bool(p[-1] <= upper * (1 + 1e-06)), bool(lower > upper),
                     bool(r[np.nanargmax(p)] < 0.98)))
    return rows
