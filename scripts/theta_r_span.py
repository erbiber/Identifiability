import numpy as np
import math
D_SOLV, D_POLY, C = (2e-09, 1e-16, 5.0)
T_FLIGHT = 0.01
DIAM = (100, 300, 1000, 2000)
PHI_STAR = (0.2, 0.14, 0.1, 0.07)
FLAT = 0.1
THETA_R_STAR = 3.0
_A = -math.log(D_POLY)
_B = math.log(D_POLY) - (1.0 + C) * math.log(D_SOLV)

def D_s(phi):
    return math.exp(-(_A + _B * phi) / (1.0 + C * phi))

def raw_span(d_nm, phi_star, t=T_FLIGHT):
    R = d_nm * 1e-09 / 2.0
    return R * R / (D_s(phi_star) * t)

def span(d_nm, phi_star, t=T_FLIGHT, theta_star=THETA_R_STAR):
    return min(theta_star, raw_span(d_nm, phi_star, t))

def saturated(d_nm, phi_star, t=T_FLIGHT, theta_star=THETA_R_STAR):
    return raw_span(d_nm, phi_star, t) > theta_star

def gordon_taylor(k, Tg1=373.0, Tg=295.0, Tg2=100.0, rho1=1.05, rho2=1.49):
    w2 = (Tg1 - Tg) / (Tg1 - Tg + k * (Tg - Tg2))
    v2 = w2 / rho2 / (w2 / rho2 + (1.0 - w2) / rho1)
    return (w2, v2, k * (Tg1 - Tg2) / 100.0)

def flat_vs_flight(phis=(0.07, 0.1, 0.14, 0.2), flights=(0.001, 0.01, 0.03, 0.1)):
    import math as _m
    return {(tf, p): 2000000000.0 * _m.sqrt(FLAT * D_s(p) * tf) for tf in flights for p in phis}

def transfer_sensitivity(phi_star=0.1, t=T_FLIGHT):
    import math as _m
    out = []
    for lab, (dsol, dpol, c) in (('baseline', (D_SOLV, D_POLY, C)), ('dilute 2.4e-9', (2.4e-09, D_POLY, C)), ('dilute halved', (1e-09, D_POLY, C)), ('dry x10', (D_SOLV, 1e-15, C)), ('dry /10', (D_SOLV, 1e-17, C)), ('xi = 3', (D_SOLV, D_POLY, 3.0)), ('xi = 7', (D_SOLV, D_POLY, 7.0))):
        a = -_m.log(dpol)
        b = _m.log(dpol) - (1.0 + c) * _m.log(dsol)
        ds = _m.exp(-(a + b * phi_star) / (1.0 + c * phi_star))
        out.append((lab, ds, 2000000000.0 * _m.sqrt(FLAT * ds * t)))
    return out

def flat_diameter(phi_star, t=T_FLIGHT):
    return 2000000000.0 * math.sqrt(FLAT * D_s(phi_star) * t)
TAU_ALPHA_TG = 100.0
N_E_SQUARED = 400.0

def a3_upper_diameter_nm(tau_channel, phi_star, skin_fraction=0.25, tol=0.1):
    core = 1.0 / 2.405 ** 2
    return 2000000000.0 * math.sqrt(tol * tau_channel * D_s(phi_star * skin_fraction) / core)
PHI_STAR_MEASURED = (0.14, 0.17, 0.2)

def flory_solvent_fraction(activity, chi):
    lo, hi = (0.0001, 0.999)
    f = lambda s: math.log(s) + (1 - s) + chi * (1 - s) ** 2 - math.log(activity)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

def joint_worst_flat_nm(phi_star):

    def ds(p, dsol, dpol, c):
        a = -math.log(dpol)
        b = math.log(dpol) - (1 + c) * math.log(dsol)
        return math.exp(-(a + b * p) / (1 + c * p))
    variants = [(2e-09, 1e-16, 5.0), (2.4e-09, 1e-16, 5.0), (1e-09, 1e-16, 5.0), (2e-09, 1e-15, 5.0), (2e-09, 1e-17, 5.0), (2e-09, 1e-16, 3.0), (2e-09, 1e-16, 7.0)]
    return max((2000000000.0 * math.sqrt(FLAT * ds(phi_star, *v) * tf) for v in variants for tf in (0.001, 0.01, 0.03, 0.1)))

def lemma1_drying_check(c_init=None, n=120, phi_star=0.2, k=5.0, tmax=400.0):
    import numpy as np
    from scipy.integrate import solve_ivp
    r = np.linspace(0, 1, n)
    dr = r[1] - r[0]
    rf = 0.5 * (r[1:] + r[:-1])
    D = lambda c: np.exp(9.2 * (np.clip(c, 0, 1) - 0.9))

    def rhs(t, c):
        cf = 0.5 * (c[1:] + c[:-1])
        flux = -D(cf) * (c[1:] - c[:-1]) / dr
        dc = np.zeros_like(c)
        i = np.arange(1, n - 1)
        dc[i] = -(rf[i] * flux[i] - rf[i - 1] * flux[i - 1]) / (r[i] * dr)
        dc[0] = -4 * flux[0] / dr
        dc[-1] = -(k * c[-1] - rf[-1] * flux[-1]) / (dr / 2)
        return dc
    c0 = np.full(n, 0.9) if c_init is None else c_init(r)
    T = np.geomspace(1e-05, tmax, 20000)
    C = solve_ivp(rhs, (0, tmax), c0, method='BDF', t_eval=T, rtol=1e-08, atol=1e-11).y
    tf = np.full(n, np.nan)
    for i in range(n):
        j = int(np.argmax(C[i] <= phi_star))
        if C[i, j] <= phi_star and j > 0:
            tf[i] = T[j - 1] + (C[i, j - 1] - phi_star) / (C[i, j - 1] - C[i, j]) * (T[j] - T[j - 1])
    lam = lambda c: np.exp(8 * (0.9 - c))
    theta = np.array([np.trapezoid(1 / lam(C[i, T <= tf[i]]), T[T <= tf[i]]) for i in range(n)])
    return (float(np.diff(C, axis=0)[1:].max()), float(np.diff(C, axis=1).max()), int((np.diff(tf) >= 0).sum()), int((np.diff(theta) >= 0).sum()))

def arrest_order_with_rich_centre(A, k, n=120, phi_star=0.2, tmax=2000.0):
    import numpy as np
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
        dc[-1] = -(k * c[-1] - rf[-1] * fl[-1]) / (dr / 2)
        return dc
    c0 = 0.9 - A * np.exp(-(r / 0.35) ** 2)
    T = np.geomspace(1e-06, tmax, 20000)
    C = solve_ivp(rhs, (0, tmax), c0, method='BDF', t_eval=T, rtol=1e-08, atol=1e-11).y
    tf = np.full(n, np.nan)
    for i in range(n):
        j = int(np.argmax(C[i] <= phi_star))
        if C[i, j] <= phi_star:
            tf[i] = 0.0 if j == 0 else T[j - 1] + (C[i, j - 1] - phi_star) / (C[i, j - 1] - C[i, j]) * (T[j] - T[j - 1])
    d = np.diff(tf)
    return 'outward-first' if np.all(d < 0) else 'inward-first' if np.all(d > 0) else 'NON-MONOTONE'

def lemma1_time_dependent_law(kfun, Rfun=lambda t: 1.0, n=120, phi_star=0.2, tmax=400.0):
    import numpy as np
    from scipy.integrate import solve_ivp
    r = np.linspace(0, 1, n)
    dr = r[1] - r[0]
    rf = 0.5 * (r[1:] + r[:-1])
    D = lambda c: np.exp(9.2 * (np.clip(c, 0, 1) - 0.9))

    def rhs(t, c):
        R = Rfun(t)
        cf = 0.5 * (c[1:] + c[:-1])
        fl = -D(cf) * (c[1:] - c[:-1]) / dr / R ** 2
        dc = np.zeros_like(c)
        i = np.arange(1, n - 1)
        dc[i] = -(rf[i] * fl[i] - rf[i - 1] * fl[i - 1]) / (r[i] * dr)
        dc[0] = -4 * fl[0] / dr
        dc[-1] = -(kfun(t) * c[-1] / R - rf[-1] * fl[-1]) / (dr / 2)
        return dc
    T = np.geomspace(1e-05, tmax, 20000)
    C = solve_ivp(rhs, (0, tmax), np.full(n, 0.9), method='BDF', t_eval=T, rtol=1e-08, atol=1e-11, max_step=0.5).y
    tf = np.full(n, np.nan)
    for i in range(n):
        j = int(np.argmax(C[i] <= phi_star))
        if C[i, j] <= phi_star and j > 0:
            tf[i] = T[j - 1] + (C[i, j - 1] - phi_star) / (C[i, j - 1] - C[i, j]) * (T[j] - T[j - 1])
    lam = lambda c: np.exp(8 * (0.9 - c))
    th = np.array([np.trapezoid(1 / lam(C[i, T <= tf[i]]), T[T <= tf[i]]) for i in range(n)])
    rewet = sum((int(np.any(C[i, T > tf[i]] > phi_star)) for i in range(n) if np.isfinite(tf[i])))
    return (float(np.diff(C, axis=1).max()), int((np.diff(tf) >= 0).sum()), int((np.diff(th) >= 0).sum()), rewet)

def case_study_min_flight_ms(phi_star, d_min_nm):
    flat10 = 2000000000.0 * math.sqrt(FLAT * D_s(phi_star) * 0.01)
    return 10.0 * (d_min_nm / flat10) ** 2


def lam_alpha(c, phi_eq, k_per_vol):
    dT = k_per_vol * 100.0 * (c - phi_eq)
    return TAU_ALPHA_TG * 10.0 ** (-17.44 * dT / (51.6 + dT))


def phi_star_mode(phi_eq, k_per_vol, t_flight, factor):
    L = math.log10(TAU_ALPHA_TG * factor / t_flight)
    dT = L * 51.6 / (17.44 - L)
    return phi_eq + dT / (k_per_vol * 100.0)


def make_D(dsol=D_SOLV, dpol=D_POLY, c=C):
    a = -math.log(dpol)
    b = math.log(dpol) - (1.0 + c) * math.log(dsol)
    return lambda phi: np.exp(-(a + b * phi) / (1.0 + c * phi))


def drying_lag_spans(R, t_flight, Dfun, combos, n=60, c0=0.9):
    from scipy.integrate import solve_ivp
    r = np.linspace(0, R, n)
    dr = r[1] - r[0]
    rf = 0.5 * (r[1:] + r[:-1])
    k = R * math.log(c0 / 0.2) / (2.0 * t_flight)

    def rhs(t, c):
        cf = 0.5 * (c[1:] + c[:-1])
        fl = -Dfun(np.clip(cf, 0, 1)) * (c[1:] - c[:-1]) / dr
        dc = np.zeros_like(c)
        i = np.arange(1, n - 1)
        dc[i] = -(rf[i] * fl[i] - rf[i - 1] * fl[i - 1]) / (r[i] * dr)
        dc[0] = -4 * fl[0] / dr
        dc[-1] = -(r[-1] * k * c[-1] - rf[-1] * fl[-1]) / (r[-1] * dr / 2)
        return dc
    phi_min = min(p for p, lam in combos)
    ev = lambda t, c: c[0] - (phi_min - 0.0001)
    ev.terminal = True
    sol = solve_ivp(rhs, (0, 1e5), np.full(n, c0), method='BDF', dense_output=True, events=ev, rtol=1e-06, atol=1e-09)
    T = np.concatenate([[0.0], np.geomspace(1e-09, sol.t[-1], 6000)])
    Cs = sol.sol(T)
    out = []
    for phi_star, lam in combos:
        hit_s = np.argmax(Cs[-1] <= phi_star) if np.any(Cs[-1] <= phi_star) else None
        hit_c = np.argmax(Cs[0] <= phi_star) if np.any(Cs[0] <= phi_star) else None
        if hit_s is None or hit_c is None:
            out.append(np.nan)
            continue
        seg = slice(hit_s, hit_c + 1)
        out.append(float(np.trapezoid(1.0 / lam(Cs[0][seg]), T[seg])))
    return out


VARIANTS = (('baseline', (D_SOLV, D_POLY, C)), ('dilute 2.4e-9', (2.4e-09, D_POLY, C)), ('dilute halved', (1e-09, D_POLY, C)), ('dry x10', (D_SOLV, 1e-15, C)), ('dry /10', (D_SOLV, 1e-17, C)), ('xi = 3', (D_SOLV, D_POLY, 3.0)), ('xi = 7', (D_SOLV, D_POLY, 7.0)))


def flat_diameter_consistent(variants=VARIANTS, flights=(0.01, 0.1, 0.2), n_R=14):
    rows = []
    for lab, (dsol, dpol, cc) in variants:
        D = make_D(dsol, dpol, cc)
        for t in flights:
            combos, keys = [], []
            for mode, fac in (('alpha', 1.0), ('network', N_E_SQUARED)):
                for phi_eq in (0.13, 0.17):
                    for kv in (5.0, 6.0, 7.0):
                        ps = phi_star_mode(phi_eq, kv, t, fac)
                        combos.append((ps, lambda x, f=fac, pe=phi_eq, kk=kv: f * lam_alpha(x, pe, kk)))
                        keys.append((mode, phi_eq, kv, ps))
            Rs = np.geomspace(1e-08, 1e-05, n_R)
            spans = np.array([drying_lag_spans(R, t, D, combos) for R in Rs])
            for j, key in enumerate(keys):
                y = spans[:, j]
                d = float('nan')
                for i in range(1, n_R):
                    if np.isfinite(y[i - 1]) and np.isfinite(y[i]) and y[i - 1] < FLAT <= y[i] and y[i - 1] > 0:
                        f = (math.log(FLAT) - math.log(y[i - 1])) / (math.log(y[i]) - math.log(y[i - 1]))
                        d = 2e9 * math.exp(math.log(Rs[i - 1]) + f * (math.log(Rs[i]) - math.log(Rs[i - 1])))
                        break
                    if i == 1 and np.isfinite(y[0]) and y[0] >= FLAT:
                        d = 2e9 * Rs[0]
                        break
                rows.append((lab, t * 1000.0, key[0], key[1], key[2], key[3], d))
    return rows


def case_study_thresholds_consistent(floors=(140.0, 500.0), flights=None, variants=VARIANTS):
    if flights is None:
        flights = tuple(np.geomspace(0.001, 0.5, 12))
    rows = flat_diameter_consistent(variants=variants, flights=flights)
    out = []
    for lab, _ in variants:
        for phi_eq in (0.13, 0.17):
            for kv in (5.0, 6.0, 7.0):
                sel = sorted([(r[1], r[6]) for r in rows if r[0] == lab and r[2] == 'network' and r[3] == phi_eq and r[4] == kv])
                for floor in floors:
                    tmin = float('nan')
                    for (t0, d0), (t1, d1) in zip(sel[:-1], sel[1:]):
                        if np.isfinite(d0) and np.isfinite(d1) and d0 < floor <= d1:
                            f = (math.log(floor) - math.log(d0)) / (math.log(d1) - math.log(d0))
                            tmin = math.exp(math.log(t0) + f * (math.log(t1) - math.log(t0)))
                            break
                    if sel and np.isfinite(sel[0][1]) and sel[0][1] >= floor:
                        tmin = sel[0][0]
                    out.append((lab, phi_eq, kv, floor, tmin))
    return out


def a3_sensitivity(phis=(0.14, 0.2), skins=(0.25, 0.5), taus=(10.0, 100.0, 1000.0), tol=0.1):
    core = 1.0 / 2.405 ** 2
    rows = []
    for lab, (dsol, dpol, cc) in VARIANTS:
        D = make_D(dsol, dpol, cc)
        for p in phis:
            for sk in skins:
                for tau in taus:
                    d_um = 2e6 * math.sqrt(tol * tau * float(D(p * sk)) / core)
                    rows.append((p, sk, lab, tau, d_um))
    return rows
