"""Dimensionless drying problem of Lemma 1 (Section S3.1, Fig. S2) and the Flory relation for the arrest composition
(Section S4.1, Fig. S3). Unchanged from the October 2026 release."""
import math

import numpy as np


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
