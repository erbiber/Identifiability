"""Finite extensibility and a bounded orientation measure (Section S2.5): FENE-P dumbbells in uniaxial extension,
segmental <P2> from the Kuhn-Grün relation, frozen at the arrest times of the diffusive arrest front of Fig. 3.
Part (called by scripts/run_results.py): finite."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from common import write

LAM = 0.3                         # relaxation time, in surface-arrest times
R = np.linspace(0.0, 1.0, 201)    # radial positions r/R
TF = 1.0 + 19.0 * (1 - R) ** 2    # arrest time: the surface at 1, the axis 20 times later (Fig. 3)
SHAPES = {'continues after surface arrest': lambda t: np.exp(-0.5 * ((t - 5.0) / 5.0) ** 2),
          'ends before surface arrest': lambda t: np.where(t < 0.8, 1.0, 0.0)}


def langevin(y):
    """coth(y) - 1/y, with its series below y = 1e-2 to avoid cancellation."""
    return y / 3.0 - y ** 3 / 45.0 + 2 * y ** 5 / 945.0 if y < 1e-2 else 1.0 / np.tanh(y) - 1.0 / y


def inv_langevin(x):
    """Inverse Langevin function, by bracketing (x in [0, 1))."""
    x = float(x)
    if x < 1e-6:
        return 3.0 * x
    return brentq(lambda y: langevin(y) - x, 1e-12, 1.0 / (1.0 - x) + 10.0, xtol=1e-15, rtol=1e-14)


def p2_kuhn_grun(a, c, b):
    """<P2> = 1 - 3x / L^-1(x), x^2 = (a - c)/b; 3x^2/5 at small extension, bounded by 1."""
    x = np.sqrt(np.clip((np.asarray(a) - np.asarray(c)) / b, 0.0, 1.0 - 1e-12))
    return np.array([1.0 - 3.0 * xi / inv_langevin(xi) if xi > 1e-6 else 0.6 * xi * xi for xi in np.atleast_1d(x)])


def fene_p(rate, b, t_end, t_eval=None):
    """A = diag(a, c, c): da/dt = 2 rate a - (f a - 1)/lam, dc/dt = -rate c - (f c - 1)/lam, f = b/(b - a - 2c)."""
    def rhs(t, y):
        a, c = y
        f = b / (b - a - 2 * c)
        e = rate(t)
        return [2 * e * a - (f * a - 1) / LAM, -e * c - (f * c - 1) / LAM]
    sol = solve_ivp(rhs, (0, t_end), [1.0, 1.0], method='LSODA', t_eval=t_eval, rtol=1e-9, atol=1e-11,
                    max_step=0.01)
    return sol


def profile(shape, wi, b):
    """<P2>(r) of elements frozen at their arrest times; stretch rate wi/lam times the history shape."""
    sol = fene_p(lambda t: wi / LAM * float(shape(t)), b, TF.max(), t_eval=np.sort(np.unique(TF)))
    a = np.interp(TF, sol.t, sol.y[0])
    c = np.interp(TF, sol.t, sol.y[1])
    return p2_kuhn_grun(a, c, b)


def area_mean(p):
    return float(np.trapezoid(p * R, R) / np.trapezoid(R, R))


def wi_for_mean(shape, b, target):
    return brentq(lambda w: area_mean(profile(shape, w, b)) - target, 0.05, 200.0, xtol=1e-6)


def part_finite():
    rows = []
    for b in (20.0, 100.0, 1000.0):
        for name, shape in SHAPES.items():
            for wi in (0.4, 1.0, 2.0, 5.0, 10.0, 20.0):
                p = profile(shape, wi, b)
                k = int(np.argmax(p))
                rows.append((b, name, wi, area_mean(p), R[k], p[k] - p[-1]))
    write('finite_extensibility_profiles.csv', ['b', 'stretching', 'peak_Wi', 'mean_P2', 'max_position_r_over_R',
                                                'interior_excess'], rows)
    shape = SHAPES['continues after surface arrest']
    rows = []
    for b in (20.0, 100.0, 1000.0):
        for target in (0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.91, 0.92, 0.95):
            try:
                wi = wi_for_mean(shape, b, target)
            except ValueError:
                continue
            p = profile(shape, wi, b)
            k = int(np.argmax(p))
            rows.append((b, target, wi, R[k], p[k] - p[-1]))
    write('finite_extensibility_contrast.csv', ['b', 'mean_P2', 'peak_Wi', 'max_position_r_over_R',
                                                'interior_excess'], rows)
    # saturation: constant-rate stretch of Hencky strain 5.4, Weissenberg number doubled
    rows = []
    for b in (20.0, 100.0, 1000.0):
        def p_end(wi):
            sol = fene_p(lambda t: wi / LAM, b, 5.4 * LAM / wi)
            return float(p2_kuhn_grun(sol.y[0][-1], sol.y[1][-1], b)[0])
        for target in (0.25, 0.5, 0.75, 0.9, 0.95, 0.97, 0.98):
            try:
                wi = brentq(lambda w: p_end(w) - target, 0.2, 5000.0, xtol=1e-8)
            except ValueError:
                continue
            rows.append((b, target, wi, p_end(2 * wi) - target))
    write('finite_extensibility_saturation.csv', ['b', 'P2', 'Wi', 'rise_on_doubling_Wi'], rows)
