"""Radial drying of a polystyrene-chloroform fiber (Sections 4.1-4.2, S4.3, S5.3, S8.3, S8.5).

Fickian diffusion in a cylinder of fixed radius R with the diffusivity of scripts/diffusivity.py and a surface flux
k c proportional to the surface solvent fraction. The reference evaporation rate k = R ln(c0/0.2)/(2 t_flight) would
take a well-mixed jet from c0 = 0.9 to 0.2 in the flight time. In units x = r/R and tau = t/t_flight the solution
depends on R and t_flight only through G = t_flight/R^2, so one solution serves every flight time.

Relaxation of a structural mode: lambda = f tau_alpha(T_g) 10^(-C1 dT/(C2 + dT)), dT = s 100 (c - phi_eq) above the
solvent-induced glass transition (WLF), and a non-equilibrium glass law below it, in which log lambda keeps rising with
a fraction x of the WLF slope at T_g. The mode arrests (in the sense of Section 4.1) at the composition phi* where
lambda = t_flight. Modes: alpha (f = 1), network (f = Ne^2), chain retraction (Z^2 Ne^2) and reptation (3 Z^3 Ne^2)."""
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags

import diffusivity as dfm

C0 = 0.9                  # initial solvent volume fraction (10-15% w/v PS in chloroform)
TAU_ALPHA_TG = 100.0      # s, alpha relaxation time at the glass transition
WLF = (17.44, 51.6)       # universal WLF constants
WLF_PS = (13.7, 50.0)     # polystyrene-specific constants (check)
X_GLASS = 0.3             # non-equilibrium glass: fraction of the WLF slope at T_g kept below it
NE = 20.0                 # Kuhn segments per entanglement strand (18-24)
Z = 900.0 / 13.3          # entanglements per chain: Mw 900 kg/mol, Me 13.3 kg/mol
MODES = {'alpha': 1.0, 'network': NE ** 2, 'chain retraction': Z ** 2 * NE ** 2, 'reptation': 3 * Z ** 3 * NE ** 2}
PHI_EQ = (0.13, 0.17)
SHIFTS = (5.0, 6.0, 7.0)  # K per volume percent
THRESHOLD = 0.1           # radial clock span defining the uniform-arrest diameter
KEEP = 0.9                # A3: the volume-weighted cross-section keeps at least 90%
AXIS_END = 0.02           # post-arrest integration runs until the axis holds 2% solvent


def log10_lambda(c, phi_eq, shift, factor, wlf=WLF, x_glass=X_GLASS):
    """log10 of the relaxation time (s) of a mode at solvent fraction c."""
    c1, c2 = wlf
    c = np.asarray(c, dtype=float)
    dT = shift * 100.0 * (c - phi_eq)
    above = -c1 * np.maximum(dT, 0) / (c2 + np.maximum(dT, 0))
    if x_glass is None:            # equilibrium WLF continued below T_g, diverging at the Vogel point dT = -C2
        dn = np.minimum(dT, 0)
        below = np.where(c2 + dn > 1e-9, -c1 * dn / np.maximum(c2 + dn, 1e-9), 300.0)
    else:
        below = -x_glass * (c1 / c2) * np.minimum(dT, 0)
    return math.log10(factor * TAU_ALPHA_TG) + np.where(dT >= 0, above, below)


def phi_star(phi_eq, shift, factor, t_flight, wlf=WLF):
    """Composition at which the mode's relaxation time equals the flight time (WLF branch)."""
    c1, c2 = wlf
    L = math.log10(factor * TAU_ALPHA_TG / t_flight)
    dT = L * c2 / (c1 - L)
    return phi_eq + dT / (shift * 100.0)


def surface_grid(n=60, h_surface=2e-4):
    """Cell faces on [0, 1], geometric with the smallest cell at the surface."""
    lo, hi = 1.0 + 1e-9, 2.0
    for _ in range(200):
        q = 0.5 * (lo + hi)
        if h_surface * (q ** n - 1) / (q - 1) > 1.0:
            hi = q
        else:
            lo = q
    h = h_surface * q ** np.arange(n)[::-1]
    faces = np.concatenate([[0.0], np.cumsum(h)])
    faces /= faces[-1]
    return faces


def uniform_grid(n=60):
    return np.linspace(0.0, 1.0, n + 1)


class Drying:
    """Solution c(x, tau) of the dimensionless drying problem for one diffusivity, G = t_flight/R^2 and evaporation
    factor (times the reference rate)."""

    def __init__(self, D, G, evap=1.0, faces=None, n_samples=6000, stop_axis=AXIS_END):
        faces = surface_grid() if faces is None else faces
        self.faces = faces
        xc = 0.5 * (faces[1:] + faces[:-1])
        self.x = xc
        self.area = 0.5 * (faces[1:] ** 2 - faces[:-1] ** 2)          # per radian
        n = len(xc)
        beta = evap * math.log(C0 / 0.2) / 2.0
        dxc = np.diff(xc)
        rf = faces[1:-1]
        A = self.area
        last = faces[-1]

        def rhs(t, c):
            cf = 0.5 * (c[1:] + c[:-1])
            flux = -G * D(cf) * (c[1:] - c[:-1]) / dxc                     # outward flux at internal faces
            out = np.empty(n)
            out[:-1] = rf * flux
            out[-1] = last * beta * max(c[-1], 0.0)
            inn = np.concatenate([[0.0], out[:-1]])
            return -(out - inn) / A

        sp = diags([np.ones(n - 1), np.ones(n), np.ones(n - 1)], [-1, 0, 1]).tocsc()
        ev = lambda t, c: c[0] - stop_axis
        ev.terminal = True
        sol = solve_ivp(rhs, (0.0, 1e12), np.full(n, C0), method='BDF', jac_sparsity=sp, events=ev, rtol=1e-6,
                        atol=1e-9, dense_output=True)
        self.status = sol.status
        self.tau_end = float(sol.t[-1])
        self.tau = np.concatenate([[0.0], np.geomspace(1e-7, self.tau_end, n_samples)])
        self.c = sol.sol(self.tau)                                      # (cells, samples)

    def crossing_times(self, level):
        """First time each cell reaches the level (linear interpolation; nan if never)."""
        c = self.c
        below = c <= level
        hit = below.any(axis=1)
        j = np.argmax(below, axis=1)
        j = np.maximum(j, 1)
        rows = np.arange(c.shape[0])
        c0, c1 = c[rows, j - 1], c[rows, j]
        t0, t1 = self.tau[j - 1], self.tau[j]
        f = np.where(c0 != c1, (c0 - level) / np.where(c0 != c1, c0 - c1, 1.0), 0.0)
        return np.where(hit, t0 + f * (t1 - t0), np.nan)

    def clock_rate(self, phi_eq, shift, factor, t_flight, **kw):
        """t_flight / lambda(c) at every sample (dimensionless clock rate)."""
        return t_flight / 10.0 ** log10_lambda(self.c, phi_eq, shift, factor, **kw)

    @staticmethod
    def _cumulative(rate, tau):
        return np.concatenate([np.zeros((rate.shape[0], 1)),
                               np.cumsum(0.5 * (rate[:, 1:] + rate[:, :-1]) * np.diff(tau), axis=1)], axis=1)

    def span_and_keep(self, phi_eq, shift, factor, t_flight, keep=True, **kw):
        """Radial clock span (Eq. S1, axis clock between surface and axis arrest) and the volume-weighted fraction kept
        after arrest under single-mode fading memory."""
        ps = phi_star(phi_eq, shift, factor, t_flight, kw.get('wlf', WLF))
        tf = self.crossing_times(ps)
        if not (np.isfinite(tf[0]) and np.isfinite(tf[-1])):
            return ps, float('nan'), float('nan')
        rate = self.clock_rate(phi_eq, shift, factor, t_flight, **kw)
        axis = self._cumulative(rate[:1], self.tau)[0]
        span = float(np.interp(tf[0], self.tau, axis) - np.interp(tf[-1], self.tau, axis))
        if not keep:
            return ps, span, float('nan')
        cum = self._cumulative(rate, self.tau)
        at_arrest = np.array([np.interp(t, self.tau, row) for t, row in zip(tf, cum)])
        theta_post = cum[:, -1] - at_arrest
        kept = float(np.sum(self.area * np.exp(-theta_post)) / np.sum(self.area))
        return ps, span, kept


# ------------------------------------------------------------------------------------------- scans over G
G_GRID = np.geomspace(1e7, 3e16, 50)          # t_flight / R^2, s/m^2
FLIGHTS = (0.010, 0.040, 0.100, 0.200)        # s
CASE_FLIGHTS = tuple(np.geomspace(0.001, 0.5, 12))
MAIN_MODES = ('alpha', 'network', 'chain retraction', 'reptation')


def scan(variant, evap=1.0, faces=None, relax=None, combos=None, keep=True, g_grid=G_GRID):
    """Solve the drying problem on the G grid and evaluate every combination.
    combos: list of (flight, mode, phi_eq, shift, threshold-irrelevant) keys; relax: kwargs for log10_lambda.
    Returns {key: (phi_star, spans over G, kept over G)}."""
    D = dfm.Diffusivity(variant)
    relax = relax or {}
    if combos is None:
        combos = [(t, m, pe, s) for t in FLIGHTS for m in MAIN_MODES for pe in PHI_EQ for s in SHIFTS]
    out = {k: (None, np.full(len(g_grid), np.nan), np.full(len(g_grid), np.nan)) for k in combos}
    for j, G in enumerate(g_grid):
        sol = Drying(D, G, evap=evap, faces=faces)
        for k in combos:
            t, m, pe, s = k
            ps, span, kept = sol.span_and_keep(pe, s, MODES[m], t, keep=keep, **relax)
            out[k] = (ps, out[k][1], out[k][2])
            out[k][1][j] = span
            out[k][2][j] = kept
    return out


def crossing_diameter_nm(t_flight, spans, level, g_grid=G_GRID, increasing_with_d=True):
    """Diameter (nm) at which a quantity tabulated on the G grid crosses the level (log-linear interpolation in
    log d; spans grow with d, the fraction kept falls with d)."""
    d = 2e9 * np.sqrt(t_flight / g_grid)            # decreasing along the grid
    o = np.argsort(d)
    d, y = d[o], np.asarray(spans)[o]
    ok = np.isfinite(y)
    d, y = d[ok], y[ok]
    if not increasing_with_d:
        y = -y
        level = -level
    above = y >= level
    if not above.any() or above.all():
        return float('nan')
    i = int(np.argmax(above))                       # first grid diameter at or above the level
    if i == 0:
        return float('nan')
    if increasing_with_d and y[i - 1] > 0 and y[i] > 0:
        f = (math.log(level) - math.log(y[i - 1])) / (math.log(y[i]) - math.log(y[i - 1]))
    else:
        f = (level - y[i - 1]) / (y[i] - y[i - 1])
    return float(math.exp(math.log(d[i - 1]) + f * (math.log(d[i]) - math.log(d[i - 1]))))


def value_at_diameter(t_flight, ys, d_nm, g_grid=G_GRID):
    d = 2e9 * np.sqrt(t_flight / g_grid)
    o = np.argsort(d)
    return float(np.interp(math.log(d_nm), np.log(d[o]), np.asarray(ys)[o]))
