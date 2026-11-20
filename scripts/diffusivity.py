"""Polystyrene-chloroform mutual diffusivity (Section S5.2) and its ten variants.

Below the glass composition phi_g the diffusivity is the measured glassy one: Crank and Park's sorption values
(second approximation, 25 °C; data/crank_park_1949_table2.csv), interpolated log-linearly, Park's exponential law
(the least-squares slope of ln D on phi) below the lowest point, and a log-linear continuation from the last measured
point to the free-volume value at phi_g. Above phi_g it is the larger of that value and the Vrentas-Duda free-volume self-diffusivity with the
parameters of Zielinski and Duda, converted to mutual diffusion with the Flory-Huggins factor (1 - phi)^2 (1 - 2 chi
phi), at 18.5 °C. Below its Vogel temperature (Tg2 - K22 = 327 K) the polymer contributes no hole free volume.
Solvent fractions are volume fractions; free-volume weights are mass fractions."""
import math
import os

import numpy as np

from common import read

T_JET = 291.65          # K, 18.5 °C, temperature of the arrest-composition data
T_25 = 298.15
RHO_SOLV, RHO_PS = 1.48, 1.05     # g/cm3, chloroform and polystyrene
CHI = 0.085
PHI_G = 0.14            # glass composition on the desorption branch [S12]
TF_HOLD = 0.3           # thermodynamic factor held at its value here in the 'held' variant

# Zielinski and Duda (1992): chloroform from their Table 2, polystyrene from Table 3, toluene/PS from Table 4 (toluene
# check and the K21 - Tg1 of the larger-free-volume variant). xi for chloroform is computed from their Eq. 12.
PS = dict(K12=5.82e-4, K22mTg2=-327.0, V2=0.850, M2j=163.6)
CHCL3 = dict(D0=5.49e-8, K11=6.52e-4, K21mTg1=-37.9, V1=0.510, M1=119.39)
TOLUENE = dict(D0=4.82e-8, K11=1.45e-3, K21mTg1=-86.32, V1=0.917, M1=92.14, xi=0.58)
XI_ZD = CHCL3['V1'] * CHCL3['M1'] / (PS['V2'] * PS['M2j'])          # 0.438
# variant ('viscosity-estimated parameters'): a 26% larger hole free volume of chloroform at 18.5 °C, with K21 - Tg1 taken over from toluene and D0
# rescaled to keep the pure-solvent self-diffusivity at 25 °C
_K11_V = 1.26 * CHCL3['K11'] * (CHCL3['K21mTg1'] + T_JET) / (TOLUENE['K21mTg1'] + T_JET)
_D0_V = CHCL3['D0'] * math.exp(-CHCL3['V1'] / (CHCL3['K11'] * (CHCL3['K21mTg1'] + T_25))
                               + CHCL3['V1'] / (_K11_V * (TOLUENE['K21mTg1'] + T_25)))
CHCL3_VISC = dict(CHCL3, D0=_D0_V, K11=_K11_V, K21mTg1=TOLUENE['K21mTg1'])


def mass_fraction_solvent(phi, rho_s=RHO_SOLV):
    phi = np.asarray(phi, dtype=float)
    return phi * rho_s / (phi * rho_s + (1 - phi) * RHO_PS)


def volume_fraction_solvent(w, rho_s=RHO_SOLV):
    w = np.asarray(w, dtype=float)
    return (w / rho_s) / (w / rho_s + (1 - w) / RHO_PS)


def self_diffusivity_mass(w1, T, solv=CHCL3, xi=None):
    """Vrentas-Duda solvent self-diffusivity (m2/s) at solvent mass fraction w1, E = 0."""
    xi = XI_ZD if xi is None else xi
    w1 = np.asarray(w1, dtype=float)
    w2 = 1 - w1
    vfh = w1 * solv['K11'] * (solv['K21mTg1'] + T) + w2 * PS['K12'] * max(PS['K22mTg2'] + T, 0.0)
    return solv['D0'] * np.exp(-(w1 * solv['V1'] + xi * w2 * PS['V2']) / vfh)


def free_volume_mutual(phi, T=T_JET, solv=CHCL3, xi=None, factor=True, hold=None):
    phi = np.asarray(phi, dtype=float)
    d = self_diffusivity_mass(mass_fraction_solvent(phi), T, solv, xi)
    if not factor:
        return d
    p = phi if hold is None else np.minimum(phi, hold)
    return d * (1 - p) ** 2 * (1 - 2 * CHI * p)


def crank_park():
    r = read('crank_park_1949_table2.csv')
    reg = np.array([float(x['regain_g_per_100g']) for x in r]) / 100
    D = np.array([float(x['D_second_approximation_cm2_per_s_1e-10']) for x in r]) * 1e-14
    return volume_fraction_solvent(mass_fraction_solvent_from_regain(reg)), D


def mass_fraction_solvent_from_regain(reg):
    return reg / (1 + reg)


def park_slope():
    phi, D = crank_park()
    return float(np.polyfit(phi, np.log(D), 1)[0])


VARIANTS = ('baseline', 'whole curve x0.5', 'whole curve x2', 'glass composition 0.13', 'glass composition 0.17',
            'glassy tail to 1e-18', 'viscosity-estimated parameters', 'jump-volume ratio fitted',
            'self-diffusion (no thermodynamic factor)', 'thermodynamic factor held above 0.3')


def xi_fitted():
    """Jump-volume ratio that best reproduces the shape (D / D(pure)) of chloroform self-diffusion in PS solutions at
    25 °C [S16]: least squares in log D with a free constant factor."""
    r = [x for x in read('waggoner_1993_table1.csv') if x['solvent'] == 'chloroform']
    w1 = 1 - np.array([float(x['wt_percent_PS']) for x in r]) / 100
    D = np.array([float(x['D_self_1e-10_m2_per_s']) for x in r]) * 1e-10
    xs = np.linspace(0.3, 0.8, 5001)
    sse = [np.var(np.log(self_diffusivity_mass(w1, T_25, CHCL3, x) / D)) for x in xs]
    return float(xs[int(np.argmin(sse))])


class Diffusivity:
    """Tabulated D(phi) (m2/s) for one variant, at 18.5 °C on the free-volume branch."""

    def __init__(self, variant='baseline', n=40001):
        self.variant = variant
        phi_g, scale, solv, xi, factor, hold = PHI_G, 1.0, CHCL3, None, True, None
        if variant == 'whole curve x0.5':
            scale = 0.5
        elif variant == 'whole curve x2':
            scale = 2.0
        elif variant == 'glass composition 0.13':
            phi_g = 0.13
        elif variant == 'glass composition 0.17':
            phi_g = 0.17
        elif variant == 'viscosity-estimated parameters':
            solv = CHCL3_VISC
        elif variant == 'jump-volume ratio fitted':
            xi = xi_fitted()
        elif variant == 'self-diffusion (no thermodynamic factor)':
            factor = False
        elif variant == 'thermodynamic factor held above 0.3':
            hold = TF_HOLD
        elif variant not in ('baseline', 'glassy tail to 1e-18'):
            raise ValueError(variant)
        self.phi_g = phi_g
        grid = np.linspace(0.0, 1.0, n)
        fv = free_volume_mutual(np.maximum(grid, 1e-9), T_JET, solv, xi, factor, hold)
        cp_phi, cp_D = crank_park()
        a = park_slope()
        lnD = np.empty_like(grid)
        lo = grid < cp_phi[0]
        lnD[lo] = math.log(cp_D[0]) + a * (grid[lo] - cp_phi[0])
        mid = (grid >= cp_phi[0]) & (grid <= cp_phi[-1])
        lnD[mid] = np.interp(grid[mid], cp_phi, np.log(cp_D))
        # above the last measured point: log-linear continuation to the free-volume value at phi_g (held at the last
        # measured value if that is lower, since a glass cannot diffuse more slowly than at a drier composition),
        # and never below the free-volume curve, which a glass with excess free volume cannot fall beneath
        ln_last = math.log(cp_D[-1])
        ln_g = max(math.log(float(free_volume_mutual(phi_g, T_JET, solv, xi, factor, hold))), ln_last)
        above = grid > cp_phi[-1]
        line = np.where(grid[above] < phi_g,
                        ln_last + (ln_g - ln_last) * (grid[above] - cp_phi[-1]) / (phi_g - cp_phi[-1]), ln_g)
        lnD[above] = np.maximum(np.log(np.maximum(fv[above], 1e-300)), line)
        if variant == 'glassy tail to 1e-18':
            tail = grid < 0.05
            d05 = float(np.interp(0.05, grid, lnD))
            lnD[tail] = math.log(1e-18) + (d05 - math.log(1e-18)) * grid[tail] / 0.05
        lnD += math.log(scale)
        self.grid, self.lnD, self.xi, self.solv = grid, lnD, (XI_ZD if xi is None else xi), solv
        self.dx = grid[1] - grid[0]

    def __call__(self, phi):
        x = np.clip(np.asarray(phi, dtype=float), 0.0, 1.0) / self.dx
        i = np.minimum(x.astype(int), len(self.grid) - 2)
        f = x - i
        return np.exp(self.lnD[i] * (1 - f) + self.lnD[i + 1] * f)


def eq9_diameter_nm(D, t_flight, threshold=0.1):
    """Eq. (9) solved for the diameter at which the clock span equals the threshold."""
    return 2e9 * math.sqrt(threshold * D * t_flight)
