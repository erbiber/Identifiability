"""Confocal readout weight of a thin skin (Section S8.4): exact internal field of an infinite homogeneous cylinder for
a focused beam built from plane waves perpendicular to the fiber axis.

Excitation: plane waves within +-arcsin(NA) of the optical axis, equally spaced in angle, with amplitude
exp(-(sin(theta)/(w NA))^2) cos(theta)^(1/2) (a Gaussian pupil of width w times the pupil radius and the aplanatic
factor), in phase at the focus. Collection by reciprocity: the field of the same focused
beam at the Stokes wavelength. Local weight = |E_exc|^2 |E_Stokes|^2, for light polarized along the axis (Z, TM) or
across it (X, TE; the in-plane components combined). kappa = (skin share of the weighted cross-section) / (skin area
fraction); kappa = 1 is the volume average."""
import numpy as np
from scipy.special import hankel1, jv, jvp, h1vp

N_FIBER = 1.587           # polystyrene
LAMBDA_EXC = 632.8e-9
LAMBDA_STOKES = 1.0 / (1.0 / LAMBDA_EXC - 620e2)   # 620 cm-1 band: 658.7 nm
SETTINGS = {'NA 0.8, centred (central)': dict(na=0.8), 'NA 0.65': dict(na=0.65), 'NA 0.9': dict(na=0.9),
            'NA 0.5': dict(na=0.5), 'NA 0.8, under-filled pupil': dict(na=0.8, w=0.5),
            'NA 0.8, over-filled pupil': dict(na=0.8, w=2.0), 'NA 0.8, focus offset 0.25 radius': dict(na=0.8, off=0.25),
            'NA 0.8, focus offset 0.5 radius': dict(na=0.8, off=0.5), 'Unfocused plane wave': dict(na=0.0)}


def coefficients(x, m, nmax):
    """Internal-field coefficients c_n (TM, E along the axis) and d_n (TE, H along the axis), n = -nmax..nmax."""
    n = np.arange(-nmax, nmax + 1)
    J, Jp = jv(n, m * x), jvp(n, m * x)
    H, Hp = hankel1(n, x), h1vp(n, x)
    w = 2j / (np.pi * x)
    return n, w / (J * Hp - m * Jp * H), w / (J * Hp - Jp * H / m)


def beam_coefficients(n, na, w=1.0, off=0.0, k=1.0, a=1.0, npw=121):
    """B_n = sum over plane waves of amplitude * exp(-i n theta) * (phase that puts the focus at y = off * a)."""
    if na == 0:
        return np.ones(len(n), dtype=complex)
    th = np.linspace(-np.arcsin(na), np.arcsin(na), npw)
    amp = np.exp(-(np.sin(th) / (w * na)) ** 2) * np.sqrt(np.cos(th))     # Gaussian pupil, aplanatic factor
    amp /= amp.sum()
    ph = np.exp(-1j * k * off * a * np.sin(th))
    return (amp * ph) @ np.exp(-1j * np.outer(th, n))


def internal_fields(d, wavelength, m=N_FIBER, na=0.8, w=1.0, off=0.0, nr=240, nphi=256):
    """|E|^2 of the Z (TM) and X (TE) internal fields on a polar grid (r from 0 to a, cell centres)."""
    a = d / 2
    k = 2 * np.pi / wavelength
    x = k * a
    nmax = int(np.ceil(m * x + 4 * (m * x) ** (1 / 3) + 12))
    n, c, dd = coefficients(x, m, nmax)
    B = beam_coefficients(n, na, w, off, k, a)
    r = (np.arange(nr) + 0.5) / nr * a
    phi = np.arange(nphi) / nphi * 2 * np.pi
    kr = m * k * r[:, None]
    J = jv(n[None, :], kr)
    Jp = jvp(n[None, :], kr)
    e = np.exp(1j * np.outer(n, phi))                                   # (n, phi)
    ez = (J * (1j ** n * c * B)) @ e
    er = -((J * (1j ** n * n * dd * B)) / (m ** 2 * k * r[:, None])) @ e
    ep = -1j / m * ((Jp * (1j ** n * dd * B)) @ e)
    return r, phi, np.abs(ez) ** 2, np.abs(er) ** 2 + np.abs(ep) ** 2


def weights(d, m=N_FIBER, **beam):
    """Radial profiles of the confocal weight (integrated over angle), Z and X, and the radii."""
    r, phi, z0, x0 = internal_fields(d, LAMBDA_EXC, m, **beam)
    _, _, z1, x1 = internal_fields(d, LAMBDA_STOKES, m, **beam)
    return r, (z0 * z1).mean(axis=1), (x0 * x1).mean(axis=1)


def skin_share(r, wr, a, delta):
    """Share of the area-weighted readout from r > a - delta (cell-centre rule with a fractional boundary cell)."""
    dr = r[1] - r[0]
    lo = r - dr / 2
    hi = r + dr / 2
    frac = np.clip((hi - (a - delta)) / dr, 0, 1)
    dA = 0.5 * (hi ** 2 - lo ** 2)
    return float(np.sum(wr * dA * frac) / np.sum(wr * dA))


def area_fraction(a, delta):
    return 1 - max(1 - delta / a, 0) ** 2


def kappa(d, delta, m=N_FIBER, **beam):
    r, wz, wx = weights(d, m, **beam)
    f = area_fraction(d / 2, delta)
    return skin_share(r, wz, d / 2, delta) / f, skin_share(r, wx, d / 2, delta) / f


def small_cylinder_check(m=N_FIBER):
    """Plane wave on a cylinder much thinner than the wavelength: internal |E| -> 1 (Z) and 2/(1+m^2) (X)."""
    r, phi, z, x = internal_fields(1e-9, LAMBDA_EXC, m, na=0.0, nr=20, nphi=32)
    return float(np.sqrt(z.mean())), float(np.sqrt(x.mean())), 2 / (1 + m ** 2)


def surface_continuity(d=800e-9, m=N_FIBER):
    """Largest relative mismatch of the tangential fields (E_z for TM; H_z for TE) across the surface, plane wave."""
    k = 2 * np.pi / LAMBDA_EXC
    a = d / 2
    x = k * a
    nmax = int(np.ceil(m * x + 4 * (m * x) ** (1 / 3) + 12))
    n = np.arange(-nmax, nmax + 1)
    _, c, dd = coefficients(x, m, nmax)
    an = (c * jv(n, m * x) - jv(n, x)) / hankel1(n, x)                 # scattered coefficients from E_z continuity
    lhs = m * c * jvp(n, m * x)                                         # inside dE_z/dr / k
    rhs = jvp(n, x) + an * h1vp(n, x)
    bn = (dd * jv(n, m * x) - jv(n, x)) / hankel1(n, x)
    lhs2 = dd * jvp(n, m * x) / m
    rhs2 = jvp(n, x) + bn * h1vp(n, x)
    return float(np.max(np.abs(lhs - rhs)) / np.max(np.abs(rhs))), float(np.max(np.abs(lhs2 - rhs2)) / np.max(np.abs(rhs2)))


def tilted(d, alpha, m=N_FIBER, **beam):
    """Beam tilted by alpha along the fiber axis, neglecting polarization coupling: the normal-incidence problem for a
    fiber of diameter d cos(alpha) and relative index sqrt(m^2 - sin^2 alpha)/cos(alpha); the shell keeps its
    fraction of the cross-section."""
    ca = np.cos(alpha)
    return d * ca, np.sqrt(m ** 2 - np.sin(alpha) ** 2) / ca
