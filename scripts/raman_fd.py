"""Finite-difference check of the confocal weight with the BaF2 window under the fiber (Section S8.4).

Two-dimensional frequency-domain problem in the plane perpendicular to the fiber axis, scattered-field form: the
background is the focused beam (the plane waves of scripts/raman_weighting.py) reflected and transmitted at the flat
air-BaF2 interface by the Fresnel coefficients of each plane wave; the fiber (index 1.587) touches the interface; a
stretched-coordinate perfectly matched layer surrounds the domain. TM: E along the axis, d2E + k^2 eps E = 0.
TE: H along the axis, div(eps^-1 grad H) + k^2 H = 0, with E from the curl of H. Without the substrate the solver
reproduces the series solution (results/raman_substrate.csv, rows 'no substrate')."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

import raman_weighting as rw

N_SUB = 1.475             # BaF2


class Grid:
    def __init__(self, d, h=6e-9, margin=0.45e-6, pml=0.35e-6, below=0.45e-6):
        self.a = d / 2
        self.h = h
        self.pml = pml
        x0, x1 = -(self.a + margin + pml), self.a + margin + pml
        y0, y1 = -(below + pml), d + margin + pml
        self.x = np.arange(x0 + h / 2, x1, h)
        self.y = np.arange(y0 + h / 2, y1, h)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        self.bounds = (x0, x1, y0, y1)

    def fiber_fraction(self, sub=4):
        """Area fraction of each cell inside the fiber (centred at (0, a)), by sub-sampling."""
        h = self.h
        off = (np.arange(sub) + 0.5) / sub - 0.5
        frac = np.zeros_like(self.X)
        for ox in off:
            for oy in off:
                frac += (self.X + ox * h) ** 2 + (self.Y + oy * h - self.a) ** 2 <= self.a ** 2
        return frac / sub ** 2

    def stretch(self, k, which):
        """PML stretch factors s = 1 + i sigma/k at cell centres and faces along x or y."""
        x0, x1, y0, y1 = self.bounds
        c = self.x if which == 'x' else self.y
        lo, hi = (x0, x1) if which == 'x' else (y0, y1)
        def s(z):
            dpen = np.maximum(np.maximum(lo + self.pml - z, z - (hi - self.pml)), 0) / self.pml
            return 1 + 1j * 5.0 * dpen ** 3
        faces = np.concatenate([[c[0] - self.h / 2], c + self.h / 2])
        return s(c), s(faces)


def background(G, k, n_sub, pol, na, w=1.0):
    """Focused beam from above (travelling towards -y), focused at the fiber centre, with the flat interface at y = 0.
    Returns the background E_z (TM) or H_z (TE) on the grid."""
    th = np.linspace(-np.arcsin(na), np.arcsin(na), 121) if na > 0 else np.array([0.0])
    amp = np.exp(-(np.sin(th) / (w * na)) ** 2) * np.sqrt(np.cos(th)) if na > 0 else np.ones(1)
    amp = amp / amp.sum()
    X, Y = G.X, G.Y
    F = np.zeros_like(X, dtype=complex)
    eps2 = n_sub ** 2
    for t, A in zip(th, amp):
        kx = k * np.sin(t)
        k1 = k * np.cos(t)
        k2 = k * np.sqrt(eps2 - np.sin(t) ** 2 + 0j)
        if pol == 'TM':
            r, tr = (k1 - k2) / (k1 + k2), 2 * k1 / (k1 + k2)
        else:
            r, tr = (k1 - k2 / eps2) / (k1 + k2 / eps2), 2 * k1 / (k1 + k2 / eps2)
        phase = A * np.exp(1j * k1 * G.a)                    # in phase at (0, a)
        air = np.exp(1j * kx * X) * (np.exp(-1j * k1 * Y) + r * np.exp(1j * k1 * Y))
        sub = tr * np.exp(1j * kx * X) * np.exp(-1j * k2 * Y)
        F += phase * np.where(Y >= 0, air, sub)
    return F


def operator(G, k, coef_cell, coef_face_x, coef_face_y, mass):
    """Discrete d/dx (cx d/dx) + d/dy (cy d/dy) + k^2 mass, with PML stretching, on the cell-centred grid."""
    nx, ny = G.X.shape
    h = G.h
    sx_c, sx_f = G.stretch(k, 'x')
    sy_c, sy_f = G.stretch(k, 'y')
    N = nx * ny
    idx = np.arange(N).reshape(nx, ny)
    rows, cols, vals = [], [], []
    diag = k ** 2 * mass.astype(complex).copy()
    # x direction
    cxp = coef_face_x[1:, :] / (sx_c[:, None] * sx_f[1:, None] * h ** 2)
    cxm = coef_face_x[:-1, :] / (sx_c[:, None] * sx_f[:-1, None] * h ** 2)
    diag -= cxp + cxm
    rows += [idx[:-1, :].ravel(), idx[1:, :].ravel()]
    cols += [idx[1:, :].ravel(), idx[:-1, :].ravel()]
    vals += [cxp[:-1, :].ravel(), cxm[1:, :].ravel()]
    cyp = coef_face_y[:, 1:] / (sy_c[None, :] * sy_f[None, 1:] * h ** 2)
    cym = coef_face_y[:, :-1] / (sy_c[None, :] * sy_f[None, :-1] * h ** 2)
    diag -= cyp + cym
    rows += [idx[:, :-1].ravel(), idx[:, 1:].ravel()]
    cols += [idx[:, 1:].ravel(), idx[:, :-1].ravel()]
    vals += [cyp[:, :-1].ravel(), cym[:, 1:].ravel()]
    rows.append(idx.ravel())
    cols.append(idx.ravel())
    vals.append(diag.ravel())
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))


def faces_avg(c, axis):
    """Arithmetic average of a cell quantity onto the faces along an axis (edge faces copy the edge cell)."""
    c = np.moveaxis(c, axis, 0)
    f = np.concatenate([c[:1], 0.5 * (c[1:] + c[:-1]), c[-1:]], axis=0)
    return np.moveaxis(f, 0, axis)


def total_field(G, wavelength, pol, na, n_sub=N_SUB, m=rw.N_FIBER, w=1.0):
    k = 2 * np.pi / wavelength
    frac = G.fiber_fraction()
    eps_bg = np.where(G.Y >= 0, 1.0, n_sub ** 2)
    eps = eps_bg * (1 - frac) + m ** 2 * frac
    inv = (1 - frac) / eps_bg + frac / m ** 2
    bg = background(G, k, n_sub, pol, na, w)
    if pol == 'TM':
        one = np.ones_like(eps)
        L = operator(G, k, None, faces_avg(one, 0), faces_avg(one, 1), eps)
        L0 = operator(G, k, None, faces_avg(one, 0), faces_avg(one, 1), eps_bg)
    else:
        L = operator(G, k, None, faces_avg(inv, 0), faces_avg(inv, 1), np.ones_like(eps))
        inv0 = 1 / eps_bg
        L0 = operator(G, k, None, faces_avg(inv0, 0), faces_avg(inv0, 1), np.ones_like(eps))
    rhs = -((L - L0) @ bg.ravel())
    sc = spsolve(L.tocsc(), rhs).reshape(G.X.shape)
    F = bg + sc
    if pol == 'TM':
        return frac, np.abs(F) ** 2
    # E = (i/k) eps^-1 curl H: take eps^-1 dH/dn on the faces (where the discrete flux lives, continuous across the
    # fiber surface) and average the two faces of each cell
    ix, iy = faces_avg(inv, 0), faces_avg(inv, 1)
    fx = ix[1:-1, :] * np.diff(F, axis=0) / G.h
    fy = iy[:, 1:-1] * np.diff(F, axis=1) / G.h
    ex = np.zeros_like(F)
    ey = np.zeros_like(F)
    ex[1:-1, :] = 0.5 * (fx[1:, :] + fx[:-1, :])
    ey[:, 1:-1] = 0.5 * (fy[:, 1:] + fy[:, :-1])
    return frac, (np.abs(ex) ** 2 + np.abs(ey) ** 2) / k ** 2


def weight_map(d, na, n_sub=N_SUB, w=1.0, h=6e-9):
    """Confocal weight |E_exc|^2 |E_Stokes|^2 for Z and X polarization on the grid, with the fiber area fraction."""
    G = Grid(d, h=h)
    out = {}
    for pol, lab in (('TM', 'Z'), ('TE', 'X')):
        frac, e0 = total_field(G, rw.LAMBDA_EXC, pol, na, n_sub, w=w)
        _, e1 = total_field(G, rw.LAMBDA_STOKES, pol, na, n_sub, w=w)
        out[lab] = e0 * e1
    rr = np.sqrt(G.X ** 2 + (G.Y - G.a) ** 2)
    return G, frac, rr, out


def shell_share(G, frac, rr, wmap, delta):
    """Share of the readout from the outer shell of thickness delta; cells straddling the shell edge are split by
    sub-sampling the radius."""
    a = G.a
    inside = frac > 0
    sub = 4
    off = (np.arange(sub) + 0.5) / sub - 0.5
    shell = np.zeros_like(frac)
    for ox in off:
        for oy in off:
            r = np.sqrt((G.X + ox * G.h) ** 2 + (G.Y + oy * G.h - a) ** 2)
            shell += (r <= a) & (r > a - delta)
    shell /= sub ** 2
    wt = wmap * inside
    return float(np.sum(wt * shell) / np.sum(wt * frac))
