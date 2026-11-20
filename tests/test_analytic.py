"""Analytic limits and internal consistency of the model code. Run from the repository root:
    python tests/test_analytic.py        (or python -m pytest tests/test_analytic.py)"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diffusivity as dfm      # noqa: E402
import finite                  # noqa: E402
import ps_drying as pd         # noqa: E402
import raman_fd as fd          # noqa: E402
import raman_weighting as rw   # noqa: E402
import records                 # noqa: E402
import separability as sep     # noqa: E402
from common import shell_fraction, thinnest_shell   # noqa: E402


def test_shell_fraction_inverse():
    for d in (300.0, 550.0, 1700.0):
        for delta in (5.0, 13.1, 30.0, 60.0, 140.0):
            assert abs(thinnest_shell(d, shell_fraction(d, delta)) - delta) < 1e-9
    assert shell_fraction(50.0, 30.0) == 1.0


def test_proposition3_cases():
    out = sep.proposition4_check()
    for name, (monotone, pos_a, pos_b) in out.items():
        assert monotone and pos_a == 1.0                  # (a) surface maximum for every kernel
        assert pos_b < 0.95                               # (b) interior maximum
    for name, (dec, drawn) in sep.proposition4c_check().items():
        assert drawn == 1.0                               # (c) drawn skin: surface maximum


def test_cylinder_field_limits():
    z, x, x_exp = rw.small_cylinder_check()
    assert abs(z - 1) < 1e-3 and abs(x - x_exp) < 1e-3     # electrostatic limit 1 and 2/(1+m^2)
    ctm, cte = rw.surface_continuity()
    assert ctm < 1e-12 and cte < 1e-12                     # tangential fields continuous at the surface


def test_fd_against_series():
    d = 500e-9
    G, frac, rr, W = fd.weight_map(d, 0.8, n_sub=1.0)
    f = rw.area_fraction(d / 2, 30e-9)
    kz, kx = fd.shell_share(G, frac, rr, W['Z'], 30e-9) / f, fd.shell_share(G, frac, rr, W['X'], 30e-9) / f
    sz, sx = rw.kappa(d, 30e-9, na=0.8)
    assert abs(kz / sz - 1) < 0.07 and abs(kx / sx - 1) < 0.07


def test_kuhn_grun_small_extension():
    for a in (1.001, 1.01):
        x2 = (a - 1.0) / 100.0
        assert abs(finite.p2_kuhn_grun(a, 1.0, 100.0)[0] / (0.6 * x2) - 1) < 0.01


def test_free_volume_pure_solvent():
    assert abs(float(dfm.self_diffusivity_mass(1.0, dfm.T_25)) - 2.7e-9) < 0.05e-9
    assert abs(dfm.XI_ZD - 0.438) < 5e-4


def test_eq9_scaling():
    D = dfm.Diffusivity('baseline')
    a, b = dfm.eq9_diameter_nm(float(D(0.17)), 0.01), dfm.eq9_diameter_nm(float(D(0.17)), 0.1)
    assert abs(b / a - math.sqrt(10)) < 1e-12


def test_arrest_composition_wlf():
    for m in pd.MODES.values():
        ps = pd.phi_star(0.15, 6.0, m, 0.1)
        assert abs(10 ** pd.log10_lambda(ps, 0.15, 6.0, m) - 0.1) < 1e-9


def test_exact_bootstrap_median():
    rng = np.random.default_rng(0)
    v = np.sort(rng.normal(size=11))
    lo, hi = records.exact_bootstrap_median_ci(v)
    sim = np.median(rng.choice(v, (200000, 11)), axis=1)
    assert np.mean(sim < lo) <= 0.025 + 0.003 and np.mean(sim <= lo) >= 0.025 - 0.003


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(name + ': ok', flush=True)
