"""Draw Figs. 1-6 of the article and Figs. S1-S11 of the Supplementary material from the result files and the model
functions. Run from the repository root after scripts/run_results.py:  python scripts/paper_figures.py [names]
Fig. 7 is drawn by scripts/synthetic_validation.py."""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.patches import Ellipse, FancyBboxPatch   # noqa: E402
from scipy.integrate import solve_ivp                    # noqa: E402
from scipy.optimize import brentq, curve_fit             # noqa: E402
from scipy.special import expit                          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusivity as dfm      # noqa: E402
import lemma1                  # noqa: E402
import records                 # noqa: E402
import separability as sep     # noqa: E402
from common import read, shell_fraction, thinnest_shell   # noqa: E402

plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.linewidth': 0.7, 'xtick.major.width': 0.7,
                     'ytick.major.width': 0.7, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'figure.dpi': 200, 'savefig.dpi': 400, 'savefig.bbox': 'tight',
                     'savefig.pad_inches': 0.03, 'pdf.fonttype': 42})
INK, MID, LITE = '#1a1a1a', '#6e6e6e', '#c8c8c8'
ACC, ACC2, ACC3 = '#1f4e79', '#a33a1f', '#3d7a3d'
OUT = 'figures'


def res(name):
    return read(name, 'results')


def f(x):
    return float(x)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    fig.savefig(os.path.join(OUT, name + '.png'))
    plt.close(fig)
    print('wrote figures/' + name, flush=True)


def tag(ax, s, x=-0.16):
    ax.text(x, 1.04, s, transform=ax.transAxes, fontweight='bold', fontsize=9, va='bottom')


# ------------------------------------------------------------------------------------- Lemma 1 drying (dimensionless)
def drying(n=120, phi_star=0.2, k=5.0, tmax=400.0, c_init=None, kfun=None, Rfun=None):
    r = np.linspace(0, 1, n)
    dr = r[1] - r[0]
    rf = 0.5 * (r[1:] + r[:-1])
    D = lambda c: np.exp(9.2 * (np.clip(c, 0, 1) - 0.9))
    kfun = kfun or (lambda t: k)
    Rfun = Rfun or (lambda t: 1.0)

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
    T = np.geomspace(1e-4, tmax, 6000)
    c0 = np.full(n, 0.9) if c_init is None else c_init(r)
    C = solve_ivp(rhs, (0, tmax), c0, method='BDF', t_eval=T, rtol=1e-7, atol=1e-10, max_step=0.5).y
    tf = np.array([T[np.argmax(C[i] <= phi_star)] if np.any(C[i] <= phi_star) else np.nan for i in range(n)])
    lam = lambda c: np.exp(8 * (0.9 - c))
    th = np.array([np.trapezoid(1 / lam(C[i, T <= tf[i]]), T[T <= tf[i]]) if np.isfinite(tf[i]) else np.nan
                   for i in range(n)])
    return r, T, C, tf, th


def fig1():
    r, T, C, tf, th = drying()
    fig, axs = plt.subplots(2, 2, figsize=(5.8, 4.6))
    axs = axs.ravel()
    ax = axs[0]
    for tt, col in zip((0.05, 1.0, 10.0, 60.0, 250.0), plt.cm.Blues(np.linspace(0.45, 0.95, 5))):
        ax.plot(r, C[:, np.argmin(abs(T - tt))], color=col, lw=1)
    ax.axhline(0.2, color=ACC2, lw=0.8, ls='--')
    ax.text(0.02, 0.23, 'arrest, $\\varphi^*$', color=ACC2, fontsize=6.5)
    ax.set_xlabel('radial position $r/R$')
    ax.set_ylabel('solvent fraction')
    ax.set_title('(a) drying from the surface', fontsize=7.5)
    ax = axs[1]
    ok = np.isfinite(th) & np.isfinite(tf)
    ax.plot(r[ok], th[ok], color=ACC, lw=1.4)
    ax.set_xlabel('$r/R$')
    ax.set_ylabel('clock at arrest, $\\Theta_r$')
    ax.set_title('(b) Lemma 1: $\\Theta_r$ monotone in $r$', fontsize=7.5)
    ax = axs[2]
    S = 1 / (1 + np.exp(-(r - 0.85) / 0.04))
    ax.plot(r[ok][::6], S[ok][::6], 'o', ms=2.6, color=INK)
    ax.set_xlabel('$r/R$')
    ax.set_ylabel('structure $S$')
    th_ok, r_ok = th[ok], r[ok]
    sec = ax.secondary_xaxis('top', functions=(lambda x: np.interp(x, r_ok, th_ok),
                                               lambda y: np.interp(y, th_ok[::-1], r_ok[::-1])))
    sec.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    sec.set_xlabel('$\\Theta_r$', fontsize=7)
    sec.tick_params(labelsize=6)
    ax.set_title('(c) Proposition 1: the same data on two axes', fontsize=7.5, pad=18)
    ax = axs[3]
    rr = np.linspace(0, 1, 200)
    bad = 1.2 * np.exp(-((rr - 0.45) / 0.3) ** 2) + 0.2
    ax.plot(rr, bad, color=ACC2, lw=1.4)
    ax.axhline(1.0, color=MID, lw=0.7, ls=':')
    xs = rr[np.where(np.diff(np.sign(bad - 1.0)))[0]]
    ax.plot(xs, [1.0, 1.0], 'o', ms=3, color=ACC2)
    ax.text(0.86, 1.18, 'one clock value,\ntwo radii:\nno inverse', fontsize=6.5, ha='center', color=ACC2)
    ax.set_xlabel('$r/R$')
    ax.set_ylabel('$\\Theta_r$')
    ax.set_yticks([])
    ax.set_title('(d) non-monotone arrest: the equivalence fails', fontsize=7.5)
    fig.tight_layout(w_pad=1.6, h_pad=1.4)
    save(fig, 'fig1_clocks_degeneracy')


def fig2():
    fig = plt.figure(figsize=(5.0, 3.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis('off')
    ax.add_patch(Ellipse((50, 34), 97, 66, fc='#eef3f8', ec=ACC, lw=1.1))
    ax.text(50, 62, '$\\mathcal{F}=\\mathcal{G}$ (Proposition 1)', ha='center', fontsize=9, color=ACC, fontweight='bold')
    ax.text(50, 57, 'unrestricted: any function of the clock = any function of radius', ha='center', fontsize=7.5)
    ax.add_patch(Ellipse((33, 29), 52, 36, fc='#fbeee9', ec=ACC2, lw=1.1, alpha=0.9))
    ax.add_patch(Ellipse((67, 29), 52, 36, fc='#dfe2e6', ec=INK, lw=1.1, alpha=0.7))
    ax.text(27, 34.5, 'Fading-memory\nflow models\n(Propositions 3–4)', ha='center', fontsize=8, color=ACC2,
            fontweight='bold')
    ax.text(25, 19.5, 'stretching continues after\nsurface arrest, outpaces the\nloss of retained deformation,\nnot carried by the skin:\n'
            'interior maximum', ha='center', fontsize=6.4, color=ACC2)
    ax.text(76, 37, '$\\mathcal{G}_\\alpha$: committed skin\n(shell of set\nthickness)', ha='center', fontsize=8,
            fontweight='bold')
    ax.text(77, 21, '$\\mathcal{G}_\\alpha\\subseteq\\mathcal{F}$\n(Proposition 2)', ha='center', fontsize=7)
    ax.text(50, 28, 'overlap:\nstretching ends\nbefore surface\narrest; surface\nmaximum', ha='center', fontsize=6.6)
    save(fig, 'fig2_nested_families')


def fig3():
    r = np.linspace(0.05, 1.0, 300)
    tf = 1.0 + 19.0 * (1 - r) ** 2 / np.max((1 - r) ** 2)
    before = lambda t: np.where(t < 0.8, 1.0, 0.0)
    ongoing = lambda t: np.exp(-0.5 * ((t - 5.0) / 5.0) ** 2)
    names = ('single exponential', 'two modes', 'linear ramp (not a mixture)')
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.0), sharey=True)
    titles = ('(a) stretching ends before\nsurface arrest (or skin halts it)', '(b) stretching continues,\nskin decouples',
              '(c) stretching continues,\nskin is drawn')
    for k, ax in enumerate(axs):
        for name, col in zip(names, (ACC, ACC3, ACC2)):
            K = sep.PROP4_KERNELS[name]
            prof = np.empty_like(r)
            for i, T in enumerate(tf):
                tt = np.linspace(0, T, 3000)
                prof[i] = np.trapezoid((before(tt) if k == 0 else ongoing(tt)) * K(T - tt), tt)
                if k == 2:
                    ss = np.linspace(T, 25.0, 3000)
                    prof[i] += np.trapezoid(ongoing(ss), ss)
            ax.plot(r, prof / prof.max(), color=col, lw=1.2, label=name)
            ax.plot(r[np.argmax(prof)], 1.0, 'v', color=col, ms=4)
        ax.set_title(titles[k], fontsize=7)
        ax.set_xlabel('radial position $r/R$')
    axs[0].set_ylabel('retained $\\mathcal{M}^*$ / max')
    axs[0].legend(fontsize=6.8, loc='lower left')
    fig.tight_layout(w_pad=0.8)
    save(fig, 'fig3_where_structure_peaks')


def fig4():
    base = dfm.Diffusivity('baseline')
    Ds = [dfm.Diffusivity(v) for v in dfm.VARIANTS]
    phi = np.linspace(0.10, 0.25, 151)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={'width_ratios': [1, 1.35]})
    ax = axs[0]
    ax.axvspan(0.14, 0.20, color=ACC, alpha=0.10, lw=0)
    ax.axvspan(0.20, 0.23, color=ACC, alpha=0.05, lw=0)
    ax.axvline(dfm.PHI_G, color=MID, lw=0.7, ls='--')
    ax.plot(phi, [dfm.eq9_diameter_nm(float(base(p)), 0.01) for p in phi], color=ACC, lw=1.3, label='10 ms flight')
    ax.plot(phi, [dfm.eq9_diameter_nm(float(base(p)), 0.1) for p in phi], color=ACC, lw=1.3, ls='--',
            label='100 ms flight')
    ax.plot(phi, [max(dfm.eq9_diameter_nm(float(D(p)), t) for D in Ds for t in (0.001, 0.01, 0.03, 0.1)) for p in phi],
            color=ACC2, lw=1.3, label='joint worst case')
    ax.axhline(100, color=MID, lw=0.6, ls=':')
    ax.set_yscale('log')
    ax.set_xlim(0.10, 0.25)
    ax.set_ylim(3, 3000)
    ax.text(0.142, 1300, 'glass\ncomposition', fontsize=6, color=MID, va='top')
    ax.text(0.17, 1700, 'arrest range', fontsize=6, color=ACC, ha='center', va='bottom')
    ax.set_xlabel('arrest composition $\\varphi^*$ (solvent volume fraction)')
    ax.set_ylabel('uniform-arrest diameter (nm)')
    ax.legend(fontsize=7, loc='lower right')
    tag(ax, 'a')
    # (b) envelope at 100 ms, baseline diffusivity, from the result files
    a3 = res('a3_post_arrest.csv')
    sel = lambda m, col, v=None: [f(r[col]) / 1000 for r in a3 if r['mode'] == m and f(r['flight_ms']) == 100
                                  and (v is None or r['diffusivity_variant'] == v)]
    net_ua_c, net_ua_all = sel('network', 'flat_diameter_nm', 'baseline'), sel('network', 'flat_diameter_nm')
    net_a3_c, net_a3_all = sel('network', 'd_nm_10pct_volume_loss', 'baseline'), sel('network', 'd_nm_10pct_volume_loss')
    al_a3_c, al_a3_all = sel('alpha', 'd_nm_10pct_volume_loss', 'baseline'), sel('alpha', 'd_nm_10pct_volume_loss')
    rows = [('published in-air record,\nrising part (onset ≤ 2.5 µm)', (0.5, 2.5), None, INK),
            ('A2: polarized Raman in air', (0.5, 10), None, ACC),
            ('A2: immersion (checked on average\nat 0.4–0.6 µm; 0.14 µm target)', (0.4, 0.6), (0.14, 10), ACC),
            ('A1 permits radial variation (above the\nuniform-arrest diameter, network)',
             (max(net_ua_c), 10), (min(net_ua_c), max(net_ua_c)), ACC3),
            ('A3: network-carried orientation', (0.03, min(net_a3_c)), (min(net_a3_c), max(net_a3_c)), ACC3),
            ('A3: α-timescale channels', (0.03, min(al_a3_c)), (min(al_a3_c), max(al_a3_c)), ACC3),
            ('test below the uniform-arrest diameter:\nreadout floor to uniform-arrest diameter',
             (0.14, max(net_ua_c)), None, ACC2)]
    ax = axs[1]
    for i, (lab, solid, unc, col) in enumerate(rows[::-1]):
        ax.barh(i, solid[1] - solid[0], left=solid[0], height=0.55, color=col, alpha=0.75, lw=0)
        if unc:
            ax.barh(i, unc[1] - unc[0], left=unc[0], height=0.55, color='none', edgecolor=col, lw=0.8, hatch='////')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=6.0)
    ax.set_xscale('log')
    ax.set_xlim(0.03, 10)
    lo, hi = max(0.5, min(net_ua_c)), min(2.5, max(net_a3_c))
    ax.axvline(0.5, color=INK, lw=0.6, ls='--')
    ax.axvline(min(2.5, max(net_a3_c)), color=INK, lw=0.6, ls='--')
    ax.axvspan(0.5, min(2.5, max(net_a3_c)), color=INK, alpha=0.05, lw=0)
    ax.text(1.1, len(rows) - 0.3, 'rising part of\nthe PS record', ha='center', fontsize=6, color=MID)
    ax.set_ylim(-0.6, len(rows) + 0.3)
    ax.set_xlabel('fiber diameter (µm)')
    tag(ax, 'b', -0.02)
    fig.tight_layout(w_pad=1.0)
    save(fig, 'fig4_uniform_arrest_envelope')


BOX = {'A': ('#e8eef6', ACC), 'L': ('#e7f2e7', ACC3), 'P': ('#fbeee9', ACC2), 'C': ('#f1ebf6', '#5b3f7a'),
       'V': ('#f2f2f2', INK), 'S': ('#ffffff', INK)}


def _box(ax, x, y, w, h, text, kind='V', fs=6.2, bold=False):
    fc, ec = BOX[kind]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.35,rounding_size=1.2', fc=fc,
                                ec=ec, lw=0.9))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal',
            linespacing=1.15)


def _arrow(ax, x1, y1, x2, y2, label=None, lx=0, ly=0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', lw=0.8, color=INK, shrinkA=0,
                                                                  shrinkB=0, mutation_scale=7))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=6, style='italic', ha='center', va='center')


def fig5():
    fig = plt.figure(figsize=(7.0, 4.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis('off')
    _box(ax, 36, 62, 56, 4, 'Structure measured against radius or diameter', 'S', 7, True)
    _arrow(ax, 36, 59.6, 36, 56.4)
    _box(ax, 36, 53.2, 56, 5.4, '1. Do A1–A3 hold for these fibers\n(readout, stability, arrest order)?', 'A', 6.8)
    _box(ax, 84, 53.2, 26, 7, 'Revalidate the readout (A2)\nand its stability (A3) first;\nif arrest is not radially\nordered (A1), Route 1', 'V', 5.8)
    _box(ax, 84, 43.6, 26, 7.6, 'Below a uniform-arrest\ndiameter (where one exists),\na readout rise beyond LgΓΔΘ*\nexcludes a bounded-slope\nskin (Remark 1)', 'V', 5.6)
    _arrow(ax, 64.4, 43.6, 70.6, 43.6, 'also', ly=1.3)
    _arrow(ax, 64.4, 53.2, 70.6, 53.2, 'no', ly=1.3)
    _arrow(ax, 36, 50.3, 36, 46.6, 'yes', lx=2.4)
    _box(ax, 36, 43.6, 56, 5, 'Unrestricted accounts are equivalent here (Proposition 1):\ncommit to a model or change '
         'the measurement', 'V', 6.2)
    _arrow(ax, 36, 40.9, 36, 37.8)
    _box(ax, 36, 34.8, 56, 5, '2. When does stretching end relative\nto surface arrest? (jet imaging)', 'P', 6.8)
    _arrow(ax, 26, 32.1, 14, 26.6, 'before', lx=-3.5)
    _arrow(ax, 48, 32.1, 60, 26.6, 'after', lx=3)
    _box(ax, 13, 23.4, 22, 6, 'Route 3: change the\nrelaxation time, compare\nat matched conformation', 'C', 6)
    _box(ax, 60, 23.4, 36, 4.6, '3. How does the arrested skin respond?', 'P', 6.8)
    _arrow(ax, 50, 21, 37, 12.6, 'decouples', lx=-4, ly=0.5)
    _arrow(ax, 60, 21, 62, 12.6, 'drawn', lx=2.6)
    _arrow(ax, 70, 21, 87, 12.6, 'halts extension', lx=5, ly=0.5)
    _box(ax, 13, 9.4, 22, 7, 'Flow-induced phase\nseparation: Route 2 in\nevery regime (Prop. 5)', 'V', 6)
    _box(ax, 37, 9.4, 21, 7, 'Route 4: radial position\nof the maximum (interior\nexcludes a smooth skin)', 'C', 6)
    _box(ax, 63, 9.4, 21, 7, 'Route 2: an independent\nstructural channel (Remark 3);\nwithout one, cannot decide', 'C', 6)
    _box(ax, 88, 9.4, 17, 7, 'stretching ends at\nsurface arrest:\nRoute 3', 'C', 6)
    save(fig, 'fig5_selecting_route')


def fig6():
    d, p, hw = records.ps_bins()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axs[0]
    dd = np.linspace(300, 5200, 400)
    ax.plot(dd, records.published_fit(dd), color=MID, lw=0.9, ls='--', label='published fit, 0.428 exp(−0.0010 $d$)')
    for delta, ls in ((13.1, ':'), (30.0, '-'), (60.0, '-.')):
        ax.plot(dd, shell_fraction(dd, delta), color=ACC2, lw=1.1, ls=ls, label='perfectly aligned %g nm skin' % delta)
    ax.plot(dd, 0.3 * shell_fraction(dd, 60.0), color=ACC, lw=1.0, ls=':', label='60 nm skin at $\\langle P_2\\rangle_{skin}$ = 0.3')
    kap = [r for r in res('raman_weighting_kappa.csv') if r['polarization'] == 'max' and f(r['delta_nm']) == 30]
    for setting, mk, lab in (('NA 0.8, centred (central)', 's', '30 nm skin, confocal weight (NA 0.8)'),
                             ('Unfocused plane wave', 'x', '30 nm skin, unfocused plane wave')):
        pts = sorted((f(r['d_nm']), f(r['kappa'])) for r in kap if r['setting'] == setting)
        ax.plot([a for a, _ in pts], [k * float(shell_fraction(a, 30.0)) for a, k in pts], mk, ms=3.2,
                mfc='none' if mk == 's' else None, color=INK, label=lab)
    ax.plot(d, p / 2, 'o', ms=2.6, mfc='none', color=MID, label='bin means halved (half added by the disk)')
    ax.errorbar(d, p, yerr=hw, fmt='o', ms=3.4, color=ACC3, mec=INK, mew=0.5, ecolor=INK, elinewidth=0.6, capsize=1.5,
                label='bin means (supplied by the authors)')
    ax.set_xlim(0, 5200)
    ax.set_ylim(0, 0.42)
    ax.set_xlabel('fiber diameter $d$ (nm)')
    ax.set_ylabel('mean orientation $\\langle P_2\\rangle$')
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=2)
    tag(ax, 'a')
    ax = axs[1]
    rows = res('pet_plate_skin_bound.csv')
    dl = np.geomspace(5.0, 150.0, 400)
    phi = shell_fraction(records.PET_SPOT_NM, dl)
    allow = np.minimum(1.0, records.PET_CEILING / phi)
    typical = np.minimum(1.0, records.PET_MEDIAN / phi)
    ax.axhspan(1.0, 20.0, color=LITE, alpha=0.45, lw=0)
    ax.text(40.0, 3.0, 'beyond perfect\nalignment', color=MID, fontsize=6.5, va='center', ha='center')
    ax.fill_between(dl, allow, 1.0, color=ACC2, alpha=0.10, lw=0)
    ax.plot(dl, records.PET_DISK / phi, color=ACC2, lw=1.1, ls='--', label='needed for the disk median, 0.24')
    ax.plot(dl, records.PET_GAP / phi, color=ACC2, lw=1.1, label='needed for the gap median, 0.18')
    ax.plot(dl, allow, color=ACC, lw=1.4, label='allowed: stated plate ceiling, 0.04')
    ax.plot(dl, typical, color=ACC, lw=1.0, ls='--', label='allowed: stated plate median, 0.02')
    for r in rows:
        if f(r['plate_orientation_ceiling']) == records.PET_CEILING:
            d0, v = f(r['skin_thickness_nm']), f(r['max_skin_orientation'])
            ax.plot([d0], [v], 'o', color=ACC, ms=3.2, zorder=5)
            ax.annotate('%.2f' % v, (d0, v), xytext=(-3, 5), textcoords='offset points', fontsize=6.5, color=ACC, ha='right')
    ax.text(112.0, 0.15, 'excluded by the\nplate ceiling', color=ACC2, fontsize=5.8, ha='center', va='center')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(5.0, 150.0)
    ax.set_ylim(0.01, 8.0)
    ax.set_xticks([5, 10, 20, 50, 100])
    ax.set_xticklabels(['5', '10', '20', '50', '100'])
    ax.set_xlabel('skin thickness $\\delta$ (nm)')
    ax.set_ylabel('skin orientation $s_{\\mathrm{skin}}$')
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=1)
    tag(ax, 'b')
    fig.tight_layout(w_pad=1.6)
    save(fig, 'fig6_record_bounds')


# ------------------------------------------------------------------------------------------- Supplementary figures
def figS1():
    import finite
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5))
    ax = axs[0]
    shape = finite.SHAPES['continues after surface arrest']
    for target, col in ((0.25, ACC), (0.75, ACC2)):
        wi = finite.wi_for_mean(shape, 100.0, target)
        prof = finite.profile(shape, wi, 100.0)
        ax.plot(finite.R, prof, color=col, lw=1.2, label='continues, mean %.2f (peak Wi %.1f)' % (target, wi))
        k = int(np.argmax(prof))
        ax.plot(finite.R[k], prof[k], 'v', color=col, ms=4)
    prof = finite.profile(finite.SHAPES['ends before surface arrest'], 5.0, 100.0)
    ax.plot(finite.R, prof, color=MID, lw=1.0, ls='--', label='ends before surface arrest')
    ax.set_xlabel('radial position $r/R$')
    ax.set_ylabel('$\\langle P_2\\rangle$')
    ax.legend(fontsize=5.6, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=1)
    tag(ax, 'a', -0.2)
    ax = axs[1]
    rows = res('finite_extensibility_contrast.csv')
    for b, col in ((20.0, ACC3), (100.0, ACC), (1000.0, ACC2)):
        pts = sorted((f(r['mean_P2']), f(r['interior_excess'])) for r in rows if f(r['b']) == b)
        ax.plot(*zip(*pts), 'o-', ms=2.5, color=col, lw=1, label='b = %g' % b)
    ax.axvspan(0.2, 0.31, color=LITE, alpha=0.5, lw=0)
    ax.axvline(0.74, color=MID, lw=0.7, ls='--')
    ax.set_xlabel('cross-sectional mean $\\langle P_2\\rangle$')
    ax.set_ylabel('interior maximum − surface value')
    ax.legend(fontsize=6)
    tag(ax, 'b', -0.24)
    fig.tight_layout(w_pad=1.5)
    save(fig, 'figS1_finite_extensibility')


def figS2():
    fig, axs = plt.subplots(2, 2, figsize=(5.8, 4.4), sharex=True)
    cases_a = (('uniform start', None), ('polymer-rich core', lambda r: np.where(r < 0.3, 0.25, 0.9)),
               ('solvent-rich annulus', lambda r: 0.3 + 0.6 * np.exp(-((r - 0.6) / 0.12) ** 2)))
    cases_b = (('constant evaporation', {}), ('evaporation on/off', {'kfun': lambda t: 5.0 if int(t / 2) % 2 == 0 else 0.0}),
               ('oscillating evaporation', {'kfun': lambda t: 5.0 * (1 + 0.95 * np.sin(3 * t))}),
               ('thinning jet', {'Rfun': lambda t: np.exp(-min(t, 40) / 4)}))
    styles = ('-', '--', '-.', ':')
    cols = (INK, ACC, ACC2, ACC3)
    for j, cases in enumerate((cases_a, cases_b)):
        for (lab, kw), ls, col in zip(cases, styles, cols):
            kw = {'c_init': kw} if j == 0 else kw
            r, T, C, tf, th = drying(**kw)
            ok = np.isfinite(tf) & np.isfinite(th)
            axs[0, j].plot(r[ok], tf[ok] / np.nanmax(tf), ls=ls, color=col, lw=1, label=lab)
            axs[1, j].plot(r[ok], th[ok] / np.nanmax(th), ls=ls, color=col, lw=1, label=lab)
        axs[0, j].legend(fontsize=5.5)
    axs[0, 0].set_title('(a) non-uniform starting composition', fontsize=7)
    axs[0, 1].set_title('(b) time-dependent evaporation', fontsize=7)
    axs[1, 0].set_title('(c) clock at arrest, same cases as (a)', fontsize=7)
    axs[1, 1].set_title('(d) clock at arrest, same cases as (b)', fontsize=7)
    for ax in axs[0]:
        ax.set_ylabel('arrest time $t_f(r)$ / max', fontsize=7)
    for ax in axs[1]:
        ax.set_ylabel('clock $\\Theta_r(r)$ / max', fontsize=7)
        ax.set_xlabel('radial position $r/R$')
    fig.tight_layout()
    save(fig, 'figS2_lemma1_check')


def figS3():
    a = np.linspace(0.02, 0.6, 200)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.axhspan(0.14, 0.20, color=ACC3, alpha=0.12, lw=0)
    ax.text(0.03, 0.17, 'adopted arrest range,\n$\\varphi^*$ = 0.14–0.20', color=ACC3, fontsize=6.5, va='center')
    for chi, ls in ((0.05, '-'), (0.12, '--')):
        ax.plot(a, [lemma1.flory_solvent_fraction(x, chi) for x in a], color=INK, ls=ls, lw=1.1,
                label='Flory relation, $\\chi$ = %g' % chi)
    comp = res('arrest_composition.csv')
    for br, mk, col, lab in (('absorption', 'o', ACC, 'glass transition, absorption'),
                             ('desorption', 's', ACC2, 'glass transition, desorption')):
        pts = [(f(r['solvent_activity']), f(r['solvent_volume_fraction'])) for r in comp if r['branch'] == br]
        ax.plot(*zip(*pts), mk, color=col, ms=4, label=lab)
    ax.axhline(0.07, color=MID, lw=0.8, ls=':')
    ax.text(0.58, 0.055, 'Fox estimate, excluded', color=MID, fontsize=6, ha='right')
    ax.set_xlabel('chloroform activity')
    ax.set_ylabel('solvent volume fraction')
    ax.legend(fontsize=5.8, loc='upper left', bbox_to_anchor=(0, 1.02))
    ax.set_ylim(0, 0.32)
    save(fig, 'figS3_arrest_composition')


def figS4():
    rows = [r for r in res('a3_post_arrest.csv') if r['mode'] in ('alpha', 'network')]
    dep = res('a3_flat_dependence.csv')
    pooled = [r for r in dep if r['grouping'] == 'pooled'][0]
    lo, hi = f(pooled['ratio_min']), f(pooled['ratio_max'])
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    x = np.geomspace(0.01, 5, 50)
    ax.fill_between(x, lo * x, hi * x, color=LITE, alpha=0.5, lw=0)
    ax.plot(x, x, color=MID, lw=0.7, ls=':')
    ax.axhspan(0.5, 2.5, color=INK, alpha=0.05, lw=0)
    for r in rows:
        if r['diffusivity_variant'] != 'baseline':
            ax.plot(f(r['flat_diameter_nm']) / 1000, f(r['d_nm_10pct_volume_loss']) / 1000, '.', color=MID, ms=2.5)
    for m, mk, col, lab in (('alpha', 'o', ACC, 'baseline, α-timescale channels'),
                            ('network', '^', ACC3, 'baseline, network orientation')):
        pts = [(f(r['flat_diameter_nm']) / 1000, f(r['d_nm_10pct_volume_loss']) / 1000) for r in rows
               if r['diffusivity_variant'] == 'baseline' and r['mode'] == m]
        ax.plot(*zip(*pts), mk, mfc='none', color=col, ms=4, label=lab)
    ax.plot([], [], '.', color=MID, label='diffusivity variants')
    ax.text(0.012, 9, 'shaded: A3 limit =\n%.1f–%.1f × uniform-arrest diameter' % (lo, hi), fontsize=6, color=MID)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.01, 5)
    ax.set_ylim(0.02, 40)
    ax.set_xlabel('uniform-arrest diameter (µm)')
    ax.set_ylabel('A3 limit, 10% post-arrest loss (µm)')
    ax.legend(fontsize=5.8, loc='lower right')
    save(fig, 'figS4_a3_limit')


def figS5():
    tab = res('chloroform_diffusivity.csv')
    phi = [f(r['phi']) for r in tab]
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    ax.axvspan(0.14, 0.23, color=LITE, alpha=0.5, lw=0)
    ax.axvline(dfm.PHI_G, color=MID, lw=0.7, ls='--')
    pb = read('pickup_blum_1989_table1.csv')
    for r in pb:
        if f(r['T_C']) != 25:
            ax.plot(1 - f(r['w_PS']), f(r['D_self_1e-10_m2_per_s']) * 1e-10, '.', color=LITE, ms=3)
    p25 = [(1 - f(r['w_PS']), f(r['D_self_1e-10_m2_per_s']) * 1e-10) for r in pb if f(r['T_C']) == 25]
    ax.plot(*zip(*p25), 'o', mfc='none', color=INK, ms=3.5, label='toluene measured, 25 °C')
    w = np.linspace(0.01, 1.0, 300)
    ax.plot(w, dfm.self_diffusivity_mass(w, dfm.T_25, dfm.TOLUENE, dfm.TOLUENE['xi']), color=ACC, lw=1,
            label='toluene free-volume, 25 °C')
    ax.plot(phi, [f(r['free_volume_25C']) for r in tab], ':', color=ACC3, lw=1, label='chloroform free volume, 25 °C')
    ax.plot(phi, [f(r['free_volume_viscosity_25C']) for r in tab], '-.', color=MID, lw=1, label='26% larger chloroform free volume, 25 °C')
    ax.plot(phi, [f(r['free_volume_xi_fitted_25C']) for r in tab], '--', color=ACC3, lw=1, label='ξ fitted to solution data, 25 °C')
    cp_phi, cp_D = dfm.crank_park()
    ax.plot(cp_phi, cp_D, 's', color=ACC3, mec=INK, mew=0.4, ms=3.5, label='chloroform measured (glass), 25 °C')
    ax.plot(phi, [f(r['baseline']) for r in tab], color=ACC2, lw=1.6, label='PS–chloroform diffusivity, 18.5 °C')
    ax.set_yscale('log')
    ax.set_ylim(1e-18, 1e-8)
    ax.set_xlim(0, 0.6)
    ax.set_xlabel('solvent volume fraction')
    ax.set_ylabel('diffusivity (m$^2$ s$^{-1}$)')
    ax.legend(fontsize=5.2, loc='lower right')
    save(fig, 'figS5_diffusivity')


def figS6():
    rows = res('bayes_factors_commitment_error.csv')
    x = [f(r['error_in_committed_upper_level']) for r in rows]
    y = [f(r['bayes_factor']) for r in rows]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.semilogy(x, y, 'o-', color=INK, ms=2.8, lw=1)
    ax.axhline(1, color=MID, lw=0.7, ls=':')
    ax.fill_between(x, 1, 1e4, color=ACC, alpha=0.07, lw=0)
    ax.fill_between(x, 1e-2, 1, color=ACC2, alpha=0.07, lw=0)
    ax.text(0.004, 2500, 'committed skin supported', fontsize=6.5, color=ACC)
    ax.text(0.004, 0.02, 'committed skin disfavored', fontsize=6.5, color=ACC2)
    ax.set_ylim(0.01, 1e4)
    ax.set_xlabel('error in the committed plateau level')
    ax.set_ylabel('Bayes factor, skin : flow')
    save(fig, 'figS6_bayes_factor')


def figS7():
    d = np.linspace(*sep.WINDOW, 400)
    fsh = sep.shell_fraction(d, 30.0)
    p, _ = curve_fit(sep.exponential, d, fsh, p0=[fsh[0], 500.0, 0.0], maxfev=200000)
    g = sep.exponential(d, *p)
    cr = res('functional_forms_crossings.csv')
    xs = [f(r['crossing_diameter_nm']) for r in cr]
    ys = [f(r['common_value']) for r in cr]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.0))
    ax = axs[0]
    ax.plot(d, fsh, color=INK, lw=1.3, label='shell, $\\delta$ = 30 nm')
    ax.plot(d, g, color=ACC2, lw=1.1, ls='--', label='fitted exponential')
    ax.plot(xs, ys, 'o', ms=3, color=ACC)
    ax.set_xscale('log')
    ax.set_xlabel('diameter (nm)')
    ax.set_ylabel('structure')
    ax.legend(fontsize=6)
    ax.set_title('(a) forms that cross', fontsize=7.5)
    d0 = brentq(lambda x: sep.shell_fraction(x, 30.0) - sep.exponential(x, *p), 300, 700)
    s0 = float(sep.shell_fraction(d0, 30.0))
    ax = axs[1]
    ax.step(d, (fsh >= s0).astype(float), where='mid', color=INK, lw=2.2, alpha=0.5, label='shell')
    ax.step(d, (g >= s0).astype(float), where='mid', color=ACC2, lw=1.0, ls='--', label='exponential')
    ax.set_xscale('log')
    ax.set_ylim(-0.1, 1.2)
    ax.set_yticks([0, 1])
    ax.set_xlabel('diameter (nm)')
    ax.set_ylabel('$g(S)=\\mathbf{1}[S\\geq s_0]$')
    ax.legend(fontsize=6, loc='center right')
    ax.set_title('(b) threshold at the crossing:\nidentical', fontsize=7.5)
    ax = axs[2]
    sp_ = res('functional_forms_separation.csv')
    k = [f(r['map_steepness_k']) for r in sp_]
    ax.loglog(k, [f(r['relative_separation_crossing_pair']) for r in sp_], color=ACC, lw=1.3, label='profiles that cross')
    ax.loglog(k, [f(r['relative_separation_noncrossing_pair']) for r in sp_], color=MID, lw=1.1, ls='--',
              label='profiles that never cross')
    ax.set_xlabel('steepness $k$ of the map')
    ax.set_ylabel('separation / range of $g\\circ p_1$')
    ax.legend(fontsize=6, loc='lower left')
    ax.set_title('(c) relative separation\nunder steepening maps', fontsize=7.5)
    fig.tight_layout(w_pad=1.0)
    save(fig, 'figS7_functional_forms')


def figS8(L=2.7, rho=2.0):
    g = lambda x: (1 - np.exp(-x)) / x
    Wi = np.logspace(0.7, 3, 100)
    b1 = L / np.log(rho) * (1 - g(2 * L / Wi) / g(2 * L / Wi / rho))
    x = np.logspace(-2, 0.5, 100)
    b2 = L / np.log(rho) * (1 - np.exp(-x * (1 - 1 / rho)))
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.1), sharey=True)
    ax = axs[0]
    ax.loglog(Wi, b1, color=ACC2, lw=1.4)
    ax.set_xlabel('Weissenberg number Wi')
    ax.set_ylabel('coupling $\\beta$')
    ax.set_title('stretching until arrest (Eq. S6)', fontsize=7)
    for w in (100, 1000):
        v = float(np.interp(w, Wi, b1))
        ax.plot(w, v, 'o', ms=3, color=ACC2)
        ax.text(w * 1.1, v * 1.25, '%.2g' % v, fontsize=6)
    ax = axs[1]
    ax.loglog(x, b2, color=ACC, lw=1.4)
    ax.set_xlabel('relaxation before arrest, $\\Delta t/\\lambda$')
    ax.set_title('stretching complete, then relaxation (Eq. S7)', fontsize=7)
    for xv in (0.1, 1.0):
        v = float(np.interp(xv, x, b2))
        ax.plot(xv, v, 'o', ms=3, color=ACC)
        ax.text(xv * 1.1, v * 0.62, '%.2g' % v, fontsize=6)
    for ax, s in zip(axs, 'ab'):
        tag(ax, s, -0.2)
    fig.tight_layout(w_pad=1.0)
    save(fig, 'figS8_coupling')


def figS9():
    a3 = res('a3_post_arrest.csv')
    cen = lambda m, col: [f(r[col]) / 1000 for r in a3 if r['diffusivity_variant'] == 'baseline' and r['mode'] == m
                          and f(r['flight_ms']) == 100]
    na = cen('network', 'd_nm_10pct_volume_loss')
    allua = [f(r['flat_diameter_nm']) / 1000 for r in res('flat_diameter_consistent.csv') if r['mode'] == 'network']
    dep = [r for r in res('a3_flat_dependence.csv') if r['grouping'] == 'pooled'][0]
    fig = plt.figure(figsize=(7.2, 9.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 128)
    ax.axis('off')
    _box(ax, 34, 121, 60, 5, 'Structure measured against radius (one fiber) or diameter (across fibers)', 'S', 6.8, True)
    gates = [
        (110.5, 'A3 (Sec. 4.2): does the measured channel stay unchanged after local arrest?\npolystyrene–chloroform, '
                '100 ms flight: yes below about %.2g–%.2g µm (network orientation,\ncentral inputs); %.1f–%.1f times '
                'the uniform-arrest diameter' % (min(na), max(na), f(dep['ratio_min']), f(dep['ratio_max'])), 'A',
         'Collected state is not the\narrested state; post-arrest\nrelaxation can itself steepen\na diameter trend (Remark 2)'),
        (98, 'A2 (Sec. 2.2): is the readout one strictly monotone function of structure,\nunchanged across the radii and '
             'diameters compared?', 'A', 'Revalidate the readout first,\ne.g. index-matched immersion\nfor polarized '
                                         'Raman below\nabout 500 nm'),
        (86, 'A1 (Sec. 2.3): is arrest radially ordered?  Lemma 1 (Sec. 2.3): yes, outward-first,\nfor a homogeneous '
             'binary solution drying by axisymmetric diffusion at fixed radius,\nwith outward evaporative flux of any time '
             'dependence', 'L', 'Route 1: non-monotone\narrest breaks the equivalence\n(Corollary 4; no known\nrealization)'),
        (74, 'Is the fiber thicker than the uniform-arrest diameter (Sec. 4.1)?\npolystyrene–chloroform: about '
             '%.2f–%.1f µm for 10–200 ms flights (network)' % (min(allua), max(allua)), 'S',
         'Below it, a skin that is one\nfunction of the post-arrest clock,\nslope ≤ Γ, adds at most 0.1Γ radial\ncontrast and a trend of at most\nΓΔΘ*, ΔΘ* the largest span (Remark 1)'),
    ]
    hgt = lambda q: 3.2 + 2.0 * q.count('\n')
    _arrow(ax, 34, 118.4, 34, 110.5 + hgt(gates[0][1]) / 2 + 0.4)
    for i, (y, q, kind, out) in enumerate(gates):
        _box(ax, 34, y, 60, hgt(q), q, kind, 5.8)
        _box(ax, 84, y, 26, 10, out, 'V', 5.4)
        _arrow(ax, 64.4, y, 70.6, y, 'no', ly=1.3)
        ny = gates[i + 1][0] + hgt(gates[i + 1][1]) / 2 + 0.4 if i + 1 < len(gates) else 66.5
        _arrow(ax, 34, y - hgt(q) / 2 - 0.4, 34, ny, 'yes', lx=2.4)
    _box(ax, 50, 64, 72, 4.6, 'Which account, and how much does it commit to?', 'S', 7, True)
    for x in (12, 36, 64, 90):
        _arrow(ax, 50, 61.4, x, 57.6)
    for x, w, t in ((12, 21, 'flow history or skin,\nunrestricted'), (36, 21, 'skin of parametrized\nthickness'),
                    (64, 25, 'flow history committed\nto a fading memory'), (90, 18, 'flow-induced\nphase separation')):
        _box(ax, x, 54.6, w, 5.4, t, 'S', 6.0, True)
    for x, a, b, c in ((12, 'Proposition 1 (A1–A3):\nidentical profile families;\nCorollary 1: every functional\nof '
                           'one fiber\'s profile', 'Corollary S1 (A5, Lemma S1):\none function of both histories\n'
                           'represents cross-fiber trends;\nweakens when jets split',
                        'observationally equivalent:\nnot identifiable in this\nmeasurement class'),
                       (36, 'Proposition 2: the parametrized\nskin family is contained in\nthe unrestricted flow-history\n'
                            'family, $\\mathcal{G}_\\alpha\\subseteq\\mathcal{F}$',
                        'Corollary 2: data can refute\nthe committed skin model and\nstay compatible with the\nunrestricted '
                        'flow family', 'asymmetric\nrefutability')):
        _arrow(ax, x, 51.8, x, 48.4)
        _box(ax, x, 44.4, 21, 7.2, a, 'P', 5.4)
        _arrow(ax, x, 40.6, x, 37.4)
        _box(ax, x, 33.4, 21, 7.2, b, 'C', 5.4)
        _arrow(ax, x, 29.6, x, 26.8)
        _box(ax, x, 23.6, 21, 5.6, c, 'V', 5.6, True)
    _arrow(ax, 64, 51.8, 64, 49)
    _box(ax, 64, 46, 25, 5.4, 'Proposition 3 (any non-increasing kernel):\ndoes stretching continue after surface arrest?',
         'P', 5.2)
    _arrow(ax, 58, 43.1, 56, 39.6, 'no', lx=-2.2)
    _arrow(ax, 70, 43.1, 73, 39.6, 'yes', lx=2.2)
    _box(ax, 55.5, 36.3, 14, 6, '(a) maximum at the\nsurface; matches a\nskin to 0.02–0.07', 'V', 5.1)
    _box(ax, 75, 36.3, 19, 5.4, 'how does the arrested\nskin respond? (Sec. 3.3)', 'P', 5.4)
    _arrow(ax, 69, 33.4, 60, 29.8, 'halts', lx=-2.4)
    _arrow(ax, 75, 33.4, 75, 29.8, 'drawn', lx=3.0)
    _arrow(ax, 81, 33.4, 89, 29.8, 'decouples', lx=4.6, ly=1.0)
    _box(ax, 58, 26.6, 12, 5.2, 'stretching\nends: case (a)', 'V', 5.2)
    _box(ax, 74.5, 26.6, 12, 5.2, '(c) maximum\nat the surface', 'V', 5.2)
    _box(ax, 90.5, 25.8, 15, 7.2, '(b) interior maximum,\nif new stretching outpaces\nloss of retained deformation;\nProposition 4 for a\n'
                                  'composition-dependent\nclock', 'V', 4.9)
    ax.text(74.5, 21.2, 'Remark 3: Routes 3 and 4\nmay not discriminate;\nadd a second channel', fontsize=4.8,
            ha='center', va='center', color=ACC2)
    _arrow(ax, 90.5, 21.9, 90.5, 19.2)
    _box(ax, 84, 16, 26, 5.4, 'Corollary 3: the radial position of the\nmaximum is the robust discriminant →\nRoute 4, '
                              'if decoupling is shown independently', 'C', 5.0)
    _arrow(ax, 90, 51.8, 90, 48.8)
    _box(ax, 90, 46, 18, 5.2, 'Proposition 5 (A4):\nis ψ = h(S)?', 'P', 5.5)
    ax.text(91.2, 41.8, 'h invertible: equivalent\nthrough one channel', fontsize=4.8, ha='center', va='center')
    ax.text(91.2, 38.6, 'h not invertible:\nseparable in principle', fontsize=4.8, ha='center', va='center')
    ax.text(91.2, 35.4, 'ψ still varies where S\nsaturates: Route 2', fontsize=4.8, ha='center', va='center', color=ACC2)
    _box(ax, 24, 14.6, 44, 6.2, 'Proposition S1: when two monotone diameter profiles cross at an interior\ndiameter, fitted '
                                'functional forms cannot decide under a monotone\nreadout of unknown form', 'P', 5.5)
    _box(ax, 50, 5.4, 96, 9.2, 'Corollary 4 (Sec. 4.3; Table 2): exits from the equivalence.  Route 1, non-monotone '
         'radial arrest (no known realization).  Route 2, an independent\nstructural channel (all three accounts).  Route 3, a '
         'relaxation-time perturbation at matched conformation: coupling β ≈ 0.005–0.05 while stretching\ncontinues to '
         'arrest, 0.19–1.5 if the chains relax before arrest (Eqs. S6–S7).  Route 4, the radial position of the maximum '
         '(stretching after surface\narrest and a decoupled skin, shown independently).  Jet imaging selects between Routes 3 '
         'and 4 (Sec. 4.4; Fig. 5).  Each route decides\nonly if the accounts make incompatible predictions for what it '
         'measures.', 'C', 5.1)
    for i, (k, lab) in enumerate((('A', 'assumption'), ('L', 'lemma'), ('P', 'proposition'), ('C', 'corollary'),
                                  ('V', 'outcome'))):
        fc, ec = BOX[k]
        ax.add_patch(FancyBboxPatch((6 + i * 18, 125.6), 2.6, 1.5, boxstyle='round,pad=0.1', fc=fc, ec=ec, lw=0.8))
        ax.text(9.8 + i * 18, 126.35, lab, fontsize=6.3, va='center')
    save(fig, 'figS9_analysis_map')


def figS10():
    col, p2, xc, ft, dia = records.pet_fibers()
    from scipy.optimize import isotonic_regression
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3))
    st = {'plate': ('s', 'none', MID, 'aluminum plate'), 'gap': ('D', ACC3, ACC3, 'gap collector'),
          'disk': ('o', ACC2, ACC2, 'rotating disk')}
    for ax, y, lab, s in zip(axs, (ft, xc, ft - xc), ('trans fraction $F_T$', 'crystallinity $X_c$',
                                                       'mesophase $F_M = F_T - X_c$'), 'abc'):
        for k, (mk, fc, ec, name) in st.items():
            m = col == k
            ax.plot(p2[m], y[m], mk, mfc=fc, mec=ec, ms=2.6, mew=0.6, ls='none', label=name)
        o = np.argsort(p2, kind='stable')
        ax.plot(p2[o], isotonic_regression(y[o]).x, color=INK, lw=0.9)
        ax.set_xlabel('$\\langle P_2\\rangle$')
        ax.set_ylabel(lab)
        tag(ax, s, -0.25)
    axs[0].legend(fontsize=5.6, loc='upper left')
    fig.tight_layout(w_pad=1.0)
    save(fig, 'figS10_pet_channels')


def figS11():
    r = read('knapp_laramee_pellerin_2026_fig3.csv')
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax = axs[0]
    x = np.linspace(-0.05, 0.65, 10)
    ax.fill_between(x, x - 0.05, x + 0.05, color=LITE, alpha=0.6, lw=0)
    ax.plot(x, x, color=INK, lw=0.8)
    for med, mk, c in (('oil', '^', ACC2), ('water', 's', ACC3)):
        pts = [(f(q['P2_calibrated']), f(q['P2_four_spectrum'])) for q in r if q['medium'] == med]
        ax.plot(*zip(*pts), mk, color=c, ms=4, ls='none',
                label='%s (%d)' % ('silicone oil' if med == 'oil' else 'water', len(pts)))
    ax.set_xlabel('calibrated $\\langle P_2\\rangle$ (air)')
    ax.set_ylabel('four-spectrum $\\langle P_2\\rangle$')
    ax.legend(fontsize=6)
    tag(ax, 'a', -0.2)
    ax = axs[1]
    d = np.linspace(300, 800, 100)
    smax = max(f(q['P2_calibrated']) for q in r if q['medium'] == 'oil')
    ax.axvspan(400, 600, color=LITE, alpha=0.5, lw=0)
    for src, ls in (('calibrated', '-'), ('oil', '--')):
        m, c = records.PAN_STATED[src]
        for s, colr in ((1.0, ACC), (round(smax, 2), ACC2)):
            ax.plot(d, thinnest_shell(d, (m - c) / (s - c)), ls=ls, color=colr, lw=1.1,
                    label='%s, $s_{skin}$ = %g' % (src, s))
    ax.set_xlabel('fiber diameter (nm)')
    ax.set_ylabel('minimum skin thickness (nm)')
    ax.legend(fontsize=5.8)
    tag(ax, 'b', -0.2)
    fig.tight_layout(w_pad=1.5)
    save(fig, 'figS11_pan_readout')


FIGURES = {'fig1': fig1, 'fig2': fig2, 'fig3': fig3, 'fig4': fig4, 'fig5': fig5, 'fig6': fig6, 'figS1': figS1,
           'figS2': figS2, 'figS3': figS3, 'figS4': figS4, 'figS5': figS5, 'figS6': figS6, 'figS7': figS7,
           'figS8': figS8, 'figS9': figS9, 'figS10': figS10, 'figS11': figS11}

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
