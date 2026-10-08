import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

def _load(name):
    spec = importlib.util.spec_from_file_location(name, 'scripts/%s.py' % name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
sep = _load('separability')
trs = _load('theta_r_span')
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'xtick.direction': 'out', 'ytick.direction': 'out', 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'figure.dpi': 200, 'savefig.dpi': 400, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03})
INK, MID, LITE = ('#1a1a1a', '#6e6e6e', '#c8c8c8')
ACC, ACC2, ACC3 = ('#1f4e79', '#a33a1f', '#3d7a3d')
OUT = 'figures/'

def save(fig, name):
    fig.savefig(OUT + name + '.pdf')
    fig.savefig(OUT + name + '.png')
    plt.close(fig)
    print('  wrote', OUT + name)

def tag(ax, s):
    ax.text(-0.16, 1.04, s, transform=ax.transAxes, fontweight='bold', fontsize=9, va='bottom')

def drying(n=120, phi_star=0.2, k=5.0, tmax=400.0):
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
    T = np.geomspace(0.0001, tmax, 6000)
    C = solve_ivp(rhs, (0, tmax), np.full(n, 0.9), method='BDF', t_eval=T, rtol=1e-07, atol=1e-10).y
    tf = np.array([T[np.argmax(C[i] <= phi_star)] if np.any(C[i] <= phi_star) else np.nan for i in range(n)])
    lam = lambda c: np.exp(8 * (0.9 - c))
    th = np.array([np.trapezoid(1 / lam(C[i, T <= tf[i]]), T[T <= tf[i]]) if np.isfinite(tf[i]) else np.nan for i in range(n)])
    return (r, T, C, tf, th)

def figure1():
    r, T, C, tf, th = drying()
    fig, axs = plt.subplots(2, 2, figsize=(5.8, 4.6))
    axs = axs.ravel()
    ax = axs[0]
    for tt, col in zip((0.05, 1.0, 10.0, 60.0, 250.0), plt.cm.Blues(np.linspace(0.45, 0.95, 5))):
        j = np.argmin(abs(T - tt))
        ax.plot(r, C[:, j], color=col, lw=1)
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
    th_ok, r_ok = (th[ok], r[ok])
    fwd = lambda x: np.interp(x, r_ok, th_ok)
    inv = lambda y: np.interp(y, th_ok[::-1], r_ok[::-1])
    sec = ax.secondary_xaxis('top', functions=(fwd, inv))
    sec.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    sec.set_xlabel('$\\Theta_r$', fontsize=7)
    sec.tick_params(labelsize=6)
    ax.set_title('(c) Proposition 1: the same data on two axes', fontsize=7.5, pad=18)
    ax = axs[3]
    rr = np.linspace(0, 1, 200)
    bad = 1.2 * np.exp(-((rr - 0.45) / 0.3) ** 2) + 0.2
    ax.plot(rr, bad, color=ACC2, lw=1.4)
    lev = 1.0
    ax.axhline(lev, color=MID, lw=0.7, ls=':')
    xs = rr[np.where(np.diff(np.sign(bad - lev)))[0]]
    ax.plot(xs, [lev, lev], 'o', ms=3, color=ACC2)
    ax.text(0.86, 1.18, 'one clock value,\ntwo radii:\nno inverse', fontsize=6.5, ha='center', color=ACC2)
    ax.set_xlabel('$r/R$')
    ax.set_ylabel('$\\Theta_r$')
    ax.set_yticks([])
    ax.set_title('(d) non-monotone arrest: the equivalence fails', fontsize=7.5)
    fig.tight_layout(w_pad=1.6, h_pad=1.4)
    save(fig, 'fig1_clocks_degeneracy')

def figure2():
    phi = np.linspace(0.05, 0.22, 120)
    flat = lambda p, t: 2000000000.0 * np.sqrt(trs.FLAT * trs.D_s(p) * t)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.axvspan(0.14, 0.2, color=ACC, alpha=0.1, lw=0)
    ax.axvspan(0.07, 0.1, facecolor='none', hatch='////', edgecolor=LITE, lw=0)
    ax.plot(phi, [flat(p, 0.01) for p in phi], color=ACC, lw=1.3, label='10 ms flight')
    ax.plot(phi, [flat(p, 0.1) for p in phi], color=ACC, lw=1.3, ls='--', label='100 ms flight')
    pw = np.linspace(0.12, 0.21, 10)
    ax.plot(pw, [trs.joint_worst_flat_nm(p) for p in pw], color=ACC2, lw=1.3, label='joint worst case')
    ax.set_yscale('log')
    ax.set_ylim(3, 1500)
    ax.set_xlim(0.05, 0.22)
    ax.axhline(100, color=MID, lw=0.6, ls=':')
    ax.text(0.052, 112, 'single-fiber window begins', fontsize=6, color=MID)
    ax.text(0.17, 4.2, 'equilibrium range\n(Sec. 3.3)', ha='center', fontsize=6.5, color=ACC)
    ax.text(0.085, 4.2, 'Fox,\nexcluded', ha='center', fontsize=6.5, color=MID)
    ax.text(0.205, 17, 'fibers thinner than their\nflat diameter: arrest nearly\nsimultaneous across the radius', ha='right', fontsize=6.0, color=INK)
    ax.set_xlabel('arrest composition $\\varphi^*$ (solvent volume fraction)')
    ax.set_ylabel('flat diameter (nm)')
    ax.legend(fontsize=6.5, loc='upper left')
    save(fig, 'fig2_flat_diameter')

def figure3():
    fig = plt.figure(figsize=(7.0, 2.6))
    ax = fig.add_axes([0.0, 0.02, 0.5, 0.93])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis('off')
    ax.add_patch(Ellipse((50, 35), 96, 64, fc='#eef3f8', ec=ACC, lw=1))
    ax.text(50, 64, '$\\mathcal{F}=\\mathcal{G}$: any function of the clock = any function of radius', ha='center', fontsize=6.8)
    ax.text(50, 59.5, 'Proposition 1, Corollaries 1–2 (with A5, Lemma 2)', ha='center', fontsize=6.3, color=MID)
    ax.add_patch(Ellipse((36, 30), 46, 34, fc='#fbeee9', ec=ACC2, lw=1))
    ax.text(26, 38, '$\\mathcal{F}_K$: fading\nmemory', ha='center', fontsize=6.5, color=ACC2)
    ax.text(22, 22, 'stretching after\nsurface arrest,\nskin gives way:\ninterior peak', ha='center', fontsize=5.8, color=ACC2)
    ax.add_patch(Ellipse((58, 30), 40, 26, fc='#f3f3f3', ec=INK, lw=1, alpha=0.85))
    ax.text(65, 34, '$\\mathcal{G}_\\Theta$: shell of\nparametrized\nthickness', ha='center', fontsize=6.5)
    ax.text(47, 25, 'overlap:\nstretching ends\nbefore arrest', ha='center', fontsize=5.8, color=MID)
    ax.text(65, 13.5, '$\\mathcal{G}_\\Theta\\subseteq\\mathcal{F}$ (Prop. 2)', ha='center', fontsize=6.3)
    ax.text(1, 66, 'a', fontweight='bold', fontsize=9)
    ax2 = fig.add_axes([0.6, 0.2, 0.37, 0.68])
    offs = np.linspace(0, 0.12, 13)
    bf = [sep.committed_levels_bf(1.0 + o, 0.0) for o in offs]
    ax2.semilogy(offs, bf, 'o-', color=INK, ms=2.8, lw=1)
    ax2.axhline(1, color=MID, lw=0.7, ls=':')
    ax2.fill_between(offs, 1, 10000.0, color=ACC, alpha=0.07, lw=0)
    ax2.fill_between(offs, 0.01, 1, color=ACC2, alpha=0.07, lw=0)
    ax2.text(0.004, 2500.0, 'committed skin supported', fontsize=6.3, color=ACC)
    ax2.text(0.004, 0.02, 'committed skin refuted', fontsize=6.3, color=ACC2)
    ax2.set_ylim(0.01, 10000.0)
    ax2.set_xlabel('error in the committed plateau level')
    ax2.set_ylabel('Bayes factor, skin : flow')
    ax2.set_title('Corollary 3: refutation, not support', fontsize=7.5)
    ax2.text(-0.2, 1.04, 'b', transform=ax2.transAxes, fontweight='bold', fontsize=9)
    save(fig, 'fig3_commitment_scale')

def figure4():
    r = np.linspace(0.05, 1.0, 300)
    tf = 1.0 + 19.0 * (1 - r) ** 2 / np.max((1 - r) ** 2)
    before = lambda t: np.where(t < 0.8, 1.0, 0.0)
    ongoing = lambda t: np.exp(-0.5 * ((t - 5.0) / 5.0) ** 2)
    names = ('single exponential', 'two modes', 'linear ramp (not a mixture)')
    cols = (ACC, ACC3, ACC2)
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.0), sharey=True)
    titles = ('(a) stretching ends before\nsurface arrest (or skin halts it)', '(b) stretching continues,\nskin gives way', '(c) stretching continues,\nskin is cold-drawn')
    for k, ax in enumerate(axs):
        for name, col in zip(names, cols):
            K = sep.PROP4_KERNELS[name]
            prof = np.empty_like(r)
            for i, T in enumerate(tf):
                tt = np.linspace(0, T, 3000)
                if k == 0:
                    prof[i] = np.trapezoid(before(tt) * K(T - tt), tt)
                else:
                    prof[i] = np.trapezoid(ongoing(tt) * K(T - tt), tt)
                    if k == 2:
                        ss = np.linspace(T, 25.0, 3000)
                        prof[i] += np.trapezoid(ongoing(ss), ss)
            ax.plot(r, prof / prof.max(), color=col, lw=1.2, label=name)
            ax.plot(r[np.argmax(prof)], 1.0, 'v', color=col, ms=4)
        ax.set_title(titles[k], fontsize=7)
        ax.set_xlabel('radial position $r/R$')
    axs[0].set_ylabel('retained $\\mathcal{M}^*$ / max')
    axs[0].legend(fontsize=5.8, loc='lower left')
    fig.tight_layout(w_pad=0.8)
    save(fig, 'fig4_where_structure_peaks')

def figure5():
    d = np.linspace(*sep.WINDOW, 400)
    f = sep.shell_fraction(d, 30.0)
    p, _ = curve_fit(sep.exponential, d, f, p0=[f[0], 500.0, 0.0], maxfev=200000)
    g = sep.exponential(d, *p)
    xs, ys, _ = sep.crossings(30.0)
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.0))
    ax = axs[0]
    ax.plot(d, f, color=INK, lw=1.3, label='shell, $\\delta$ = 30 nm')
    ax.plot(d, g, color=ACC2, lw=1.1, ls='--', label='fitted exponential')
    ax.plot(xs, ys, 'o', ms=3, color=ACC)
    ax.set_xscale('log')
    ax.set_xlabel('diameter (nm)')
    ax.set_ylabel('structure')
    ax.legend(fontsize=6)
    ax.set_title('(a) forms that cross', fontsize=7.5)
    from scipy.optimize import brentq
    from scipy.special import expit
    h = lambda x: sep.shell_fraction(x, 30.0) - sep.exponential(x, *p)
    d0 = brentq(h, 300, 700)
    s0 = float(sep.shell_fraction(d0, 30.0))
    ax = axs[1]
    ax.step(d, (f >= s0).astype(float), where='mid', color=INK, lw=2.2, alpha=0.5, label='shell')
    ax.step(d, (g >= s0).astype(float), where='mid', color=ACC2, lw=1.0, ls='--', label='exponential')
    ax.set_xscale('log')
    ax.set_ylim(-0.1, 1.2)
    ax.set_yticks([0, 1])
    ax.set_xlabel('diameter (nm)')
    ax.set_ylabel('$g(S)=\\mathbf{1}[S\\geq s_0]$')
    ax.legend(fontsize=6, loc='center right')
    ax.set_title('(b) threshold at the crossing:\nidentical', fontsize=7.5)
    ax = axs[2]
    dd = np.linspace(*sep.WINDOW, 20001)
    ff = sep.shell_fraction(dd, 30.0)
    gg = sep.exponential(dd, *p)
    qq = sep.shell_fraction(1.3 * dd, 30.0)

    def rel(a, b, mapf):
        ga, gb = (mapf(a), mapf(b))
        return np.sqrt(np.trapezoid((ga - gb) ** 2, dd) / np.ptp(dd)) / np.ptp(ga)
    ks = np.logspace(0, 5, 26)
    cross = [rel(ff, gg, lambda z: expit(k * (z - s0))) for k in ks]
    nocross = [min((rel(ff, qq, lambda z: expit(k * (z - t))) for t in np.linspace(ff.min() + 0.005, ff.max() - 0.005, 80))) for k in ks]
    ax.loglog(ks, cross, color=ACC, lw=1.3, label='profiles that cross')
    ax.loglog(ks, nocross, color=MID, lw=1.1, ls='--', label='profiles that never cross')
    ax.set_xlabel('steepness $k$ of the map')
    ax.set_ylabel('separation / range of $g\\circ p_1$')
    ax.legend(fontsize=6, loc='lower left')
    ax.set_title('(c) relative separation\nunder steepening maps', fontsize=7.5)
    fig.tight_layout(w_pad=1.0)
    save(fig, 'fig5_functional_forms')

def figure6(L=2.7, rho=2.0):
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
    ax.set_title('stretching until arrest (Eq. 11):\nRoute 3 weak, Route 4 can decide', fontsize=7)
    for w in (100, 1000):
        v = float(np.interp(w, Wi, b1))
        ax.plot(w, v, 'o', ms=3, color=ACC2)
        ax.text(w * 1.1, v * 1.25, '%.2g' % v, fontsize=6)
    ax = axs[1]
    ax.loglog(x, b2, color=ACC, lw=1.4)
    ax.set_xlabel('relaxation before arrest, $\\Delta t/\\lambda$')
    ax.set_title('stretching complete, then relaxation\n(Eq. 12): Route 3 can decide', fontsize=7)
    for xv in (0.1, 1.0):
        v = float(np.interp(xv, x, b2))
        ax.plot(xv, v, 'o', ms=3, color=ACC)
        ax.text(xv * 1.1, v * 0.62, '%.2g' % v, fontsize=6)
    for ax, s in zip(axs, 'ab'):
        ax.text(-0.2, 1.04, s, transform=ax.transAxes, fontweight='bold', fontsize=9)
    fig.tight_layout(w_pad=1.0)
    save(fig, 'fig6_coupling_regimes')
COL = {'A': ('#e8eef6', ACC), 'L': ('#e7f2e7', ACC3), 'P': ('#fbeee9', ACC2), 'C': ('#f1ebf6', '#5b3f7a'), 'V': ('#f2f2f2', INK), 'S': ('#ffffff', INK)}

def figure7():
    """Map of the analysis: the conditions a measurement must meet, then the results at each level of commitment."""
    fig = plt.figure(figsize=(7.2, 9.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 137)
    ax.axis('off')

    def box(x, y, w, h, text, kind='V', fs=6.2, bold=False):
        fc, ec = COL[kind]
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.35,rounding_size=1.2', fc=fc, ec=ec, lw=0.9))
        ax.text(x, y, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', linespacing=1.15)

    def arrow(x1, y1, x2, y2, label=None, lx=0, ly=0, color=INK):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', lw=0.8, color=color, shrinkA=0, shrinkB=0, mutation_scale=7))
        if label:
            ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=6, style='italic', color=color, ha='center', va='center')

    box(34, 129, 60, 5.5, 'Structure measured against radius (one fiber) or diameter (across fibers)', 'S', 6.8, True)
    gates = [
        (119, 'A3 (Sec. 3.4): does the measured channel stay unchanged after local arrest?\npolystyrene: network-carried orientation, up to several to tens of µm;\nα-timescale channels, only below about 0.7–4.4 µm', 'A', 'no',
         'Collected state is not the\narrested state; post-arrest\nrelaxation can itself create a\ndiameter trend (Remark 2)'),
        (106, 'A2 (Sec. 2.3): is the readout one strictly monotone function of structure,\nunchanged across the radii and diameters compared?', 'A', 'no',
         'Revalidate the readout first,\ne.g. index-matched immersion\nfor polarized Raman below\nabout 500 nm'),
        (93.5, 'A1 (Sec. 3.1): is arrest radially ordered?  Lemma 1 (Sec. 3.2): yes, outward-first,\nfor a homogeneous binary solution drying by axisymmetric diffusion,\nwith outward evaporative flux of any time dependence', 'L', 'no',
         'Route 1: non-monotone\narrest breaks the equivalence\n(Corollary 5; no known\nrealization)'),
        (80.5, 'Is the fiber thicker than the flat diameter (Sec. 3.3)?\npolystyrene–chloroform: about 0.03–1.1 µm for 10–200 ms flights', 'S', 'no',
         'Below it, a skin of clock slope\n≤ Γ adds at most ~0.1Γ radial\ncontrast (Remark 1): a trend that\ncontinues here is not from it'),
    ]
    hgt = lambda q: 3.2 + 2.0 * q.count('\n')
    arrow(34, 126.2, 34, 119 + hgt(gates[0][1]) / 2 + 0.4)
    for i, (y, q, kind, lab, out) in enumerate(gates):
        box(34, y, 60, hgt(q), q, kind, 5.9)
        box(84, y, 26, 9.6, out, 'V', 5.6)
        arrow(64.4, y, 70.6, y, lab, ly=1.3)
        ny = gates[i + 1][0] + hgt(gates[i + 1][1]) / 2 + 0.4 if i + 1 < len(gates) else 73.0
        arrow(34, y - hgt(q) / 2 - 0.4, 34, ny, 'yes', lx=2.4)
    box(50, 69.6, 72, 5.0, 'Which account, and how much does it commit to?', 'S', 7, True)
    for x in (12, 36, 64, 90):
        arrow(50, 66.8, x, 62.6)
    box(12, 59.4, 21, 5.6, 'flow history or skin,\nunrestricted', 'S', 6.2, True)
    box(36, 59.4, 21, 5.6, 'skin of parametrized\nthickness', 'S', 6.2, True)
    box(64, 59.4, 25, 5.6, 'flow history committed\nto a fading memory', 'S', 6.2, True)
    box(90, 59.4, 18, 5.6, 'flow-induced\nphase separation', 'S', 6.2, True)

    arrow(12, 56.5, 12, 52.6)
    box(12, 48.3, 21, 7.6, 'Proposition 1 (A1–A3):\nidentical profile families;\nCorollary 1: every functional\nof one fiber\'s profile', 'P', 5.6)
    arrow(12, 44.3, 12, 40.9)
    box(12, 36.8, 21, 7.6, 'Corollary 2 (A5, Lemma 2):\none function of both histories\nreproduces cross-fiber trends;\nweakens when jets split', 'C', 5.6)
    arrow(12, 32.8, 12, 29.6)
    box(12, 26.0, 21, 6.2, 'observationally equivalent:\nnot identifiable in this\nmeasurement class', 'V', 5.6, True)

    arrow(36, 56.5, 36, 52.6)
    box(36, 48.3, 21, 7.6, 'Proposition 2: the parametrized\nskin family is contained in\nthe unrestricted flow-history\nfamily, Gα ⊂ F', 'P', 5.6)
    arrow(36, 44.3, 36, 40.9)
    box(36, 36.8, 21, 7.6, 'Corollary 3: data can refute\nthe committed skin model and\nstay compatible with the\nunrestricted flow family', 'C', 5.6)
    arrow(36, 32.8, 36, 29.6)
    box(36, 26.0, 21, 6.2, 'asymmetric\nrefutability', 'V', 6.0, True)

    arrow(64, 56.5, 64, 53.4)
    box(64, 50.4, 25, 5.6, 'Proposition 3 (any non-increasing kernel):\ndoes stretching continue after surface arrest?', 'P', 5.4)
    arrow(58, 47.4, 56, 43.6, 'no', lx=-2.2)
    arrow(70, 47.4, 73, 43.6, 'yes', lx=2.2)
    box(55.5, 40.2, 14, 6.2, '(a) maximum at the\nsurface; matches a\nskin to 0.02–0.07', 'V', 5.2)
    box(75, 40.2, 19, 5.6, 'how does the arrested\nskin respond? (Sec. 4.4)', 'P', 5.5)
    arrow(69, 37.2, 60, 33.6, 'halts', lx=-2.4, ly=0.2)
    arrow(75, 37.2, 75, 33.6, 'drawn', lx=3.0)
    arrow(81, 37.2, 89, 33.6, 'decouples', lx=4.6, ly=1.0)
    box(58, 30.4, 12, 5.4, 'stretching\nends: case (a)', 'V', 5.3)
    box(74.5, 30.4, 12, 5.4, '(c) maximum\nat the surface', 'V', 5.3)
    box(90.5, 29.6, 15, 7.4, '(b) interior maximum,\nif new stretching\noutpaces relaxation;\nProposition 4 for a\ncomposition-dependent\nclock', 'V', 5.0)
    ax.text(74.5, 24.6, 'Remark 4: Routes 3 and 4\nmay not discriminate;\nadd a second channel', fontsize=4.9, ha='center', va='center', color=ACC2, linespacing=1.1)
    arrow(90.5, 25.5, 90.5, 22.4)
    box(84, 18.9, 26, 5.6, 'Corollary 4: the radial position of the\nmaximum is the robust discriminant →\nRoute 4, if decoupling is shown independently', 'C', 5.1)

    arrow(90, 56.5, 90, 53.1)
    box(90, 50.2, 18, 5.4, 'Proposition 5 (A4):\nis ψ = h(S)?', 'P', 5.6)
    ax.text(91.2, 46.0, 'h invertible: equivalent\nthrough one channel', fontsize=4.9, ha='center', va='center', color=INK, linespacing=1.1)
    ax.text(91.2, 42.7, 'h not invertible:\nseparable in principle', fontsize=4.9, ha='center', va='center', color=INK, linespacing=1.1)
    ax.text(91.2, 39.4, 'ψ still varies where S\nsaturates: Route 2', fontsize=4.9, ha='center', va='center', color=ACC2, linespacing=1.1)

    box(24, 17.2, 44, 6.4, 'Proposition 6: when two monotone diameter profiles cross at an interior\ndiameter, fitted functional forms cannot decide under a monotone\nreadout of unknown form', 'P', 5.7)
    box(50, 6.6, 96, 10.4, 'Corollary 5 (Sec. 5.1; Table 5): exits from the equivalence.  Route 1, non-monotone radial arrest (no known realization).  Route 2, an independent\nstructural channel (all three accounts).  Route 3, a relaxation-time perturbation at matched conformation: coupling β ≈ 0.005–0.05 while stretching\ncontinues to arrest, 0.19–1.5 if the chains relax before arrest (Eqs. 11–12).  Route 4, the radial position of the maximum (stretching after surface\narrest and a decoupled skin, shown independently).  Jet imaging selects between Routes 3 and 4 (Sec. 5.3; Table 6).  Each route decides\nonly if the accounts make incompatible predictions for what it measures.', 'C', 5.4)
    arrow(24, 14.0, 24, 11.8)
    arrow(84, 16.1, 84, 11.8)
    for i, (k, lab) in enumerate((('A', 'assumption'), ('L', 'lemma'), ('P', 'proposition'), ('C', 'corollary'), ('V', 'outcome'))):
        fc, ec = COL[k]
        ax.add_patch(FancyBboxPatch((6 + i * 18, 134.6), 2.6, 1.5, boxstyle='round,pad=0.1', fc=fc, ec=ec, lw=0.8))
        ax.text(9.8 + i * 18, 135.35, lab, fontsize=6.3, va='center')
    save(fig, 'fig7_decision_tree')


def figure8():
    import csv
    _net = [float(r['flat_diameter_nm']) for r in csv.DictReader(open('results/flat_diameter_consistent.csv')) if r['mode'] == 'network']
    flat_lo, flat_hi = (min(_net) / 1000.0, max(_net) / 1000.0)
    a3_lo = trs.a3_upper_diameter_nm(trs.TAU_ALPHA_TG, 0.14, 0.25) / 1000.0
    a3_hi = trs.a3_upper_diameter_nm(trs.TAU_ALPHA_TG, 0.2, 0.5) / 1000.0
    rows = [('published in-air record,\nrising part (onset ≤ 2.5 µm)', (0.5, 2.5), None, INK), ('A2: polarized Raman in air', (0.5, 10), None, ACC), ('A2: with index-matched immersion\n(floor ~0.14 µm, if reached)', (0.14, 10), None, ACC), ('A1 has content (above the flat\ndiameter, network orientation)', (flat_hi, 10), (flat_lo, flat_hi), ACC3), ('A3: network-carried orientation', (0.03, 10), None, ACC3), ('A3: alpha-timescale channels', (0.03, a3_lo), (a3_lo, a3_hi), ACC3), ('below-flat-diameter test:\nreadout floor to flat diameter', (0.14, flat_hi), None, ACC2)]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    for i, (lab, solid, unc, col) in enumerate(rows[::-1]):
        ax.barh(i, solid[1] - solid[0], left=solid[0], height=0.55, color=col, alpha=0.75, lw=0)
        if unc:
            ax.barh(i, unc[1] - unc[0], left=unc[0], height=0.55, color=col, alpha=0.22, lw=0, hatch='///')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=6.4)
    ax.set_xscale('log')
    ax.set_xlim(0.03, 10)
    ax.set_xlabel('fiber diameter (µm)')
    ax.axvspan(0.5, 2.5, color=INK, alpha=0.05, lw=0)
    ax.text(1.12, len(rows) - 0.35, 'window where the record\nis interpretable', ha='center', fontsize=6, color=MID)
    ax.text(0.031, -0.95, 'hatched: edge set by transferred inputs (diffusivity, flight time, relaxation time)', fontsize=5.6, color=MID)
    ax.set_ylim(-1.2, len(rows) - 0.1)
    save(fig, 'fig8_case_study_windows')


def figure9():
    dl = np.geomspace(5.0, 150.0, 400)
    phi = sep.shell_fraction(sep.PET_SPOT_NM, dl)
    allow, typical = sep.pet_skin_bound(dl), sep.pet_skin_bound(dl, sep.PET_MEDIAN)
    need_gap, need_disk = sep.PET_GAP / phi, sep.PET_DISK / phi
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.axhspan(1.0, 20.0, color=LITE, alpha=0.45, lw=0)
    ax.text(30.0, 3.0, 'beyond perfect\nalignment', color=MID, fontsize=6.5, va='center')
    ax.fill_between(dl, allow, 1.0, color=ACC2, alpha=0.10, lw=0)
    ax.plot(dl, need_disk, color=ACC2, lw=1.1, ls='--', label='needed for the disk median, 0.24')
    ax.plot(dl, need_gap, color=ACC2, lw=1.1, label='needed for the gap median, 0.18')
    ax.plot(dl, allow, color=ACC, lw=1.4, label='allowed: no plate fiber above 0.04')
    ax.plot(dl, typical, color=ACC, lw=1.0, ls='--', label='allowed: plate median, 0.02')
    for d0 in (13.1, 30.0, 60.0):
        v = float(sep.pet_skin_bound(d0))
        ax.plot([d0], [v], 'o', color=ACC, ms=3.2, zorder=5)
        ax.annotate('%.2f' % v, (d0, v), xytext=(5, -13), textcoords='offset points', fontsize=6.5, color=ACC)
    ax.text(42.0, 0.33, 'excluded by\nthe plate', color=ACC2, fontsize=7, ha='center', va='center')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(5.0, 150.0); ax.set_ylim(0.01, 8.0)
    ax.set_xticks([5, 10, 20, 50, 100]); ax.set_xticklabels(['5', '10', '20', '50', '100'])
    ax.set_xlabel('skin thickness $\\delta$ (nm)')
    ax.set_ylabel('orientation carried by the skin itself, $s_{\\mathrm{skin}}$')
    ax.legend(loc='lower left', fontsize=6.2, handlelength=2.2)
    save(fig, 'fig9_pet_plate_bound')

for f in (figure1, figure2, figure3, figure4, figure5, figure6, figure7, figure8, figure9):
    f()
