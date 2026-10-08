"""Table of Contents graphic: the electrospinning setup and, magnified, the chains inside two fibers.

The fibers show the paper's main results. A maximum of alignment at the surface fits both accounts, so a resolved
profile does not identify the mechanism (Propositions 1-2). A maximum inside the fiber excludes a committed skin model,
which peaks at the surface (Propositions 3-4, Corollary 3). Below the flat diameter a skin of bounded slope produces at most
about 0.1 of its slope as radial contrast, so orientation that still rises with thinning there is not from the skin (Remark 1).

Schematic, not computed from the results. Drawn at the ACS size, 3.25 in x 1.75 in; one axis unit is 0.01 in.
The random numbers that place the chains are seeded, so the drawing is identical on every run.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle

plt.rcParams.update({'font.family': 'serif', 'font.size': 7, 'savefig.dpi': 600})
INK, MID, LITE = ('#1a1a1a', '#6e6e6e', '#c8c8c8')
ACC, ACC2 = ('#1f4e79', '#a33a1f')
TINT = LinearSegmentedColormap.from_list('tint', ['#fbf6f3', '#a9c6e3'])
OUT = 'figures/'
X0, X1 = 78.0, 140.0  # the magnified fibers: left end, right end


def surface_peak(u):
    """Alignment against depth u (0 on the axis, 1 at the surface): highest at the surface."""
    return u ** 3


def interior_peak(u):
    """Alignment highest in a buried band, with a weakly aligned surface."""
    return np.minimum(0.95, 0.22 * u + 0.9 * np.exp(-((u - 0.57) / 0.17) ** 2))


def chain(rng, x, y, p, half):
    """One chain: a coil where the alignment p is low, drawn out along the fiber axis where it is high."""
    steps, step = (30, 1.55) if p > 0.5 else (30, 0.95)
    angle = rng.normal(0.0, 0.2) if p > 0.5 else rng.uniform(-np.pi, np.pi)
    xs, ys = [0.0], [0.0]
    for _ in range(steps):
        angle = (1.0 - 0.8 * p) * angle + rng.normal(0.0, 0.85 * (1.0 - p) ** 1.5 + 0.06)
        xs.append(xs[-1] + step * np.cos(angle))
        ys.append(ys[-1] + step * np.sin(angle) * (1.0 - 0.6 * p))
    xs, ys = np.array(xs), np.array(ys)
    lim = (0.13 if p > 0.5 else 0.30) * half  # each chain stays within its own layer
    return x + xs - xs.mean(), y + lim * np.tanh((ys - ys.mean()) / lim)


def setup(ax):
    """Syringe and needle, straight jet with a magnified view of its partly oriented chains, whipping coil, collector."""
    xj = 28.0
    ax.add_patch(Rectangle((xj - 6, 138), 12, 28, facecolor='#eef3f8', edgecolor=INK, lw=0.6))
    ax.add_patch(Rectangle((xj - 6, 138), 12, 17, facecolor='#c9d9ea', edgecolor='none'))
    ax.plot([xj - 9, xj + 9], [170, 170], color=INK, lw=0.9)
    ax.plot([xj, xj], [166, 170], color=INK, lw=0.9)
    ax.add_patch(Rectangle((xj - 1.1, 126), 2.2, 12, facecolor=LITE, edgecolor=INK, lw=0.5))
    ax.add_patch(Polygon([(xj - 1.6, 126), (xj + 1.6, 126), (xj, 121)], facecolor=ACC, edgecolor='none'))
    ax.plot([xj, xj], [121, 106], color=ACC, lw=0.7)
    th = np.linspace(0.0, 7.5 * 2 * np.pi, 1400)
    amp = 21.0 * (th / th[-1]) ** 0.85
    ys = 106.0 - 62.0 * th / th[-1] + 0.20 * amp * np.sin(th)
    ax.plot(xj + amp * np.cos(th), ys, color=ACC, lw=0.5)
    ax.add_patch(Rectangle((3, 30), 50, 3.2, facecolor=LITE, edgecolor=INK, lw=0.6))
    ax.plot([xj, xj], [30, 24], color=INK, lw=0.6)
    for k, w in enumerate((7, 4.5, 2)):
        ax.plot([xj - w, xj + w], [24 - 2.2 * k, 24 - 2.2 * k], color=INK, lw=0.6)
    ax.plot([xj + 6, xj + 15, xj + 15], [152, 152, 146], color=INK, lw=0.5)
    ax.text(xj + 15, 142, 'kV', ha='center', va='center', fontsize=6, color=INK)
    ax.text(xj - 3, 110, 'jet', ha='right', va='center', fontsize=6, color=ACC)
    jet_inset(ax, np.random.default_rng(3), xj, 114.0)
    ax.text(3, 12, 'collector', ha='left', va='center', fontsize=6, color=INK)
    return xj + amp[-1] * np.cos(th[-140]), ys[-140]


def jet_inset(ax, rng, xj, yj, cx=49.0, cy=112.0, rad=8.5):
    """Magnified view of the jet just below the nozzle: chains partly drawn out along the (vertical) jet axis."""
    disk = Circle((cx, cy), rad, facecolor='#f3f6fa', edgecolor=INK, lw=0.5, zorder=6)
    ax.add_patch(disk)
    ax.plot([xj + 0.8, cx - rad], [yj, cy + 1.5], color=MID, lw=0.4, zorder=5)
    for x0 in (-5.0, -1.7, 1.7, 5.0):
        ang = np.pi / 2 + rng.normal(0.0, 0.3)
        xs, ys = [cx + x0], [cy - 7.0 + rng.uniform(0, 2.5)]
        for _ in range(24):
            ang = 0.75 * ang + 0.25 * (np.pi / 2) + rng.normal(0.0, 0.55)
            xs.append(xs[-1] + 0.62 * np.cos(ang))
            ys.append(ys[-1] + 0.62 * np.sin(ang))
        line, = ax.plot(xs, ys, color='#6a4c93', lw=0.5, solid_capstyle='round', zorder=7)
        line.set_clip_path(disk)
    ax.text(cx + 1, cy + rad + 5.0, 'partially\noriented', ha='center', va='center', linespacing=0.9, fontsize=4.6, color='#6a4c93')


def core_shell(u):
    """Axial alignment highest in the core, with a weakly aligned shell: a maximum inside the fiber."""
    u = np.asarray(u, dtype=float)
    return 0.05 + 0.9 / (1.0 + np.exp((u - 0.58) / 0.05))


def fiber(ax, rng, yc, order, half, layers, radial_shell=False):
    """Magnified longitudinal section of one fiber, shaded and filled with chains according to order(depth)."""
    body = Rectangle((X0, yc - half), X1 - X0, 2 * half, facecolor='none', edgecolor='none')
    ax.add_patch(body)
    ys = np.linspace(yc - half, yc + half, 240)
    ax.imshow(order(np.abs(ys - yc) / half)[:, None], extent=(X0, X1, yc - half, yc + half), origin='lower',
              cmap=TINT, vmin=0.0, vmax=1.0, aspect='auto', zorder=1)
    for u in layers:
        p = float(order(abs(u)))
        for x in np.arange(X0 - 12, X1 + 24, 30 if p > 0.5 else 10) + rng.uniform(-5, 5):
            cx, cy = chain(rng, x + rng.uniform(-3, 3), yc + u * half, p, half)
            line, = ax.plot(cx, cy, color=ACC if p > 0.5 else ACC2, lw=0.7 if p > 0.5 else 0.4,
                            alpha=1.0 if p > 0.5 else 0.8, solid_capstyle='round', zorder=2)
            line.set_clip_path(body)
    if radial_shell:  # shell chains lie across the fiber, from the core boundary out to the surface
        for side in (-1.0, 1.0):
            for x in np.arange(X0 + 1.5, X1 + 2, 3.2) + rng.uniform(-0.6, 0.6):
                ys_ = yc + side * np.linspace(0.64, 0.98, 9) * half
                xs_ = x + rng.normal(0.0, 0.35, ys_.size).cumsum() * 0.5
                line, = ax.plot(xs_, ys_, color='#6a4c93', lw=0.55, solid_capstyle='round', zorder=2)
                line.set_clip_path(body)
    for y in (yc - half, yc + half):
        ax.plot([X0, X1], [y, y], color=INK, lw=0.8, zorder=3)
    ax.plot([X0, X0], [yc - half, yc + half], color=MID, lw=0.5, ls=(0, (2, 2)), zorder=3)
    for k, f in enumerate(np.linspace(1.0, 0.12, 12)):  # end face: the same pattern as concentric rings
        ax.add_patch(Ellipse((X1, yc), 0.8 * half * f, 2 * half * f, facecolor=TINT(float(order(f - 0.04))), edgecolor='none', zorder=4 + k))
    ax.add_patch(Ellipse((X1, yc), 0.8 * half, 2 * half, facecolor='none', edgecolor=INK, lw=0.8, zorder=20))


def mark(ax, x, y, fits):
    """A round badge: a check where the account can explain the fiber, a cross where it cannot."""
    ax.add_patch(Circle((x, y), 6.2, facecolor=ACC if fits else ACC2, edgecolor='none', zorder=6))
    kw = dict(color='white', lw=1.2, solid_capstyle='round', solid_joinstyle='round', zorder=7)
    if fits:
        ax.plot([x - 3.1, x - 0.9, x + 3.3], [y - 0.2, y - 2.6, y + 2.8], **kw)
    else:
        ax.plot([x - 2.5, x + 2.5], [y - 2.5, y + 2.5], **kw)
        ax.plot([x - 2.5, x + 2.5], [y + 2.5, y - 2.5], **kw)


def level(value):
    """Alignment the same at every depth: below the flat diameter drying sets up almost no radial contrast."""
    return lambda u: value + 0.0 * np.asarray(u, dtype=float)


def toc():
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(3.25, 1.75))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 325)
    ax.set_ylim(0, 175)
    ax.set_aspect('equal')
    ax.axis('off')
    rows = ((122.0, 10.0, surface_peak, (-0.82, -0.42, 0.0, 0.42, 0.82), 'maximum at the surface'),
            (89.0, 10.0, core_shell, (-0.82, -0.45, -0.15, 0.15, 0.45, 0.82), 'maximum inside the fiber'))
    thin = ((65.0, 3.4, level(0.6), 'less aligned'), (55.5, 2.2, level(0.95), 'more aligned'))
    zx, zy = setup(ax)
    top, bot = rows[0][0] + rows[0][1], thin[1][0] - thin[1][1]
    ax.add_patch(Circle((zx, zy), 4.0, facecolor='none', edgecolor=INK, lw=0.6, zorder=6))
    ax.plot([zx + 2.2, X0 - 5], [zy + 3.3, top], color=MID, lw=0.4)
    ax.plot([zx + 3.6, X0 - 5], [zy - 1.8, bot], color=MID, lw=0.4)
    ax.plot([X0 - 3, X0 - 5, X0 - 5, X0 - 3], [top, top, bot, bot], color=MID, lw=0.4)
    ax.text(196, 167, 'Which mechanism aligned the chains?', ha='center', va='center', fontsize=8, fontweight='bold', color=INK)
    c1, c2, tx = 196.0, 228.0, 243.0  # the two accounts as columns; the verdict beside them
    for x, top_word, bottom in ((c1, 'Flow', 'history'), (c2, 'Drying', 'skin')):
        ax.text(x, 152, top_word, ha='center', va='baseline', fontsize=5.8, fontweight='bold', color=INK)
        ax.text(x, 145, bottom, ha='center', va='baseline', fontsize=5.8, fontweight='bold', color=INK)

    def verdict(yc, f1, f2, v1, v2, col):
        mark(ax, c1, yc, f1)
        mark(ax, c2, yc, f2)
        ax.text(tx, yc + 4.2, v1, ha='left', va='center', fontsize=6.2, fontweight='bold', color=col)
        ax.text(tx, yc - 4.6, v2, ha='left', va='center', fontsize=6.2, fontweight='bold', color=col)

    for (yc, half, order, layers, label) in rows:
        fiber(ax, rng, yc, order, half, layers)
        ax.text(X0, yc + half + 4.6, label, ha='left', va='center', fontsize=5.8, color=ACC)
    verdict(rows[0][0], True, True, 'same profile:', 'not identifiable', INK)
    verdict(rows[1][0], True, False, 'committed skin', 'model excluded', ACC2)
    # below the flat diameter: two thinner fibers, the thinner one more aligned
    ax.text(X0, thin[0][0] + thin[0][1] + 4.6, 'below the flat diameter', ha='left', va='center', fontsize=5.8, color=ACC)
    for yc, half, order, note in thin:
        fiber(ax, rng, yc, order, half, (0.0,))
        ax.text(X1 + 3.5, yc, note, ha='left', va='center', fontsize=4.2, color=MID)
    verdict(0.5 * (thin[0][0] + thin[1][0]), True, False, 'skin cannot', 'explain the rise', ACC2)
    ax.plot([X0, 318], [41, 41], color=LITE, lw=0.5)
    ax.text(198, 31, 'Resolving structure alone does not identify its origin;', ha='center', va='center', fontsize=5.7, color=INK)
    ax.text(198, 19.5, 'peak position and trends below the flat diameter can', ha='center', va='center', fontsize=5.7, fontweight='bold', color=ACC)
    fig.savefig(OUT + 'toc_graphic.pdf')
    fig.savefig(OUT + 'toc_graphic.png')
    plt.close(fig)
    print('  wrote', OUT + 'toc_graphic.pdf/.png')


toc()
