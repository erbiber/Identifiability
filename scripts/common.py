"""Shared helpers: CSV writing at six significant figures and the shell fraction of Eq. (8)."""
import csv
import os

import numpy as np

RES = 'results'


def write(name, header, rows):
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, name), 'w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(header)
        for row in rows:
            w.writerow([fmt(v) for v in row])
    print('wrote results/' + name, flush=True)


def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        return '%.6g' % v
    if isinstance(v, np.integer):
        return str(int(v))
    return v


def read(name, folder='data'):
    with open(os.path.join(folder, name)) as fh:
        return list(csv.DictReader(fh))


def shell_fraction(d, delta):
    """Eq. (8): area fraction of a shell of thickness delta in a cylinder of diameter d (1 when d < 2 delta)."""
    d = np.asarray(d, dtype=float)
    return np.where(d >= 2 * delta, 1.0 - (1.0 - 2.0 * delta / np.maximum(d, 1e-300)) ** 2, 1.0)


def thinnest_shell(d, f):
    """Inverse of Eq. (8): the shell thickness whose area fraction in a cylinder of diameter d is f (nan if f > 1)."""
    f = np.asarray(f, dtype=float)
    with np.errstate(invalid='ignore'):
        out = 0.5 * np.asarray(d, dtype=float) * (1.0 - np.sqrt(1.0 - np.clip(f, 0, None)))
    return np.where(f <= 1.0, out, np.nan)
