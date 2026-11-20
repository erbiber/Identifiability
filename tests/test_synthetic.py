"""Regression test: the values of Section 5.4 of the article and Section S10 of the Supplementary material
against the simulation results written by scripts/synthetic_validation.py. Run from the repository root:
    python -m pytest tests/test_synthetic.py
Each quoted percentage is checked to within half a unit of its last quoted digit."""
import csv
import os

import pytest

RES = 'results'


def rows(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        pytest.skip(f'{path} missing; run python scripts/synthetic_validation.py first')
    with open(path) as f:
        return list(csv.DictReader(f))


def pick(rs, col, **kw):
    hit = [r for r in rs if all(str(r[k]) == str(v) for k, v in kw.items())]
    assert len(hit) == 1, (kw, len(hit))
    return float(hit[0][col])


def pct(x, quoted):
    assert abs(100 * x - quoted) <= 0.5 + 1e-9, (100 * x, quoted)


def test_plate_control():
    r = rows('synth_plate_control.csv')
    base = dict(flight_share='0.3', icc='0.1')
    pct(pick(r, 'coverage_95', **base, sessions='4', fibers_per_collector='32', resampling='fiber'), 86)
    pct(pick(r, 'coverage_95', **base, sessions='4', fibers_per_collector='64', resampling='fiber'), 78)
    pct(pick(r, 'coverage_95', **base, sessions='2', fibers_per_collector='64', resampling='fiber'), 65)
    pct(pick(r, 'p_conclude_collector_majority', **base, sessions='8', fibers_per_collector='64', resampling='session'), 77)
    pct(pick(r, 'p_conclude_collector_majority', **base, sessions='2', fibers_per_collector='64', resampling='session'), 28)
    sess = [float(x['coverage_95']) for x in r if x['resampling'] == 'session']
    assert 0.855 <= min(sess) and max(sess) < 0.965                       # 86-96% over all settings
    at30 = [float(x['coverage_95']) for x in r if x['resampling'] == 'session' and x['flight_share'] == '0.3']
    assert 0.905 <= min(at30) and max(at30) < 0.955                       # 91-95% at a share of 30%


def test_route3():
    r = rows('synth_route3.csv')
    b = dict(design='blocked', sessions='8', icc='0.1', truth='flow history', readout_offset='0')
    pct(pick(r, 'p_detect_shift', **b, true_shift='0.14', fibers_per_solution='512'), 78)
    pct(pick(r, 'p_detect_shift', **b, true_shift='0.14', fibers_per_solution='1024'), 97)
    pct(pick(r, 'p_detect_shift', **b, true_shift='0.1', fibers_per_solution='512'), 51)
    pct(pick(r, 'p_detect_shift', **b, true_shift='0.1', fibers_per_solution='1024'), 82)
    sk = dict(design='blocked', sessions='8', icc='0.1', truth='skin', true_shift='0')
    pct(pick(r, 'p_detect_shift', **sk, readout_offset='0.01', fibers_per_solution='512'), 28)
    pct(pick(r, 'p_detect_shift', **sk, readout_offset='0.01', fibers_per_solution='1024'), 51)
    sep = [float(x['p_detect_shift']) for x in r if x['design'] == 'separate' and x['icc'] == '0.1'
           and x['truth'] == 'flow history' and x['true_shift'] == '0.14']
    assert max(sep) <= 0.315                                              # at most 31%


def test_route4():
    r = rows('synth_route4.csv')
    deep = [float(x['p_detect_interior_given_fading_memory']) for x in r if x['mean_P2'] == '0.25'
            and x['pixel_noise'] == '0.03' and x['phi'] in ('0.15', '0.3') and x['psf_sigma_nm'] in ('20', '50')]
    assert 0.635 <= min(deep) and max(deep) <= 1.0                        # 64-100%
    pct(pick(r, 'p_detect_interior_given_fading_memory', lambda_over_ts='1', phi='0.15', mean_P2='0.75',
             pixel_noise='0.03', psf_sigma_nm='20'), 20)
    assert max(float(x['p_false_interior_given_skin']) for x in r) <= 0.01   # at most 1%
    assert max(float(x['p_detect_interior_given_fading_memory']) for x in r if x['psf_sigma_nm'] == '150') <= 0.05


def test_robustness():
    r = rows('synth_robustness.csv')
    pc = [x for x in r if x['test'] == 'plate control']
    cov = [float(x['coverage_95']) for x in pc]
    assert 0.915 <= min(cov) and max(cov) < 0.955                         # 92-95%
    maj = {(x['setting'], x['case']): float(x['p_collector_majority']) for x in pc}
    pct(maj[('8 sessions', 'baseline')], 77)
    pct(maj[('8 sessions', 'readout error 0.09')], 63)
    pct(maj[('4 sessions', 'baseline')], 58)
    pct(maj[('4 sessions', 'readout error 0.09')], 44)
    r4 = {(x['setting'], x['case']): x for x in r if x['test'] == 'Route 4'}
    pct(float(r4[('depth 190 nm; PSF 20 nm; noise 0.03', 'sectioning damage 0.20')]['p_false_interior_given_skin']), 82)
    pct(float(r4[('depth 190 nm; PSF 50 nm; noise 0.03', 'sectioning damage 0.20')]['p_false_interior_given_skin']), 2)
    det = [float(x['p_detect_interior_given_fading_memory']) for x in r4.values()]
    assert min(det) >= 0.925                                              # 93-100%


if __name__ == '__main__':
    # also runnable as a plain script, like the other tests: python tests/test_synthetic.py
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'{name}: ok')
