"""Check every value listed in claims.csv against the result files. Run from the repository root:
    python tests/test_claims.py        (or python -m pytest tests/test_claims.py)
Each row names a result file, a row filter, a column and an aggregate; the computed value must agree with the value
quoted in the article or Supplementary material to within half a unit of its last quoted digit (or the stated
tolerance). Filters: col==v, col!=v, col<=v, col>=v, col<v, col>v, joined by ';'. Aggregates: value (exactly one
row), min, max, median, count, count_true, and the same with the suffix _um (divided by 1000) or _pct (times 100)."""
import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPS = (('==', lambda a, b: a == b), ('!=', lambda a, b: a != b), ('<=', lambda a, b: float(a) <= float(b)),
       ('>=', lambda a, b: float(a) >= float(b)), ('<', lambda a, b: float(a) < float(b)),
       ('>', lambda a, b: float(a) > float(b)))


def same(a, b):
    try:
        return float(a) == float(b)
    except ValueError:
        return a == b


def select(rows, flt):
    out = rows
    for cond in [c for c in flt.split(';') if c.strip()]:
        for sym, fn in OPS:
            if sym in cond:
                col, val = cond.split(sym, 1)
                col, val = col.strip(), val.strip()
                if sym == '==':
                    out = [r for r in out if same(r[col], val)]
                elif sym == '!=':
                    out = [r for r in out if not same(r[col], val)]
                else:
                    out = [r for r in out if r[col] not in ('', 'nan') and fn(r[col], val)]
                break
        else:
            raise ValueError('bad filter ' + cond)
    return out


def compute(claim):
    with open(os.path.join(ROOT, 'results', claim['file'])) as fh:
        rows = select(list(csv.DictReader(fh)), claim['filter'])
    agg = claim['aggregate']
    scale = 1.0
    if agg.endswith('_um'):
        agg, scale = agg[:-3], 1e-3
    elif agg.endswith('_pct'):
        agg, scale = agg[:-4], 100.0
    if agg == 'count':
        return float(len(rows))
    if agg == 'count_true':
        return float(sum(r[claim['column']] == 'True' for r in rows))
    vals = [float(r[claim['column']]) for r in rows if r[claim['column']] not in ('', 'nan')]
    if agg == 'value':
        assert len(rows) == 1, '%s: %d rows match' % (claim['claim_id'], len(rows))
        return float(rows[0][claim['column']]) * scale
    assert vals, '%s: no rows match' % claim['claim_id']
    if agg == 'min':
        return min(vals) * scale
    if agg == 'max':
        return max(vals) * scale
    if agg == 'median':
        vals.sort()
        n = len(vals)
        return (vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])) * scale
    raise ValueError(agg)


def tolerance(claim):
    if claim.get('tolerance'):
        return float(claim['tolerance'])
    m = re.search(r'\.(\d+)$', claim['expected'])
    return 0.5 * 10 ** -(len(m.group(1)) if m else 0) + 1e-9


def check_all():
    with open(os.path.join(ROOT, 'claims.csv')) as fh:
        claims = list(csv.DictReader(fh))
    bad = []
    for c in claims:
        try:
            got = compute(c)
        except (AssertionError, FileNotFoundError, KeyError, ValueError) as e:
            bad.append((c['claim_id'], 'error: %s' % e))
            continue
        if abs(got - float(c['expected'])) > tolerance(c):
            bad.append((c['claim_id'], 'computed %.6g, quoted %s (%s)' % (got, c['expected'], c['paper_location'])))
    return claims, bad


def test_claims():
    claims, bad = check_all()
    assert not bad, '\n'.join('%s: %s' % b for b in bad)


if __name__ == '__main__':
    claims, bad = check_all()
    for b in bad:
        print('FAIL %s: %s' % b)
    print('%d of %d claims reproduce' % (len(claims) - len(bad), len(claims)))
    sys.exit(1 if bad else 0)
