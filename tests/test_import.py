"""Check the PET per-fiber table against the supplied workbook by value and identifier, never by row position.
Run from the repository root: python tests/test_import.py"""
import csv
import re
import sys

import openpyxl

wb = openpyxl.load_workbook('data/raw/pellerin_compilation_of_data.xlsx', data_only=True)
tidy = list(csv.DictReader(open('data/pet_per_fiber.csv')))
used = list(csv.DictReader(open('data/laramee_lanthier_pellerin_2022_fig3b.csv')))
fails = []


def cells(ref):
    m = re.fullmatch(r"'(.+)'!([A-Z]+)(\d+):([A-Z]+)(\d+)", ref)
    ws = wb[m.group(1)]
    return ws['%s%s' % (m.group(2), m.group(3))].value, ws['%s%s' % (m.group(4), m.group(5))].value


def same(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


counts = {c: sum(1 for t in tidy if t['collector'] == c) for c in ('plate', 'gap', 'disk')}
if counts != {'plate': 56, 'gap': 105, 'disk': 79}:
    fails.append('counts %s' % counts)
if len({t['fiber_id'] for t in tidy}) != len(tidy):
    fails.append('fiber identifiers are not unique')
for t in tidy:
    p2, xc = cells(t['cells_P2_Xc'])
    if not (same(p2, t['P2']) and same(xc, t['X_c'])):
        fails.append('%s: P2/X_c differ from %s' % (t['fiber_id'], t['cells_P2_Xc']))
    if t['cells_diameter_P2']:
        d, p = cells(t['cells_diameter_P2'])
        if not (same(d, t['diameter_nm']) and abs(float(p) - float(t['P2'])) <= 5e-4):
            fails.append('%s: diameter/P2 differ from %s' % (t['fiber_id'], t['cells_diameter_P2']))
    if t['cells_diameter_FT'].startswith("'"):
        d, ft = cells(t['cells_diameter_FT'])
        if not (same(d, t['diameter_nm']) and same(ft, t['F_T'])):
            fails.append('%s: diameter/F_T differ from %s' % (t['fiber_id'], t['cells_diameter_FT']))
    if t['below_550nm_spot'] != ('yes' if t['diameter_nm'] and float(t['diameter_nm']) < 550 else 'no'):
        fails.append('%s: inclusion flag' % t['fiber_id'])
# the analysis input must hold exactly the same fibers, matched by value
key = lambda r: (r['collector'], round(float(r['P2']), 6), round(float(r['X_c']), 6), round(float(r['F_T']), 6),
                 r['diameter_nm'])
if sorted(map(key, tidy)) != sorted(map(key, used)):
    fails.append('analysis input and per-fiber table hold different fibers')
for f in fails:
    print('FAIL', f)
print('%d fibers checked against their source cells; %d failures' % (len(tidy), len(fails)))
sys.exit(1 if fails else 0)
