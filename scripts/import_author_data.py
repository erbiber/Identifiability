"""Convert the workbook supplied by C. Pellerin (data/raw/pellerin_compilation_of_data.xlsx) into the CSV files that
the analysis reads. Run from the repository root: python scripts/import_author_data.py
The PET fibers are linked across Figs. S2, S3 and the inset of Fig. 3b of the source by their orientation; the trans
fraction of the four fibers that have no diameter (and so are absent from Fig. S3) is taken from the vector extraction
of Fig. 3b in data/digitized/laramee_2022_fig3b_main.csv. The 240 orientation-crystallinity pairs of the sheet
'AWL 2022 Fig3 inset' (56 plate, 105 gap, 79 disk) define the fibers; the other sheets are matched to them by value,
because four records lack diameter or trans entries and several gap records are reordered or rounded, so row-wise
alignment would create spurious discrepancies. data/pet_per_fiber.csv is the tidy per-fiber table, with the source
cells of every value; spinning sessions are not reported in the source. Two plate fibers share the orientation 0.00118 (669 and
972 nm); their crystallinities (0.0484 and 0.0453) are assigned in row order, which affects no result, because no
analysis uses diameter and crystallinity of the plate fibers together."""
import csv
import os

import openpyxl

RAW = 'data/raw/pellerin_compilation_of_data.xlsx'


def rows(ws, start):
    return list(ws.iter_rows(values_only=True))[start:]


def column_pair(ws, i, j, start=1):
    return [(r[i], r[j]) for r in rows(ws, start) if r[i] is not None and r[j] is not None]


def column_pair_cells(ws, i, j, start=1):
    """As column_pair, with the worksheet cell range each pair came from (1-based Excel row numbers)."""
    from openpyxl.utils import get_column_letter
    out = []
    for k, r in enumerate(rows(ws, start)):
        if r[i] is not None and r[j] is not None:
            row = start + k + 1
            out.append(((r[i], r[j]), "'%s'!%s%d:%s%d" % (ws.title, get_column_letter(i + 1), row,
                                                          get_column_letter(j + 1), row)))
    return out


def write(name, header, data):
    with open(os.path.join('data', name), 'w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(header)
        w.writerows(data)
    print('wrote data/' + name, len(data), 'rows')


def link(values, pool, tol=5e-4):
    """Index of the pool entry matching each value, each pool entry used once (nearest first)."""
    used, out = set(), []
    for v in values:
        best = min((abs(p - v), k) for k, p in enumerate(pool) if k not in used) if len(used) < len(pool) else (1, None)
        if best[0] <= tol:
            used.add(best[1])
            out.append(best[1])
        else:
            out.append(None)
    return out


def main():
    wb = openpyxl.load_workbook(RAW, data_only=True)
    # polystyrene, Richard-Lacroix and Pellerin 2015, Figs. 2A and 5A (bin means and error-bar half-widths)
    write('richard_lacroix_pellerin_2015_fig2a.csv', ['diameter_nm', 'P2', 'half_width'],
          [r[:3] for r in rows(wb['MRL 2015 Fig2A'], 1) if r[0] is not None])
    write('richard_lacroix_pellerin_2015_fig5a.csv', ['diameter_nm', 'EID_percent', 'half_width'],
          [r[:3] for r in rows(wb['MRL 2015 Fig5A'], 1) if r[0] is not None])

    # PET, Laramée, Lanthier and Pellerin 2022
    dig = list(csv.DictReader(open('data/digitized/laramee_2022_fig3b_main.csv')))
    out, tidy = [], []
    for name, (a, b) in (('disk', (0, 1)), ('plate', (3, 4)), ('gap', (6, 7))):
        s2c = column_pair_cells(wb['AWL 2022 FigS2'], a, b)     # (diameter, P2)
        s3c = column_pair_cells(wb['AWL 2022 FigS3'], a, b)     # (diameter, F_T), same fibers and order as S2
        insc = column_pair_cells(wb['AWL 2022 Fig3 inset'], a, b)  # (P2, X_c): the 240 orientation-crystallinity pairs
        s2, s3, ins = [v for v, _ in s2c], [v for v, _ in s3c], [v for v, _ in insc]
        assert [d for d, _ in s2] == [d for d, _ in s3[:len(s2)]]
        extra = list(range(len(s2), len(s3)))                   # fibers in S3 without a P2 in S2
        idx = link([p for p, _ in ins], [p for _, p in s2])     # value-based alignment, never row position
        dg = [(float(r['P2']), float(r['F_T'])) for r in dig if r['collector'] == name]
        for n, ((p2, xc), k) in enumerate(zip(ins, idx)):
            fid = 'PET-%s-%03d' % (name, n + 1)
            if k is not None:
                rec = (name, p2, xc, s3[k][1], s2[k][0], 'Fig. S3')
                cells = (insc[n][1], s2c[k][1], s3c[k][1])
            elif extra:
                e = extra.pop(0)
                d, ft = s3[e]
                chk = min(dg, key=lambda t: abs(t[0] - p2))       # the pairing must match the vector extraction
                assert abs(chk[0] - p2) < 0.003 and abs(chk[1] - ft) < 0.003, (name, p2, ft, chk)
                rec = (name, p2, xc, ft, d, 'Fig. S3 (no orientation in Fig. S2)')
                cells = (insc[n][1], '', s3c[e][1])
            else:
                ft = min(dg, key=lambda t: abs(t[0] - p2))
                assert abs(ft[0] - p2) < 0.003, (name, p2, ft)
                rec = (name, p2, xc, round(ft[1], 4), '', 'vector extraction of Fig. 3b')
                cells = (insc[n][1], '', 'data/digitized/laramee_2022_fig3b_main.csv')
            out.append(rec)
            d = rec[4]
            tidy.append((fid, name, d, p2, rec[3], xc, rec[5], *cells, 'yes', 'yes' if d != '' else 'no',
                         'yes' if d != '' and d < 550 else 'no', 'not reported'))
        assert not extra
    counts = {c: sum(1 for t in tidy if t[1] == c) for c in ('plate', 'gap', 'disk')}
    assert counts == {'plate': 56, 'gap': 105, 'disk': 79}, counts
    write('pet_per_fiber.csv', ['fiber_id', 'collector', 'diameter_nm', 'P2', 'F_T', 'X_c', 'F_T_source',
                                'cells_P2_Xc', 'cells_diameter_P2', 'cells_diameter_FT', 'in_channel_analyses',
                                'in_diameter_analyses', 'below_550nm_spot', 'session_id'], tidy)
    write('laramee_lanthier_pellerin_2022_fig3b.csv', ['collector', 'P2', 'X_c', 'F_T', 'diameter_nm', 'F_T_source'], out)

    # PAN, Knapp, Laramée and Pellerin 2026, Fig. 3 (immersion) and Fig. 4 (medians by diameter bin)
    ws = wb['PK 2026 Fig3']
    air = [(r[0], r[1]) for r in rows(ws, 2) if r[0] is not None]
    water = [('water', r[3], r[4]) for r in rows(ws, 2) if r[3] is not None]
    oil = [('oil', r[7], r[6]) for r in rows(ws, 2) if r[6] is not None]
    write('knapp_laramee_pellerin_2026_fig3.csv', ['medium', 'P2_four_spectrum', 'P2_calibrated'], water + oil)
    write('knapp_laramee_pellerin_2026_fig3_air.csv', ['P2_calibrated', 'P2_four_spectrum'], air)
    ws = wb['PK 2026 Fig4']
    f4 = [('air calibrated',) + tuple(r[0:4]) for r in rows(ws, 2) if r[0] is not None]
    f4 += [('oil four-spectrum',) + tuple(r[5:9]) for r in rows(ws, 2) if r[5] is not None]
    write('knapp_laramee_pellerin_2026_fig4.csv', ['series', 'diameter_bin_nm', 'P2_median', 'col_25th', 'col_75th'], f4)


if __name__ == '__main__':
    main()
