"""Independent check of the PET per-fiber record: read the markers of Fig. 3b of Laramée, Lanthier and Pellerin
(Appl. Spectrosc. 2022, 76, 51-61) from the drawing commands of the published PDF.
Usage: python scripts/digitize_pet_fig3b.py path/to/10.1177_00037028211049242.pdf
Writes data/digitized/laramee_2022_fig3b_main.csv (orientation and trans fraction of every marker of the main panel).
The PDF itself is not redistributed. Requires PyMuPDF."""
import csv
import os
import sys

import pymupdf

# main panel of Fig. 3b on page 6: frame and axis limits read from the tick marks
X0, X1, XV0, XV1 = 101.55, 264.95, -0.1, 0.8
Y0, Y1, YV0, YV1 = 425.17, 272.61, 0.0, 0.5
SIZES = {2.7: 'gap', 2.2: 'disk', 1.9: 'plate'}   # marker widths (pt) in the main panel


def main(pdf):
    page = pymupdf.open(pdf)[5]
    rows = []
    for d in page.get_drawings():
        r = d['rect']
        if not (X0 - 2 < r.x0 and r.x1 < X1 + 2 and Y1 - 2 < r.y0 and r.y1 < Y0 + 2):
            continue
        if d['type'] != 's':            # each marker has one stroked outline; the plate fill duplicates it
            continue
        w = round(r.width, 1)
        if w not in SIZES or abs(r.height - r.width) > 0.2:
            continue
        cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
        p2 = XV0 + (cx - X0) / (X1 - X0) * (XV1 - XV0)
        ft = YV0 + (cy - Y0) / (Y1 - Y0) * (YV1 - YV0)
        rows.append((SIZES[w], round(p2, 5), round(ft, 5)))
    os.makedirs('data/digitized', exist_ok=True)
    with open('data/digitized/laramee_2022_fig3b_main.csv', 'w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(['collector', 'P2', 'F_T'])
        w.writerows(rows)
    print({c: sum(1 for r in rows if r[0] == c) for c in SIZES.values()})


if __name__ == '__main__':
    main(sys.argv[1])
