"""Update the text labels drawn by scripts/paper_figures.py to the terminology of the article, then redraw the figures.
Run from the repository root:  python tools/update_figure_labels.py [--write]
Without --write it only reports what would change. Labels split over several lines or built from variables are
not caught: check Figs. 4, 5 and 6 by eye after redrawing (see UPDATE_NOTES.md)."""
import re
import sys

PATH = 'scripts/paper_figures.py'
RULES = [
    # terminology
    (r'[Ff]lat[- ]diameter', lambda m: ('Uniform-arrest' if m.group(0)[0] == 'F' else 'uniform-arrest') + ' diameter'),
    (r'below-flat-diameter', lambda m: 'below-uniform-arrest-diameter'),
    (r'\bflat\b(?=\s*\\n\s*diameter)', lambda m: 'uniform-arrest'),
    (r'A1 has content', lambda m: 'A1 permits radial variation'),
    (r'validity window', lambda m: 'validity envelope'),
    # Fig. 6b and Fig. 6a legend
    (r'excluded by\\nthe plate', lambda m: 'excluded under the\\nobserved plate ceiling'),
    (r'excluded by the plate', lambda m: 'excluded under the observed plate ceiling'),
    (r'\[laboratory values\]', lambda m: '(supplied by the authors)'),
]

src = open(PATH, encoding='utf-8').read()
out = src
for pat, rep in RULES:
    hits = [m.group(0) for m in re.finditer(pat, out)]
    if hits:
        print(f'{pat!r}: {len(hits)} change(s)')
    out = re.sub(pat, rep, out)
for kw in ('flat', 'Remark 2', 'excluded', 'laboratory', 'has content'):
    for i, line in enumerate(out.splitlines(), 1):
        if kw in line:
            print(f'check line {i}: {line.strip()[:110]}')
if '--write' in sys.argv:
    open(PATH, 'w', encoding='utf-8').write(out)
    print('written; now run: python scripts/paper_figures.py')
else:
    print('dry run; add --write to apply')
