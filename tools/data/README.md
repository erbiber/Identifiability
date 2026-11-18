# Data

## Values supplied by the source authors

`raw/pellerin_compilation_of_data.xlsx` is the workbook supplied by C. Pellerin (Université de Montréal), corresponding
author of the source articles, compiled to reproduce their published figures. It is shared here with his agreement and
is the source of the per-fiber and bin values used in Section 5 of the article (Sections S8.3 and S8.6–S8.8).

| Sheet | Source | Content | Used for |
|---|---|---|---|
| MRL 2015 Fig2A | Richard-Lacroix & Pellerin, Macromolecules 48 (2015) 4511 | PS bin means and error bars of orientation | PS skin bound (Section 5.1.2) |
| MRL 2015 Fig5A | same | PS disentanglement marker by bin | Two-channel analysis (Section 5.1.3) |
| AWL 2022 FigS2, FigS3, Fig3 inset | Laramée, Lanthier & Pellerin, Appl. Spectrosc. 76 (2022) 51 | PET per-fiber orientation, diameter, trans fraction and crystallinity by collector | PET plate control and channels (Section 5.2) |
| PK 2026 Fig3, Fig4 | Knapp, Laramée & Pellerin, Appl. Spectrosc. (2026) | PAN four-spectrum and calibrated orientation in air, water and oil; medians | PAN readout agreement (Section 5.3.1) |

In the workbook, "drum" is the rotating disk and "foil" the static plate of the article, and "SM" the four-spectrum (standard methodology) readout.

The CSV files in `data/` (`richard_lacroix_pellerin_2015_fig2a.csv`, `_fig5a.csv`,
`laramee_lanthier_pellerin_2022_fig3b.csv`, `knapp_laramee_pellerin_2026_fig3.csv`, `_fig3_air.csv`, `_fig4.csv`) are
imported from this workbook by `scripts/import_author_data.py`.

The disk speed of the PS record, 29.4 m/s, is a personal communication from C. Pellerin (October 2026) and is cited as
such in the article.

## Digitized values

`digitized/` holds values read independently from the published figures as a check, with the scripts that produced
them. Their agreement with the supplied values is reported in Section S8.3 (PS within 0.001 in
⟨P2⟩), S8.7 (PET within 0.0003 and 0.004) and S8.8 (PAN within 0.007, one oil fiber missed).

## Licence

CC BY 4.0 (see LICENSE-DATA). Please cite the source articles when reusing the record values.
