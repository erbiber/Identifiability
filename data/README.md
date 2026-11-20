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

### PET per-fiber table

`pet_per_fiber.csv` is the tidy table of the 240 PET fibers: fiber identifier, collector, diameter, ⟨P2⟩, trans
fraction, crystallinity, the workbook cells each value was read from, inclusion flags (channel analyses, diameter-based
analyses, below the 550 nm laser spot) and the spinning session, which the source does not report. The fibers are
defined by the 240 orientation–crystallinity pairs of the sheet `AWL 2022 Fig3 inset` (56 plate, 105 gap, 79 disk). The
diameter and trans-fraction sheets are matched to them by value, not by row: four records have no diameter or trans
entry, and several gap records are reordered or rounded, so row-wise alignment would create spurious discrepancies.
`tests/test_import.py` re-reads every cited cell and checks it against the table, and checks that the analysis input
`laramee_lanthier_pellerin_2022_fig3b.csv` holds exactly the same fibers.

The disk speed of the PS record, 29.4 m/s, is a personal communication from C. Pellerin (October 2026) and is cited as
such in the article.

## Literature tables used by the diffusivity model

| File | Source | Content |
|---|---|---|
| `crank_park_1949_table2.csv` | Crank & Park, Trans. Faraday Soc. 45 (1949) 240, Table II | Chloroform diffusivity in glassy polystyrene at 25 °C (second approximation) against equilibrium regain |
| `waggoner_1993_table1.csv` | Waggoner, Blum & MacElroy, Macromolecules 26 (1993) 6841, Table I | Chloroform and toluene self-diffusivity in polystyrene solutions at 25 °C |
| `pickup_blum_1989_table1.csv` | Pickup & Blum, Macromolecules 22 (1989) 3961, Table I | Toluene self-diffusivity in polystyrene solutions, 25–115 °C |

The free-volume parameters (Zielinski & Duda, AIChE J. 38 (1992) 405: chloroform from Table 2, polystyrene from Table 3,
toluene–polystyrene from Table 4) are set in `scripts/diffusivity.py`; the chloroform jump-volume ratio is computed from
their Eq. 12. Of Crank and Park's Table II, the row extrapolated to 5.0% regain and the 16.8% row, which has no second
approximation, are not used.

## Digitized values

`digitized/laramee_2022_fig3b_main.csv` holds the orientation and trans fraction of every marker of the main panel of
Fig. 3b of the PET source, read from the drawing commands of the published PDF by `scripts/digitize_pet_fig3b.py` (the
PDF itself is not redistributed). It agrees with the supplied values within 0.0002 in both quantities, and supplies the
trans fraction of the four fibers that have no diameter, and therefore no entry in Fig. S3 of the source.

## License

CC BY 4.0 (see LICENSE-DATA). Please cite the source articles when reusing the record values.
