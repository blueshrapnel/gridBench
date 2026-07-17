# Dominant-basin atlas migration and six-environment result

The legacy five-environment saturation notebook has been replaced by
`00-dominant-basin-atlas.py` under the current gridCore/gridBench stack.

The revised atlas contains six columns:

1. `wrap_grid`
2. `helical`
3. `open_grid`
4. `pinwheel`
5. `pillar_3`
6. `four_rooms`

The Cartesian, paired-random, and evolved rows all use the same plotting and
decomposition path. The paired-random row resets seed 3011 for each 49-state
environment and therefore uses the exact same 49x4 sigma in every column.
The wrap and helical maps differ only where the shifted seam changes a
successor.

All evolved panels were checked against the functional-graph parquet cache and
are full-goal free-energy run-bests. The evolved `wrap_grid`, `pinwheel`, and
`four_rooms` sigmas are exactly those used by the paper's following all-label
anatomy figure. Their sigma hashes are recorded in
`artifacts/dominant-basin-atlas-evolved-exemplars.csv`.

The helical Cartesian panel is the internal control for interpreting coverage:
its coverage is 1, but its terminal cycle contains all 49 states. The evolved
helical exemplar also has coverage 1 while draining to a two-state cycle.
Coverage therefore measures catchment extent; compactness of the recurrent
core is still required for homing.

Outputs:

- `figures/F-dominant-basin-atlas.pdf` (paper asset)
- `figures/F-dominant-basin-atlas.png` (inspection copy)
- `artifacts/dominant-basin-atlas-panels.csv` (all 18 plotted panels)
- `artifacts/dominant-basin-atlas-evolved-exemplars.csv` (run provenance)

