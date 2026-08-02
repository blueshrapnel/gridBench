# Fingerprint metrics

Current, selectively promoted analyses of structural axes for the paper suite.
The historical exploratory notebooks remain preserved in gridFour.

## Notebooks

- `00-bits-and-palette.py` — Daniel's bits relabelling of the cycle/basin
  axis and the non-Louvain palette candidates.
- `01-q-specialisation-screen.py` — the only promoted Louvain analysis. It
  tests basin co-membership Q on clean schema-11, pure-free-energy, full-goal
  7x7 run-bests against Cartesian and fresh environment-matched random nulls.

## Deliberate exclusions

The Q screen does not load or reproduce:

- the contaminated `free_energy_plus_modularity` campaigns;
- goal-subsampled or `k0xx` runs;
- K-annealing runs or the older scale fan;
- the older per-label one-step Louvain notebook;
- the community/skill claims from fingerprint notebooks 11 and 12.

Q is interpreted as cross-label basin-partition agreement. The graph has no
spatial-adjacency term, so spatially compact communities and policy boundaries
require independent tests.

## Running

The default data root is `/media/merlin/grid-twist/data-schema-11`; override it
with `GRIDBENCH_DATA_ROOT`. The random-null size defaults to 120 per environment
and can be changed with `GRIDBENCH_Q_NULL_SAMPLES`.

Outputs are written to `artifacts/` and `figures/` beside the notebook.
