# Result: Q is a secondary specialisation-versus-homing diagnostic

The current gridCore/schema-11 screen supports promoting the corrected basin
co-membership Q implementation, but not the older Louvain notebook sequence or
its spatial-community interpretation.

## Cohort

- 94 unique pure-free-energy, full-goal 7x7 run-best sigmas;
- 120 fresh uniform-random twists per environment (720 total);
- one Cartesian control per environment;
- beta 1, determinism 0.97;
- no FEP+M, `gss`, `k0xx`, K-annealing, or random-search runs.

## Selection effect

Free-energy run-bests have lower Q than their environment-matched random null
in all six current-paper environments.

| environment | GA median Q | random median Q | difference / random SD |
|---|---:|---:|---:|
| open_grid | 0.090 | 0.374 | -3.62 |
| wrap_grid | 0.000 | 0.209 | -3.68 |
| helical | 0.006 | 0.204 | -3.58 |
| pinwheel | 0.326 | 0.659 | -12.32 |
| four_rooms | 0.270 | 0.541 | -4.22 |
| pillar_3 | 0.202 | 0.534 | -7.60 |

Except for two open-grid run-bests, every GA value lies outside its random
pool's central 95% interval. Louvain's scalar is reproducible: median GA
restart SD is at most 0.003 in every environment. Median partition agreement
across restarts is ARI 1.0 except on open_grid (0.739).

## Incremental contribution

Q is strongly anticorrelated with the existing coverage axis (within-env
Spearman -0.50 to -0.92), so it should not become another headline
fingerprint. After fitting Q from coverage, basin count, and cycle/basin ratio
on each random null, substantial GA residuals remain in pinwheel (-6.52 random
residual SD) and open_grid (+3.87). The remaining environments are closer to
the current fingerprint prediction.

The useful scientific statement is therefore:

> Pure free-energy selection favours broad homing at the expense of cross-label
> basin-partition agreement; the strength and residual form of that trade-off
> depend on geometry.

This is suitable as a secondary diagnostic for later action-space or
quality-diversity work. It is not a new headline for the home-vector paper and
does not establish spatially compact navigational communities.

## Outputs

- `artifacts/q-specialisation-observations.parquet`
- `artifacts/q-specialisation-summary.csv`
- `figures/F-q-specialisation-screen.png`
- `figures/F-q-specialisation-screen.pdf`
