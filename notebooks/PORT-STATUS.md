# Notebook port status: gridFour → gridBench

Tracks the migration of the exploratory / figure notebooks from the **frozen**
`gridFour/notebooks/` tree into gridBench topic folders, on the three-repo
split (`gridcore` = numerics, `gridbench` = analysis + hive-IO, `gridvis` =
plotting). gridFour is never edited; notebooks are **promoted** (rewritten with
new imports), not copied 1:1. Superseded and other-paper scripts are recorded
here so they are not re-ported by mistake.

Source of most pending work: `gridFour/notebooks/attractor-fingerprint-probe/`.

## Porting checklist (per notebook)

1. Copy the script into the right gridBench **topic folder** (create one with a
   `README.md` if the theme is new).
2. Rewrite imports only — keep the body byte-identical where possible so figure
   parity is verifiable:
   - `utility.twists` → `gridcore.twists`
   - `utility.information_theory.get_expected_vector` → `gridcore.information`
   - `analysis.functional_graph.*` → `gridbench.functional_graph.*`
   - `utility.display` / `display_twist*` → `gridvis.*`
   - `utility.constants.metric_clims` → `gridvis.constants.metric_clims`
   - `utility.metrics` → `gridbench.metrics`; `directory_tools` → `gridbench.store`
   - drop the `sys.path` bootstrap.
3. **Jupyter-safe location — REQUIRED.** Karen opens these in `jupyter notebook`
   (jupytext), where `__file__` is undefined in kernel cells. Every notebook
   must resolve its directory through the `_nb_dir()` helper (three fallbacks:
   `__file__` when run as a script, kernel cwd when it holds the notebook, else
   the canonical repo path). Copy the exact helper from
   `goal-geometry/17-moat-experiment.py`. A bare `__file__` (or even a plain
   `try/except NameError` that guesses cwd) is not sufficient — use `_nb_dir()`.
   Canonical commit: `3bf1b99`.
4. Run both ways — a script run AND an exec without `__file__` from a foreign
   cwd — and **parity-check** the emitted figures against the gridFour originals
   before considering the port done ([[feedback-preserve-builds-before-promoting]]).

## Ported & in sync

| gridFour probe | gridBench | notes |
|---|---|---|
| `21-fingerprint-journey-two-panel` | `paper-homing/21-fingerprint-planes` | verified in sync 2026-07-17 (only header/import lines differ) |
| `36-reach-by-class-two-shapes` | `single-label-reach/01-reach-by-class-two-shapes` | |
| `54-attractor-placement-catchment` | `goal-geometry/13-attractor-placement-catchment` | |

Other gridBench topic folders originate from screens/scratchpad, not the probe
tree: `basin-typology/00` (dominant-basin atlas), `single-label-reach/00`
(four-rooms reach exemplar), `twist-generation/00` (initialiser footprints),
`goal-geometry/14-18`, `fingerprint-metrics/00`, `relabelling-actions/` (the 4
ALife figure notebooks).

## Superseded — do NOT port

| gridFour probe | superseded by |
|---|---|
| `20-saturation-5env` | `basin-typology/00-dominant-basin-atlas` (dominant-basin atlas replaced the saturation-basins-5env figure, checkpoint 2026-07-14) |

## Other-paper (K-annealing / goal-subsampling) — port only if that paper needs it

These belong to the K-annealing / goal-subsampling thread, not twists-home-vectors.
K≠1 data is sync-only (never schema-11-imported unasked).

- `34-K025-vs-K1-comparison` — K=0.25 vs K=1.0 coverage on the 3 anchor seeds
- `44-kj-coverage-comparison` — K-only vs K+J coverage(gen), single seed
- `45-kj-multiseed-comparison` — K-only vs K+J=16, n=4 seeds (retires 44's n=1 caveat)
- `46-goal-subsampling-coverage-analysis` — canonical consolidated K/J analysis
- `48-subsample-attractor-similarity` — do K-subsampled evolutions find the same attractor
- `49-anneal-pilot-verdict`, `50-anneal-visualisation`, `53-anneal-scale-identity` — annealing verdict/viz/scale

## Pending — twists-home-vectors relevant

Ordered roughly by paper-currency. Pick a target and port per the checklist above.

| gridFour probe | what it is | paper hook |
|---|---|---|
| `51-cycle-escape-exemplar-and-heatmap` | cycle-escape exemplar+table figure + role-summary heatmap (prototype replacement for the 12-grid figure) | `sec:cycle-escape` (geometric-gate / retirement) |
| `43-journey-and-reach` | journey, repeatability, reachability figures | pairs with `paper-homing/21`; reach-recovered section |
| `33-three-regimes-triptych` | total / asymmetric / modest saturation triptych | saturation regimes |
| `32-per-generation-evolution` | per-generation home-vector formation of a single run | mechanism / evolution |
| `35-9x9-scale-coverage` | 9×9 anchor seeds, coverage across budgets (g200/g500/g1000) | "Homing at scale" |
| `37-fourrooms-scale-study` | coverage + basin count vs grid side | "Homing at scale" |
| `31-9x9-g500-vs-g200` | paired g200-vs-g500 (under-training vs geometry blocks saturation) | budget / under-training |
| `30-paper1-extended-analysis` | all envs + β-sweep + DI-vs-FE on cached σs | broad diagnostic |
| `52-q-modularity-null-scale` | Q_modularity null check + scale trend (follow-up to 51's seam negative) | modularity (metric-selection) |
| `40-build-metric-dataset` | build the cross-env metric-zoo expressive-range dataset | metric-zoo report (data build) |
| `41-plot-metric-figures` | per-metric expressive-range figures | metric-zoo report |
| `42-metric-diagnostics` | metric-zoo decision diagnostics | metric-zoo report |

`40→41→42` are a chain (dataset build → figures → diagnostics); port together if
the metric-zoo report is the target.
