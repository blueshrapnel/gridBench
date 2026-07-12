# relabelling-actions — ALife paper figure notebooks

Working notebooks for the ALife "Relabelling Actions" conference paper figures.
This is the **development home** (per-topic folder under `gridBench/notebooks/`,
like `goal-geometry/`). The public paper repo `alife-relabelling-actions`
receives a frozen, self-contained bundle only at publish time (see below).

## Figures

| notebook | paper figure(s) |
|---|---|
| `figure-1-simplified.ipynb` | `figure1_ga_sigma_simple` |
| `figure-2-simplified.ipynb` | `figure2_random_sigma` |
| `figure-5-env-survey.ipynb` | `figure5_envs` (+ per-env / diagnostic / per-action panels) |
| `figure-6.ipynb` | `figure-6-*` (per-action direction, per-goal structure) |

## Dependencies

Editable installs in the `py-3.12-grid` env: `gridcore` (numerics/envs),
`gridvis` (plotting), `gridbench` (analysis: `marginal_policy`, `store`).
Plotting is imported directly from `gridvis`; `figure_support.py` /
`wrap_grid_helpers.py` / `ga_report_helpers.py` here are paper-specific glue.

## Data (schema-11, DATA_ROOT-relative)

Notebooks read the schema-11 store through `gridbench.store`, whose root is
`GRIDBENCH_DATA_ROOT` (default `/media/merlin/grid-twist/data-schema-11`).
Run references are **specs**, not absolute paths, in `paper-refs.json` — each
carries `init_method` + `fitness_objective` (schema-11 made both first-class
partition levels) plus `env_id/shape/beta/det/run_name`. `figure_support`
resolves them via `store.run_dir(**spec)`. Nothing assembles partition paths by
hand or assumes a single init method.

## Run

```bash
cd gridBench/notebooks/relabelling-actions
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 figure-6.ipynb
```
Figures land in `./figures/`.

## Publish freeze (manual, once-off per paper — do NOT over-automate)

At publication, assemble a self-contained bundle for the public paper repo so
readers can rerun without the lab store (mirrors `cognitive-geometry/`'s
`{data/, notebooks/, src/}`, ~a few MB):

1. **Minimal data** → `<bundle>/data/` preserving the schema-11-relative layout:
   for each run in `paper-refs.json`, copy `*.summary.json`,
   `sigma_aggregates.csv`, and only the needed `sigma_id=.../` dirs
   (`sigma.npy` + `metrics.npz` + `goal-*.npz`, ~36 KB each) — not the full
   `hive_sigma/` (530 MB). Plus the committed `four_rooms/assets/`.
2. **Notebooks** → a simplified copy (paper-figure cells only) + `figure_support.py`
   + `paper-refs.json`.
3. **Config**: set `GRIDBENCH_DATA_ROOT=./data` in the bundle (the only switch
   needed — all reads are DATA_ROOT-relative).
4. **Code**: vendor a snapshot of `gridcore`/`gridvis`/`gridbench` + a
   `requirements.txt`.
5. Drop the bundle in `alife-relabelling-actions/reproduce/` (ships with the
   HTML report site).

Verify by running the bundle notebooks with `GRIDBENCH_DATA_ROOT=./data` and the
live store unreachable — the paper figures must reproduce from bundled data alone.
