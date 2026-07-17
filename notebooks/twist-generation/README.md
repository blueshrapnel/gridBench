# Twist generation

Diagnostic notebooks for the priors over twists used by the current papers.
This folder is about how twists are generated before selection; it is separate
from the evolved-population analyses in `goal-geometry/`.

The frozen gridFour notebook
`notebooks/action-alignment/action_ordering_epsilon_profiles.ipynb` records
some of the development history.  It is a provenance pointer, not the
specification: the current gridTwist implementations and the populations they
generate are authoritative.  For the twists-home-vectors paper we restrict
attention to the two initialisers actually used there.

## Notebooks

| notebook | question |
|---|---|
| `00-compare-initialiser-footprints.py` | Is the paper's fresh-uniform null structurally distinguishable from generation-zero row-shuffle and permutation-balanced populations? Also renders the shared-axis comparison and the two-prior previews for Figures 7 and 8, and audits generation, objective, initialiser, and seed explanations for the open-grid run-best bands. |

The first result is recorded in
`RESULT-2026-07-16-initialiser-footprints.md`.

Run from the `py-3.12-grid` environment.  The notebook writes diagnostic
figures to `figures/`, a compact numerical summary to `artifacts/`, and its
recomputable sample cache to `_cache/`.
