# Single-label reach

Current gridBench notebooks for the deterministic functional graph obtained by
pressing one action label forever.  This folder supersedes the relevant figure
generation in the legacy gridFour
`notebooks/attractor-fingerprint-probe/43-journey-and-reach.py`; the legacy file
is retained as provenance.

| notebook | purpose |
|---|---|
| `00-four-rooms-reach-exemplar.py` | Rebuild the twists-home-vectors per-state reach exemplar from Cartesian, a fixed random twist, and the highest-mean-reach full-goal GA run-best.  Audits tied random home labels and the evolved-run selection. |
| `01-reach-by-class-two-shapes.py` | Compare Cartesian, row-shuffle-null, and GA-best mean single-label reach at two grid sizes across the six retained paper environments.  Supersedes the legacy gridFour notebook and excludes `x_wall`. |

The notebook uses current `gridbench` functional-graph code, writes figures to
`figures/`, and records its exact run and case selections in `artifacts/`.
