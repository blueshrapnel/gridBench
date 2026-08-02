# Basin typology and anatomy

This folder owns the basin figures used to connect the small-grid structural
analysis in *Twists, home vectors* to its scale study.

- `00-dominant-basin-atlas.py` replaces the frozen gridFour notebook
  `notebooks/attractor-fingerprint-probe/20-saturation-5env.py`. It adds
  `pillar_3`, uses current gridCore/gridBench functional graphs, treats the
  seed-3011 random twist as a paired topology control, and records the source
  of every evolved exemplar.
- The `wrap_grid`, `pinwheel`, and `four_rooms` evolved sigmas are the same
  twists opened into all four labels in the paper's following anatomy figure.
  The atlas is therefore the dominant-label projection; the anatomy is its
  within-twist expansion.
- `01-habit-cycle-intro.py` owns the Cartesian/GA-best habit-cycle comparison.
- `02-cycle-escape-policy.py` owns the cycle-escape and goal-optimal policy
  views, including the paper's v2 policy figure.
- `03-per-label-basin-anatomy.py` owns the six four-label anatomy panels at
  7×7, 9×9, and 13×13 and records their exact sigma provenance.

Run from the gridBench virtual environment:

```bash
MPLBACKEND=Agg python notebooks/basin-typology/00-dominant-basin-atlas.py
MPLBACKEND=Agg python notebooks/basin-typology/01-habit-cycle-intro.py
MPLBACKEND=Agg python notebooks/basin-typology/02-cycle-escape-policy.py
MPLBACKEND=Agg python notebooks/basin-typology/03-per-label-basin-anatomy.py
```

Outputs are written under `figures/` and `artifacts/`. The legacy notebook is
retained unchanged as provenance rather than deleted.
