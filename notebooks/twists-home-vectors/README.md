# *Twists, home vectors* figure ownership

gridBench is the development home for every figure used by the paper.  The
paper repository owns the prose and a published copy of each asset; gridFour
is frozen provenance and is not a runtime dependency.

The generators remain in topic folders so analyses stay with their supporting
code.  `figure-manifest.py` is the paper-wide index and verifies that every
paper include and every owning generator still exists.

| paper figure | published asset(s) | gridBench owner |
|---|---|---|
| 1 | `F-environment-palette.{png,pdf}` | `environment-topology/00-environment-palette.py` |
| 2 | `F-alife-four-rooms-twist.pdf` | `relabelling-actions/figure-1-simplified.ipynb` |
| 3 | `F-alife-tradeoff.pdf` | `relabelling-actions/figure-3-tradeoff.py` |
| 4 | `F0-habit-cycle-intro.png` | `basin-typology/01-habit-cycle-intro.py` |
| 5–6 | `F1-trajectory-streaks.png`, `F2-diffusion-summary.png` | `single-label-reach/02-random-twist-diffusion.py` |
| 7 | `F-reach-by-class.png` | `single-label-reach/01-reach-by-class-two-shapes.py` |
| 8 | `F-reach-trajectories.png` | `single-label-reach/00-four-rooms-reach-exemplar.py` |
| 9–10 | `F-fingerprint-{open,walled}-interiors.png` | `paper-homing/21-fingerprint-planes.py` |
| 11 | `F-dominant-basin-atlas.pdf` | `basin-typology/00-dominant-basin-atlas.py` |
| 12–13 | six `F-basins-*.png` panels | `basin-typology/03-per-label-basin-anatomy.py` |
| 14 | `F-seed-strips-four-label.png` | `goal-geometry/14-seed-strips-four-label.py` |
| 15 | `F-cycle-escape-policy-view-v2.png` | `basin-typology/02-cycle-escape-policy.py` |
| 16 | `F-alignment-roundup.png` | `paper-homing/21-fingerprint-planes.py` |

## Wall palette

Figure 8 fixes the paper wall colour at `#595959`.  Figures 1, 4, 8, 11, 12,
13, and 15 import `gridbench.papers.home_vectors.WALL_COLOUR`; walls are rendered
opaque.  Figure 1 additionally overlays vector rectangles, keeping the PDF
edges hard when a viewer rescales its small raster topology layer.  The paper
uses the PNG for Figure 1, while the repaired PDF remains available.

## Checks and execution

Use the project environment:

```bash
PYENV_VERSION=py-3.12-grid python \
  notebooks/twists-home-vectors/figure-manifest.py
```

The manifest is an index/check, not a monolithic rerun command.  Several
figures perform full policy solves or consume pinned run data; run the owning
topic notebook when its analysis changes.  Paper-facing generators export to
`/home/karen/Dropbox/phd/writing/twists-home-vectors/figures/` and keep their
normal topic-local copy where applicable.
