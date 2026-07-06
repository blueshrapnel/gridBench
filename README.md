# gridBench

The active research workbench: notebooks and analysis scripts that are
currently in use, running on the modern package family (gridCore for
evaluation, gridVis for plotting).

## Why this repo exists

gridFour grew into a monorepo where sharing any one result meant sharing
everything.  The family now separates by audience:

- **gridCore** — evaluation kernel.  Shareable, pip-installable,
  cluster-cheap.  Papers may depend on it.
- **gridVis** — plotting recipes + tiny self-contained demos.  Shareable.
  Papers may depend on it.
- **gridBench** (this repo) — the private working bench.  Research
  notebooks, exploratory analysis, work in progress.  Never a
  dependency of anything.
- **gridFour** — frozen archive.  The paper-1/thesis notebooks stay
  there untouched, on the gridFour kernel, reproducing the published
  figures forever.  Do not port for porting's sake: the run-replay
  equivalence sweep (gridCore, 8.4e-14 across four code eras) is the
  certificate that old figures and new kernel agree.
- **paper repos** — text plus thin figure recipes that import pinned
  gridCore/gridVis only.  Nothing here may import gridBench or gridFour.

## Porting policy (lazy)

Port a gridFour notebook here only when you actually need to run it
again for new work.  Porting means:

1. imports come from gridcore/gridvis (no gridFour sys.path headers);
2. first cell prints the gridcore/gridvis commits (kernel provenance);
3. Jupyter-runnable throughout (no __file__), jupytext .py format;
4. the gridFour original stays where it is, untouched.

Known porting dependency: the functional-graph machinery
(decompose/fingerprints) lives in gridFour/src/analysis and is not yet
in gridCore; the attractor notebooks (35-50) need it moved (or a
gridcore.functional_graph module) before they can be ported.  The same
module is a prerequisite for gridTwist#23's full gridFour decoupling.
