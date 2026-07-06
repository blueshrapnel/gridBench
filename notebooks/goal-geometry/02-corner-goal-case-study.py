# %% [markdown]
# # Goal-geometry series, 02 — the corner-goal case study, quantified
#
# The space-of-goals paper walks through the goal one-in-from-the-corner:
# reaching it is cheap, but the policy that reaches it works hard against
# the prevailing (prior) flow on the way OUT — an information cost the
# averaged-bidirectional MDS treatment cannot show.  This notebook
# quantifies that story on the PAPER'S OWN matrices: the beta-family
# pickles (11x11 open grid, Manhattan, live distribution, det=1.0)
# from cognitive-geometry/data/11-11-det/.
#
# Three quantifications:
#   1. **Sink strength** per goal g: s(g) = mean_i (F(g->i) - F(i->g))/2,
#      positive when leaving g costs more than reaching it — g is
#      downstream of the prevailing policy.  Rendered as a grid heatmap:
#      the corner-goal case becomes the extreme of a field, not an
#      anecdote.
#   2. **Named-goal asymmetries** across beta: corner (0,0), the paper's
#      one-in-from-corner (1,1), edge midpoint, and centre — F(centre->g)
#      vs F(g->centre) in bits.
#   3. **Drift-MDS panels** at high and low beta with the named goals
#      starred: at beta=100 control is near-deterministic and the
#      current nearly vanishes; as beta drops the information term
#      dominates and the corner current switches on.

# %%
import pickle
import subprocess
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gridcore
import gridvis

for pkg in (gridcore, gridvis):
    repo = Path(pkg.__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = "unknown"
    print(f"{pkg.__name__}: {repo.name} @ {commit}")

# %% [markdown]
# ## Load the paper's beta family
#
# Legacy pickles need a pandas shim (pre-2.0 Int64Index) and the
# cognitive-geometry src on the path for their env classes.  We only
# keep the frees matrices.

# %%
import pandas as pd

_shim = types.ModuleType("pandas.core.indexes.numeric")
_shim.Int64Index = pd.Index
_shim.Float64Index = pd.Index
_shim.UInt64Index = pd.Index
sys.modules["pandas.core.indexes.numeric"] = _shim
sys.path.insert(0, "/media/merlin/phd-marlyn/cognitive-geometry/src")

DATA = Path("/media/merlin/phd-marlyn/cognitive-geometry/data/11-11-det")
BETAS = [0.1, 0.3, 0.5, 1, 10, 100]
SHAPE = (11, 11)
W = SHAPE[1]

D_by_beta = {}
for b in BETAS:
    p = DATA / f"data-11-11-man-liv-det-1.0-b-{b}-Z.pickle"
    if not p.exists():
        print(f"missing: {p.name}")
        continue
    with open(p, "rb") as fh:
        d = pickle.load(fh)
    frees = np.asarray(d["frees"], dtype=float)   # frees[g][s] = F(s -> g)
    D = frees.T.copy()                            # D[s][g]
    np.fill_diagonal(D, 0.0)
    D_by_beta[b] = D
print("loaded betas:", sorted(D_by_beta))

# %% [markdown]
# ## 1. Sink strength across beta

# %%
from gridcore.geometry import drift_vectors, gower_decompose, smacof_embed

NAMED = {
    "corner (0,0)": 0,
    "one-in (1,1)": 1 * W + 1,
    "edge mid (0,5)": 5,
    "centre (5,5)": 5 * W + 5,
}

fig, axes = plt.subplots(1, len(D_by_beta), figsize=(3.1 * len(D_by_beta), 3.6), dpi=150)
sink_by_beta = {}
for ax, (b, D) in zip(np.atleast_1d(axes), sorted(D_by_beta.items())):
    _, skew = gower_decompose(D)
    sink = -skew.mean(axis=0)      # s(g) = mean_i (D[g,i]-D[i,g])/2... sign below
    # skew[i,g] = (F(i->g) - F(g->i))/2; sink(g) = -mean_i skew[i,g] > 0
    # when reaching g is CHEAP relative to leaving it.
    sink = -skew.mean(axis=0)
    sink_by_beta[b] = sink
    im = ax.imshow(sink.reshape(SHAPE), cmap="RdBu_r",
                   vmin=-np.abs(sink).max(), vmax=np.abs(sink).max())
    ax.set_title(f"$\\beta={b}$", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("Sink strength s(g) [bits]: red = cheap to reach, expensive to leave "
             "(downstream of the prevailing policy)", fontsize=11)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 2. The named goals, across beta

# %%
print(f"{'beta':>6s} {'skew/sym':>9s} " + " ".join(f"{k:>15s}" for k in NAMED))
centre = NAMED["centre (5,5)"]
for b, D in sorted(D_by_beta.items()):
    sym, skew = gower_decompose(D)
    ratio = np.linalg.norm(skew) / np.linalg.norm(sym)
    cells = []
    for name, g in NAMED.items():
        # asymmetry of the centre<->g pair, in bits: F(centre->g) - F(g->centre)
        cells.append(f"{D[centre, g] - D[g, centre]:15.3f}")
    print(f"{b:6g} {ratio:9.4f} " + " ".join(cells))
print("\n(negative = the goal is cheaper to reach from the centre than to "
      "return from: a sink; the corner-goal story predicts corner < one-in "
      "< edge < centre ~ 0, growing as beta falls)")

# %% [markdown]
# ## 3. Drift-MDS at the beta extremes, named goals starred

# %%
from gridvis.goalspace import plot_drift_mds

show_betas = [b for b in (100, 1, 0.3) if b in D_by_beta]
fig2, axes2 = plt.subplots(1, len(show_betas), figsize=(6.4 * len(show_betas), 6.2), dpi=150)
for ax, b in zip(np.atleast_1d(axes2), show_betas):
    D = D_by_beta[b]
    sym, skew = gower_decompose(D)
    coords, stress = smacof_embed(sym, components=2)
    drift = drift_vectors(coords, skew)
    mag = np.linalg.norm(drift, axis=1)
    keep = mag > 1e-6
    if not keep.all():
        drift = np.where(keep[:, None], drift, 0.0)
    plot_drift_mds(ax, coords, drift,
                   node_values=[g // W for g in range(D.shape[0])],
                   title=f"$\\beta={b}$", stress=stress,
                   skew_ratio=float(np.linalg.norm(skew) / np.linalg.norm(sym)))
    for name, g in NAMED.items():
        ax.plot(*coords[g], marker="*", ms=16, mfc="crimson", mec="white", zorder=5)
        ax.annotate(name.split(" ")[0], coords[g], fontsize=8, xytext=(6, 6),
                    textcoords="offset points")
fig2.suptitle("The paper's goal space with its current restored "
              "(11x11 open grid, Manhattan, live distribution, det=1.0)",
              fontsize=12)
fig2.tight_layout()
plt.show()

# %% [markdown]
# ## Cross-kernel validation: does gridCore regenerate the paper's data?
#
# The chapter will recompute these matrices with the modern kernel, so
# the agreement has to be on record.  For each beta: rebuild D through
# gridcore.bridge (11x11 open grid, Manhattan, LIVE distribution,
# det=1.0 — the pickles' exact configuration) and compare both the raw
# matrices and the chapter's derived objects (sink field, named-goal
# asymmetries, skew ratio).  Live-distribution fixed points carry
# path-dependent p_s updates, so expect agreement at the ~1e-2 level
# (as measured for figure-3), not round-off; the derived asymmetries
# are 0.1-47 bits, so that is ample.

# %%
from gridcore.bridge import EvalConfig, _state_dist_class, build_twisted_env_from_sigma
from gridcore.info import DecisionInformation

nS = SHAPE[0] * SHAPE[1]
identity = np.tile(np.arange(4, dtype=np.int64), (nS, 1))


def gridcore_D(beta: float) -> np.ndarray:
    D = np.zeros((nS, nS))
    for g in range(nS):
        cfg = EvalConfig(env_id="open_grid", shape=SHAPE, goal=g, beta=float(beta),
                         determinism=1.0, manhattan=True, theta=1e-5,
                         state_dist="live")
        env = build_twisted_env_from_sigma(identity, cfg)
        di = DecisionInformation(env, _state_dist_class("live")(env), 1e-5,
                                 max_iterations=200_000, max_info_iterations=10_000)
        _, _, F = di.get_opt_policy_Z_free_vector(float(beta))
        if not di.converged:
            return None  # float64 low-beta wall (gridFour#15)
        D[:, g] = F
    np.fill_diagonal(D, 0.0)
    return D


print(f"{'beta':>6s} {'max|dD|':>9s} {'max|d sink|':>12s} {'max|d asym|':>12s} "
      f"{'skew/sym old':>13s} {'new':>8s}")
for b in sorted(D_by_beta):
    D_old = D_by_beta[b]
    D_new = gridcore_D(b)
    if D_new is None:
        print(f"{b:6g}  NOT CONVERGED in float64 BA (the gridFour#15 low-beta "
              f"wall at 11x11/det=1.0/live) — yet the 2021 pickle exists: "
              f"did the paper-era run converge, or is the pickle a forced-exit "
              f"iterate?  Follow-up for the chapter + the log-domain work.")
        continue
    _, skew_o = gower_decompose(D_old)
    sym_o, _ = gower_decompose(D_old)
    sym_n, skew_n = gower_decompose(D_new)
    d_sink = np.abs((-skew_n.mean(axis=0)) - (-skew_o.mean(axis=0))).max()
    d_asym = max(abs((D_new[centre, g] - D_new[g, centre])
                     - (D_old[centre, g] - D_old[g, centre])) for g in NAMED.values())
    print(f"{b:6g} {np.abs(D_new - D_old).max():9.2e} {d_sink:12.2e} {d_asym:12.2e} "
          f"{np.linalg.norm(skew_o)/np.linalg.norm(sym_o):13.4f} "
          f"{np.linalg.norm(skew_n)/np.linalg.norm(sym_n):8.4f}")

# %% [markdown]
# ## Discernments (2026-07-06 run, gridcore d432c4b)
#
# **The corner-goal story, quantified.**  Sink strength peaks at the TRUE
# corners (the paper's one-in goal is strong but intermediate:
# corner < one-in < edge-mid < centre = 0 at every beta), the field
# shape is beta-invariant, and the amplitude follows ~ -10/beta bits
# (corner asymmetry -0.100 at beta=100, -1.00 at 10, -9.1 at 1, -47 at
# 0.1).  The drift is a PURE information-cost phenomenon, exactly
# (1/beta)-weighted, vanishing in the value-dominated limit — which is
# why figure 3, computed at beta=100 (skew/sym 0.0026), could not show
# the corner story its own text walks through.  For the chapter figure,
# beta ~ 0.5-1 makes the current legible without leaving the
# certified-convergence regime.
#
# **Cross-kernel validation: a three-part verdict.**
# 1. At the paper's published setting (beta=100) gridCore regenerates
#    the pickle to max|dD| = 1.7e-2 — the same ~2e-2 live-distribution
#    agreement measured for figure 3.  Validated.
# 2. As beta falls the kernels diverge (max|dD| 0.17 at beta=10, 2.7 at
#    beta=1, 7.3 at beta=0.5 — roughly 1/beta, i.e. a CONSTANT fraction
#    of the information-cost structure, ~5-8% on the derived sink/
#    asymmetry fields).  Both kernels converge; they converge to
#    path-dependent fixed points, because the LIVE distribution couples
#    p_s to the policy through a thresholded update and the coupling
#    stiffens as control softens.  The qualitative story (ordering,
#    1/beta law, field shape) is identical in both kernels; the exact
#    values are not a well-posed single-valued target under live.
# 3. beta <= 0.3 does not converge in float64 at 11x11/det=1.0/live at
#    all (residual stuck ~1.8) — the gridFour#15 wall, WIDER here than
#    on the smaller uniform-distribution cases (0.3 converged there).
#    Yet the 2021 pickles exist: follow up whether the paper-era runs
#    truly converged or stored forced-exit iterates.
#
# **Recommendation for the chapter**: recompute the beta-family under
# the UNIFORM state distribution as the canonical surface — uniform p_s
# is fixed, the fixed point is unique, and the run-replay sweep
# certifies cross-kernel agreement at 8.4e-14 there — and keep the
# paper's live-distribution pickles as historical reference.  The
# corner-goal narrative survives either choice; its numbers should rest
# on the well-posed one.
