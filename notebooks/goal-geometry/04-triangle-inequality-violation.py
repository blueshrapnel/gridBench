# %% [markdown]
# # Goal-geometry series, 04 — triangle-inequality violation, systematically
#
# Experiment 1 of the infodesic paper (~/writing/twists-infodesics): the
# policy-switching notebook (gridFour decision-information/
# policy-switching.ipynb) concluded that the free-energy triangle
# inequality FAILS across goal-specific policies — reaching C via an
# intermediate goal B, switching policy at B, can cost less than the
# direct route under C's own policy.  That failure is the formal seed of
# segmentation: where it happens, multi-policy (segmented-infodesic)
# routing beats single-policy routing.
#
# This notebook measures it systematically.  With D[i, j] = F_j(i) — the
# uniform-prior free energy of reaching goal j from state i under j's
# optimal policy (exactly notebook 03's D matrix) — an ordered triple
# (A, B, C) VIOLATES when
#
#     D[A, B] + D[B, C]  <  D[A, C] - tol,
#
# i.e. the segmented cost (each leg under its own goal policy, prior and
# all) undercuts the direct cost.  We report, per environment and beta:
# the violating fraction of ordered triples, the gap distribution, and —
# the infodesic prediction — WHERE the profitable midpoints B sit
# (doorways should dominate in walled worlds).
#
# Untwisted (Cartesian identity) environments first: this is a property
# of the environment's free-energy geometry, before any twist.

# %%
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gridcore

repo = Path(gridcore.__file__).resolve().parents[2]
try:
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], text=True
    ).strip()
except Exception:
    commit = "unknown"
print(f"gridcore: {repo.name} @ {commit}")

# %%
from gridcore.bridge import EvalConfig, _state_dist_class, build_twisted_env_from_sigma
from gridcore.info import DecisionInformation

SHAPE, DET, THETA = (7, 7), 0.97, 1e-5
ENVS = ["four_rooms", "open_grid", "wrap_grid", "corr_1d_ring"]
BETAS = [0.3, 1.0, 3.0]
TOL = 1e-6


def d_matrix(env_id, beta):
    """D[i, j] = F_j(i) over available states (notebook-03 pattern)."""
    nS = SHAPE[0] * SHAPE[1]
    identity = np.tile(np.arange(4, dtype=np.int64), (nS, 1))

    def build(goal):
        cfg = EvalConfig(env_id=env_id, shape=SHAPE, goal=int(goal), beta=beta,
                         determinism=DET, manhattan=True, theta=THETA,
                         state_dist="uniform")
        return build_twisted_env_from_sigma(identity, cfg)

    env0 = build(0)
    goals = [int(s) for s in env0.available_states]
    n = len(goals)
    D = np.zeros((n, n))
    for j, g in enumerate(goals):
        env = build(g)
        di = DecisionInformation(env, _state_dist_class("uniform")(env), THETA,
                                 max_iterations=200_000,
                                 max_info_iterations=10_000)
        _, _, F = di.get_opt_policy_Z_free_vector(beta)
        assert di.converged, (env_id, beta, g)
        D[:, j] = np.asarray(F, dtype=float)[goals]
    np.fill_diagonal(D, 0.0)
    return D, goals


def triangle_stats(D):
    """Violation fraction + gaps over ordered triples A != B != C.

    Vectorised: gap[A, B, C] = D[A, C] - (D[A, B] + D[B, C]); a triple
    violates when gap > TOL (segmenting at B is strictly cheaper).
    """
    n = D.shape[0]
    seg = D[:, :, None] + D[None, :, :]          # (A, B, C): D[A,B] + D[B,C]
    gap = D[:, None, :] - seg                    # (A, B, C)
    A, B, C = np.ogrid[:n, :n, :n]
    valid = (A != B) & (B != C) & (A != C)
    gaps = gap[valid]
    viol = gaps > TOL
    # midpoint profitability: how often is B the best (max-gap) midpoint
    # of a violating (A, C) pair?
    gap_ab = np.where((A != B) & (B != C) & (A != C), gap, -np.inf)
    best_b = gap_ab.argmax(axis=1)               # (A, C)
    best_gap = gap_ab.max(axis=1)
    pair_valid = ~np.eye(n, dtype=bool)
    midpoint_counts = np.zeros(n)
    sel = pair_valid & (best_gap > TOL)
    np.add.at(midpoint_counts, best_b[sel], 1)
    return {
        "frac": float(viol.mean()),
        "gaps": gaps,
        "midpoint_counts": midpoint_counts,
        "n_pairs_violating": int(sel.sum()),
        "n_pairs": int(pair_valid.sum()),
    }


# %%
results = {}
for env_id in ENVS:
    for beta in BETAS:
        try:
            D, goals = d_matrix(env_id, beta)
        except Exception as e:  # noqa: BLE001
            print(f"{env_id} beta={beta}: skipped ({e})")
            continue
        st = triangle_stats(D)
        st["goals"] = goals
        results[(env_id, beta)] = st
        g = st["gaps"]
        print(f"{env_id:14s} beta={beta:<4} viol={st['frac']:6.1%}  "
              f"pairs-with-profitable-midpoint={st['n_pairs_violating']}/{st['n_pairs']}  "
              f"max gap={g.max():.3f} bits-equiv")

# %% [markdown]
# ## Figures: violation landscape + where the midpoints live

# %%
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=150)

# (1) violation fraction, env x beta
M = np.full((len(ENVS), len(BETAS)), np.nan)
for i, e in enumerate(ENVS):
    for j, b in enumerate(BETAS):
        if (e, b) in results:
            M[i, j] = results[(e, b)]["frac"]
im = axes[0].imshow(M, cmap="YlOrRd", vmin=0, aspect="auto")
axes[0].set_xticks(range(len(BETAS)), [f"$\\beta$={b}" for b in BETAS])
axes[0].set_yticks(range(len(ENVS)), ENVS)
for i in range(len(ENVS)):
    for j in range(len(BETAS)):
        if not np.isnan(M[i, j]):
            axes[0].text(j, i, f"{M[i, j]:.1%}", ha="center", va="center",
                         fontsize=10,
                         color="white" if M[i, j] > 0.5 * np.nanmax(M) else "black")
axes[0].set_title("violating fraction of ordered goal triples")
fig.colorbar(im, ax=axes[0], shrink=0.8)

# (2) gap distributions at beta = 1
for e in ENVS:
    if (e, 1.0) in results:
        g = results[(e, 1.0)]["gaps"]
        axes[1].hist(g, bins=80, histtype="step", label=e, density=True)
axes[1].axvline(0, color="black", lw=0.8)
axes[1].set_xlabel("gap = direct $-$ segmented cost  (positive = violation)")
axes[1].set_title("triple-gap distributions, $\\beta = 1$")
axes[1].legend(fontsize=8)

# (3) four_rooms midpoint map at beta = 1
key = ("four_rooms", 1.0)
if key in results:
    st = results[key]
    H, W = SHAPE
    grid = np.full(H * W, np.nan)
    for k, s in enumerate(st["goals"]):
        grid[s] = st["midpoint_counts"][k]
    im3 = axes[2].imshow(grid.reshape(H, W), cmap="viridis")
    axes[2].set_title("four_rooms: best profitable midpoint counts\n"
                      "(how often state B segments a route, $\\beta = 1$)")
    axes[2].set_xticks([]); axes[2].set_yticks([])
    fig.colorbar(im3, ax=axes[2], shrink=0.8)

fig.suptitle("Triangle-inequality violation in the free-energy geometry "
             f"(Cartesian identity, det={DET}, uniform prior)")
fig.tight_layout()
plt.show()

# %%
# top midpoints, four_rooms beta=1: are they the doorways?
if key in results:
    st = results[key]
    order = np.argsort(st["midpoint_counts"])[::-1]
    print("top profitable midpoints (state: count):")
    for k in order[:8]:
        s = st["goals"][k]
        print(f"  state {s:2d} (row {s // SHAPE[1]}, col {s % SHAPE[1]}): "
              f"{st['midpoint_counts'][k]:.0f}")

# %% [markdown]
# ## Discernments (2026-07-08 run, gridcore d432c4b)
#
# - **The curvature prediction lands on the first run.**  Flat worlds
#   (wrap_grid, open_grid) produce ZERO violating triples at every beta —
#   max gap is strictly negative — while walled worlds violate:
#   four_rooms 1.0–1.1% of triples, corr_1d_ring 2.8–3.8%.  Segmentation
#   is profitable exactly where walls concentrate curvature; on flat
#   geometry the pure infodesic, value geodesic, and cost infodesic
#   coincide and no policy switch ever pays.  This is the infodesic
#   paper's Figure-1 fact.
#
# - **Pairs, not triples, are the headline unit.**  The triple fraction
#   dilutes over all midpoints; per (A, C) PAIR, four_rooms has a
#   profitable midpoint for 248–266 of 1,560 ordered pairs (~16–17%):
#   one in six routes is cheaper segmented.
#
# - **The gap is informational.**  Max gap GROWS as beta falls
#   (four_rooms: 1.14 at beta=3, 3.43 at beta=1, 11.05 at beta=0.3) —
#   the value term alone would satisfy the triangle inequality; it is the
#   information cost of one policy serving a two-legged route that makes
#   segmentation pay, and parsimony (low beta) amplifies it.
#
# - **Midpoints sit AT or ONE STEP PAST the doorways, not on them
#   exclusively.**  Top four_rooms midpoints at beta=1: states 44 (6,2),
#   19 (2,5), 17 (2,3), 37 (5,2), 46 (6,4) — compare the doorway-salience
#   set {18, 30, 34, 45} from the salient-routing analysis.  37 and 44
#   flank the bottom-left doorway; 17/19 flank a top corridor cell.  The
#   profitable switch happens around door-crossing, plausibly AFTER
#   entering the destination room.  Follow-up for the paper: formal
#   overlap statistic between the midpoint distribution and the doorway
#   set +/- one step (experiment 2's switch-point analysis will test the
#   same prediction behaviourally).
#
# - Next: (i) the same measurement on the GA-best exemplar twist — does
#   the twist REDUCE the violating fraction (absorbing segmentation into
#   the labelling) or exploit it?  (ii) epsilon-infodesic gap version
#   (normalised by direct cost) for cross-env comparability.
