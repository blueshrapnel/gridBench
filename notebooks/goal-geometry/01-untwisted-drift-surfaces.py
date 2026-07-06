# %% [markdown]
# # Goal-geometry series, 01 — drift surfaces of untwisted environments
#
# Thesis-chapter series: takes the space-of-goals MDS plots and improves
# the interpretation of the information cost of working against the
# prevailing policy (the goal-one-in-from-the-corner walkthrough of the
# space-of-goals paper).  Planned series:
#
#   01 (this) — surface survey: drift fields of UNTWISTED environments,
#        what is discernible, and the torus zero-drift control;
#   02 — the corner-goal case study quantified (drift projection vs the
#        paper's information-cost narrative);
#   03 — drift and the policy-switching cost (the skew part as the
#        antisymmetric component of the pairwise switching matrix);
#   04 — twisted worlds (circulation), beta families.
#
# ## How to read a drift vector
#
# D[i, j] = free energy of reaching goal j from state i = value cost +
# information cost of steering away from the prior.  The symmetric part
# is terrain (embedded by SMACOF); the skew part is current: the drift
# arrow at node i points toward goals that are cheap to reach FROM i
# relative to the return trip — i.e. downstream of the prevailing
# policy.  Goals upstream of the arrows cost information at every step.
#
# Control: an untwisted torus has full translation symmetry, hence no
# prevailing current, hence drift ~ 0.  If wrap_grid shows structure,
# the method is broken.

# %%
import subprocess
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

# %%
from gridcore.bridge import EvalConfig, _state_dist_class, build_twisted_env_from_sigma
from gridcore.geometry import (
    drift_vectors,
    embedding_diagnostics,
    goal_free_energy_matrix,
    gower_decompose,
    smacof_embed,
)
from gridcore.info import DecisionInformation

BETA, DET, THETA = 1.0, 0.97, 1e-5
ENVS = [
    ("wrap_grid", (9, 9)),    # the zero-drift control (torus)
    ("open_grid", (9, 9)),
    ("four_rooms", (9, 9)),
    ("plus_cross", (9, 9)),
    ("x_wall", (9, 9)),
    ("pinwheel", (9, 9)),
]


def untwisted_D(env_id, shape):
    cfg0 = EvalConfig(env_id=env_id, shape=shape, goal=0, beta=BETA, determinism=DET,
                      manhattan=True, theta=THETA, state_dist="uniform")
    nS = shape[0] * shape[1]
    identity = np.tile(np.arange(4, dtype=np.int64), (nS, 1))
    env0 = build_twisted_env_from_sigma(identity, cfg0)
    goals = [int(s) for s in env0.available_states]

    def builder(goal):
        cfg = EvalConfig(env_id=env_id, shape=shape, goal=int(goal), beta=BETA,
                         determinism=DET, manhattan=True, theta=THETA,
                         state_dist="uniform")
        return build_twisted_env_from_sigma(identity, cfg)

    def solve(env):
        di = DecisionInformation(env, _state_dist_class("uniform")(env), THETA,
                                 max_iterations=200_000, max_info_iterations=10_000)
        _, _, F = di.get_opt_policy_Z_free_vector(BETA)
        return F

    return goal_free_energy_matrix(builder, goals, solve), goals

# %%
from gridvis.goalspace import plot_drift_mds

results = {}
fig, axes = plt.subplots(2, 3, figsize=(16.8, 11.2), dpi=150)
for ax, (env_id, shape) in zip(axes.ravel(), ENVS):
    D, goals = untwisted_D(env_id, shape)
    sym, skew = gower_decompose(D)
    coords, stress = smacof_embed(sym, components=2)
    drift = drift_vectors(coords, skew)
    diag = embedding_diagnostics(sym)
    skew_ratio = float(np.linalg.norm(skew) / np.linalg.norm(sym))
    results[env_id] = dict(D=D, goals=goals, sym=sym, skew=skew, coords=coords,
                           drift=drift, stress=stress, skew_ratio=skew_ratio, diag=diag)
    plot_drift_mds(ax, coords, drift,
                   node_values=[g // shape[1] for g in goals],
                   title=env_id, stress=stress, skew_ratio=skew_ratio)
fig.suptitle("Drift surfaces of untwisted environments "
             f"($\\beta={BETA}$, det $={DET}$, uniform prior, identity $\\sigma$)",
             fontsize=13)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Quantitative comparison

# %%
print(f"{'env':12s} {'skew/sym':>9s} {'mean|drift|':>12s} {'max|drift|':>11s} "
      f"{'stress':>7s} {'neg-eig':>8s}")
for env_id, r in results.items():
    mag = np.linalg.norm(r["drift"], axis=1)
    print(f"{env_id:12s} {r['skew_ratio']:9.4f} {mag.mean():12.4f} {mag.max():11.4f} "
          f"{r['stress']:7.3f} {r['diag']['negative_eigen_mass']:8.3f}")

# %% [markdown]
# ## Where does the drift point?  (grid-space rendering)
#
# The MDS panels show drift in embedding coordinates; this view maps the
# same skew field back onto the grid: for each goal-state, the arrow is
# the drift vector re-expressed against grid positions of the other
# goals (unit vectors in GRID space, not embedding space), so "toward
# the corner" is literal.

# %%
fig2, axes2 = plt.subplots(2, 3, figsize=(15.6, 10.4), dpi=150)
for ax, (env_id, shape) in zip(axes2.ravel(), ENVS):
    r = results[env_id]
    goals = r["goals"]
    pos = np.array([(g % shape[1], -(g // shape[1])) for g in goals], dtype=float)
    grid_drift = drift_vectors(pos, r["skew"])
    mag = np.linalg.norm(grid_drift, axis=1)
    # Noise floor: quiver AUTOSCALES, so float round-off (the torus's
    # exact-zero field is ~1e-15) would render as confident arrows.
    # Suppress arrows below an absolute magnitude threshold.
    keep = mag > 1e-6
    ax.scatter(pos[:, 0], pos[:, 1], c=mag, cmap="magma", s=110,
               edgecolors="white", linewidths=0.5)
    if keep.any():
        ax.quiver(pos[keep, 0], pos[keep, 1],
                  grid_drift[keep, 0], grid_drift[keep, 1],
                  angles="xy", color="#0077bb", width=0.005)
    else:
        ax.annotate("drift = 0 (torus control)", (0.5, 0.5),
                    xycoords="axes fraction", ha="center", fontsize=11)
    ax.set_title(f"{env_id} — grid-space drift (colour = |drift|)", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
fig2.suptitle("The same skew fields on the grid: which way does the current flow?",
              fontsize=13)
fig2.tight_layout()
plt.show()

# %% [markdown]
# ## Discernments (2026-07-06 run, gridcore d432c4b)
#
# - **The torus control is exact**: wrap_grid skew/sym = 0.0000 and every
#   drift vector is zero to round-off.  With no boundary there is no
#   prevailing current — the construction passes its falsification test.
#   (Rendering caveat learned here: quiver autoscale had amplified the
#   1e-15 noise into confident-looking arrows; hence the noise floor.)
# - **Drift is a boundary phenomenon in untwisted worlds**: in every
#   walled/bounded environment the interior states have near-zero drift
#   and the current flows outward, into corners, arm tips, and room
#   corners — the low-option pockets.  This is the space-of-goals
#   corner-goal story made geometric: those pockets are downstream of
#   the uniform prior (many paths fall in, informative actions are
#   needed to get out), so goals there are cheap to reach and
#   information-expensive to leave.
# - **Magnitude ranks by boundary exposure, not wall count**: open_grid
#   (0.150) > plus_cross (0.128) > four_rooms (0.096) > pinwheel (0.050)
#   > x_wall (0.046) > wrap (0).  The open grid — no internal walls at
#   all — carries the strongest current, because ALL of its structure is
#   boundary.  Internal walls mostly partition the terrain (symmetric
#   part); the boundary drives the current (skew part).
# - Next in series: 02 quantifies the corner-goal case (drift projection
#   vs the paper's information-cost walkthrough); 03 tests whether the
#   skew term predicts the antisymmetric component of pairwise
#   policy-switching costs.
