# %% [markdown]
# # Goal-geometry series, 07 — what the twist does to the space-of-goals MDS
#
# Karen's hypothesis: a literally free state-goal combination should
# reduce its edge on the MDS graph to zero.  Refinement: free energy
# always carries the path-length term, so full-cost edges shrink but stay
# positive; the DECISION-INFORMATION matrix is where "literally free"
# means a literally zero edge.  So we compare two matrices per world:
#
#   D_F[s, g]    = F_g(s)      (full cost: value + information)
#   D_I[s, g]    = I_D^{pi_g}(s)  (information only)
#
# each under the Cartesian identity and under the paper-1 exemplar twist
# (four_rooms 7x7).  Predictions: under the twist the home-cycle column
# of D_I collapses toward zero from EVERYWHERE -- the home becomes a
# wormhole: near every state informationally, while ordinary pairs stay
# apart.  A wormhole point is exactly what a flat map cannot draw, so the
# twist should also make the geometry LESS embeddable (more negative
# eigen-mass, higher stress) -- the MDS distortion is itself evidence of
# the shortcut structure.

# %%
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gridcore.bridge import EvalConfig, _state_dist_class, build_twisted_env_from_sigma
from gridcore.info import DecisionInformation

SHAPE, DET, BETA, THETA = (7, 7), 0.97, 1.0, 1e-5
H, W = SHAPE
nS = H * W
IDENTITY = np.tile(np.arange(4, dtype=np.int64), (nS, 1))
SCHEMA_EXPORT = Path("/media/merlin/grid-twist/data-schema-10/multi/schema10-export")
EXEMPLAR_HASH_PREFIX = "0e5cb0bf"


def load_exemplar_sigma():
    root = SCHEMA_EXPORT / "shape=7x7" / "env_id=four_rooms"
    for p in sorted(root.rglob("*.pickle")):
        blob = pickle.load(open(p, "rb"))
        prov = blob.get("provenance", {}) if isinstance(blob.get("provenance"), dict) else {}
        if str(prov.get("sigma_hash", "")).startswith(EXEMPLAR_HASH_PREFIX):
            return np.asarray(blob["sigma"], dtype=int)
    raise FileNotFoundError(EXEMPLAR_HASH_PREFIX)


def matrices(sigma):
    """D_F and D_I over available states for four_rooms."""
    def build(goal):
        cfg = EvalConfig(env_id="four_rooms", shape=SHAPE, goal=int(goal),
                         beta=BETA, determinism=DET, manhattan=True,
                         theta=THETA, state_dist="uniform")
        return build_twisted_env_from_sigma(sigma, cfg)
    env0 = build(0)
    goals = [int(s) for s in env0.available_states]
    n = len(goals)
    DF = np.zeros((n, n))
    DI = np.zeros((n, n))
    for j, g in enumerate(goals):
        env = build(g)
        di = DecisionInformation(env, _state_dist_class("uniform")(env), THETA,
                                 max_iterations=200_000,
                                 max_info_iterations=10_000)
        pi, _, F = di.get_opt_policy_Z_free_vector(BETA)
        info = di.get_decision_information_given_policy(pi)
        DF[:, j] = np.asarray(F, dtype=float)[goals]
        DI[:, j] = np.asarray(info, dtype=float)[goals]
    np.fill_diagonal(DF, 0.0)
    np.fill_diagonal(DI, 0.0)
    return DF, DI, goals


def classical_mds(D, k=2):
    Ds = (D + D.T) / 2.0
    n = Ds.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (Ds ** 2) @ J
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    X = evecs[:, :k] * np.sqrt(np.maximum(evals[:k], 0))
    neg_mass = float(np.abs(evals[evals < 0]).sum() / np.abs(evals).sum())
    return X, evals, neg_mass


sigma_ex = load_exemplar_sigma()
DF_c, DI_c, goals = matrices(IDENTITY)
DF_t, DI_t, _ = matrices(sigma_ex)

# home cycle of the exemplar: dominant-label attractor cells (from the
# twisted kernel's intended-move map, dominant label = most policy use --
# here simply take the goal column with the smallest mean D_I under the
# twist as "home region" proxy, plus report the top few)
mean_di_t = DI_t.mean(axis=0)
home_idx = np.argsort(mean_di_t)[:10]
home_states = [goals[k] for k in home_idx]
print("cheapest-to-reach goals under the twist (info-only, mean bits):")
for k in home_idx[:5]:
    print(f"  state {goals[k]:2d} (row {goals[k]//W}, col {goals[k]%W}): "
          f"{mean_di_t[k]:.3f} bits  (Cartesian: {DI_c.mean(axis=0)[k]:.3f})")

# %% [markdown]
# ## The edge test: distances to the home goals, Cartesian vs twist

# %%
best = home_idx[0]
print(f"\nedge collapse, goal state {goals[best]}:")
print(f"  D_I column mean:  Cartesian {DI_c[:, best].mean():.3f} bits  "
      f"-> twist {DI_t[:, best].mean():.3f} bits")
print(f"  D_I column max:   Cartesian {DI_c[:, best].max():.3f}  "
      f"-> twist {DI_t[:, best].max():.3f}")
print(f"  D_F column mean:  Cartesian {DF_c[:, best].mean():.2f}  "
      f"-> twist {DF_t[:, best].mean():.2f}  (path term survives)")

# %% [markdown]
# ## MDS embeddings and their distortion

# %%
fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6), dpi=150)
rows_arr = np.array([g // W for g in goals])

conds = [("Cartesian, full cost $D_F$", DF_c), ("twist, full cost $D_F$", DF_t),
         ("edge histogram ($D_I$ to home)", None),
         ("Cartesian, info only $D_I$", DI_c), ("twist, info only $D_I$", DI_t),
         ("eigen-spectra", None)]

for ax, (ttl, D) in zip(axes.flat, conds):
    if D is None:
        continue
    X, evals, neg = classical_mds(D)
    sc = ax.scatter(X[:, 0], X[:, 1], c=rows_arr, cmap="viridis", s=40)
    ax.scatter(X[home_idx[:5], 0], X[home_idx[:5], 1], marker="*", s=240,
               facecolor="#ffcc00", edgecolor="black", zorder=5,
               label="cheapest goals (home)")
    ax.set_title(f"{ttl}\nnegative eigen-mass {neg:.1%}", fontsize=10)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")

ax = axes[0, 2]
bins = np.linspace(0, max(DI_c[:, best].max(), DI_t[:, best].max()) * 1.05, 30)
ax.hist(DI_c[:, best], bins=bins, alpha=0.65, label="Cartesian", color="#4477aa")
ax.hist(DI_t[:, best], bins=bins, alpha=0.65, label="twist", color="#ee6677")
ax.set_title(f"edges to home goal {goals[best]}: information cost\n"
             "(the twist drives them toward zero)", fontsize=10)
ax.set_xlabel("$D_I[s, \\mathrm{home}]$ (bits)")
ax.legend(fontsize=8)

ax = axes[1, 2]
for D, lab, colr in ((DF_c, "Cartesian $D_F$", "#4477aa"),
                     (DF_t, "twist $D_F$", "#ee6677"),
                     (DI_c, "Cartesian $D_I$", "#88bbdd"),
                     (DI_t, "twist $D_I$", "#ffaabb")):
    _, evals, neg = classical_mds(D)
    ax.plot(range(1, 11), evals[:10], "o-", ms=4, label=f"{lab} (neg {neg:.0%})",
            color=colr)
ax.axhline(0, color="black", lw=0.7)
ax.set_title("MDS eigen-spectra: the twist geometry\nis less flat-map-able",
             fontsize=10)
ax.set_xlabel("component")
ax.legend(fontsize=7)

fig.suptitle("What the twist does to the space-of-goals geometry (four_rooms 7x7, "
             "$\\beta = 1$): home edges collapse, wormholes resist flat maps")
fig.tight_layout()
fig.savefig(Path(__file__).parent / "figs" / "F-twist-mds-wormholes.png"
            if "__file__" in dir() else "figs/F-twist-mds-wormholes.png",
            dpi=150, bbox_inches="tight")
plt.show()
