# %% [markdown]
# # Goal-geometry series, 06 — why the wrap-grid MDS looks like a cylinder
#
# Karen's question: the wrap_grid free-energy MDS resembles a cylinder.
# Hypothesis: the free-energy geometry of the untwisted torus is (near)
# flat and doubly periodic, so its distance matrix is that of a FLAT
# 2-TORUS.  A flat torus embeds isometrically not in R^3 but in R^4 as
# the Clifford torus S^1 x S^1: classical MDS should therefore show four
# leading eigenvalues in two nearly equal PAIRS (one pair per circle).
# A 3-D MDS keeps one full circle (two coordinates) plus half of the
# second (one coordinate) -- a tube: circle x interval = cylinder.  The
# fourth dimension would close the cylinder's open direction into the
# second circle.
#
# Test: eigen-spectrum of classical MDS on the symmetrised free-energy
# matrix; scatter of coordinate pairs coloured by row/column phase; a
# 3-D view showing the cylinder.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gridcore.bridge import EvalConfig, _state_dist_class, build_twisted_env_from_sigma
from gridcore.geometry import gower_decompose
from gridcore.info import DecisionInformation

SHAPE, DET, BETA, THETA = (7, 7), 0.97, 1.0, 1e-5
H, W = SHAPE
nS = H * W
IDENTITY = np.tile(np.arange(4, dtype=np.int64), (nS, 1))


def d_matrix(env_id, beta=BETA):
    def build(goal):
        cfg = EvalConfig(env_id=env_id, shape=SHAPE, goal=int(goal), beta=beta,
                         determinism=DET, manhattan=True, theta=THETA,
                         state_dist="uniform")
        return build_twisted_env_from_sigma(IDENTITY, cfg)
    env0 = build(0)
    goals = [int(s) for s in env0.available_states]
    n = len(goals)
    D = np.zeros((n, n))
    for j, g in enumerate(goals):
        env = build(g)
        di = DecisionInformation(env, _state_dist_class("uniform")(env), THETA,
                                 max_iterations=200_000, max_info_iterations=10_000)
        _, _, F = di.get_opt_policy_Z_free_vector(beta)
        assert di.converged
        D[:, j] = np.asarray(F, dtype=float)[goals]
    np.fill_diagonal(D, 0.0)
    return D, goals


D, goals = d_matrix("wrap_grid")
sym, skew = gower_decompose(D)
Dsym = (D + D.T) / 2.0
print(f"wrap_grid D: skew/sym ratio {np.linalg.norm(skew)/np.linalg.norm(sym):.4f} "
      "(near zero: the torus geometry is essentially symmetric)")

# %% [markdown]
# ## Classical MDS eigen-spectrum

# %%
n = Dsym.shape[0]
J = np.eye(n) - np.ones((n, n)) / n
B = -0.5 * J @ (Dsym ** 2) @ J
evals, evecs = np.linalg.eigh(B)
order = np.argsort(evals)[::-1]
evals, evecs = evals[order], evecs[:, order]
top = evals[:8]
print("top-8 MDS eigenvalues:", np.array2string(top, precision=2))
print(f"pair structure: l1/l2 = {top[0]/top[1]:.3f}, l3/l4 = {top[2]/top[3]:.3f}, "
      f"l2/l3 = {top[1]/top[2]:.3f}, l4/l5 = {top[3]/max(top[4],1e-9):.1f}")
X = evecs[:, :4] * np.sqrt(np.maximum(evals[:4], 0))

# %% [markdown]
# ## Do coordinate pairs trace the two circles?

# %%
rows = np.array([g // W for g in goals])
cols = np.array([g % W for g in goals])

fig = plt.figure(figsize=(15.5, 4.4), dpi=150)

ax0 = fig.add_subplot(1, 4, 1)
ax0.bar(range(1, 9), top, color=["#4477aa"] * 4 + ["#bbbbbb"] * 4)
ax0.set_title("MDS eigenvalues:\ntwo near-equal pairs = two circles")
ax0.set_xlabel("component")

ax1 = fig.add_subplot(1, 4, 2)
sc = ax1.scatter(X[:, 0], X[:, 1], c=cols, cmap="twilight", s=42)
ax1.set_title("coords (1,2) coloured by COLUMN:\ncircle no. 1")
ax1.set_aspect("equal")
fig.colorbar(sc, ax=ax1, shrink=0.75)

ax2 = fig.add_subplot(1, 4, 3)
sc = ax2.scatter(X[:, 2], X[:, 3], c=rows, cmap="twilight", s=42)
ax2.set_title("coords (3,4) coloured by ROW:\ncircle no. 2")
ax2.set_aspect("equal")
fig.colorbar(sc, ax=ax2, shrink=0.75)

ax3 = fig.add_subplot(1, 4, 4, projection="3d")
ax3.scatter(X[:, 0], X[:, 1], X[:, 2], c=rows, cmap="twilight", s=28)
ax3.set_title("3-D MDS (coords 1-3):\ncircle x interval = the cylinder")

fig.suptitle("wrap_grid free-energy geometry is a flat torus: Clifford torus in $R^4$, "
             "cylinder when truncated to $R^3$")
fig.tight_layout()
fig.savefig(Path(__file__).parent / "figs" / "F-wrap-mds-cylinder.png"
            if "__file__" in dir() else "figs/F-wrap-mds-cylinder.png",
            dpi=150, bbox_inches="tight")
plt.show()

# circle-quality stats
r12 = np.hypot(X[:, 0], X[:, 1])
r34 = np.hypot(X[:, 2], X[:, 3])
print(f"radius CV, coords (1,2): {r12.std()/r12.mean():.3f}  "
      f"coords (3,4): {r34.std()/r34.mean():.3f}  (0 = perfect circles)")

# %% [markdown]
# ## Aligning the degenerate eigenspace to the two circles
#
# The top four eigenvalues are EXACTLY equal (square torus symmetry), so
# the raw eigenvectors are an arbitrary rotation of the 4-space and mix
# the two circles.  Align by orthogonal Procrustes onto the known torus
# harmonics (cos/sin of row and column phase), then replot.

# %%
theta_c = 2 * np.pi * cols / W
theta_r = 2 * np.pi * rows / H
Y = np.stack([np.cos(theta_c), np.sin(theta_c),
              np.cos(theta_r), np.sin(theta_r)], axis=1)
Yc = Y - Y.mean(0)
U, _, Vt = np.linalg.svd(X.T @ Yc)
R = U @ Vt
Xa = X @ R

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), dpi=150)
sc = axes[0].scatter(Xa[:, 0], Xa[:, 1], c=cols, cmap="twilight", s=42)
axes[0].set_title("aligned coords (1,2) by column:\ncircle 1")
axes[0].set_aspect("equal"); fig.colorbar(sc, ax=axes[0], shrink=0.75)
sc = axes[1].scatter(Xa[:, 2], Xa[:, 3], c=rows, cmap="twilight", s=42)
axes[1].set_title("aligned coords (3,4) by row:\ncircle 2")
axes[1].set_aspect("equal"); fig.colorbar(sc, ax=axes[1], shrink=0.75)
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
ax3.scatter(Xa[:, 0], Xa[:, 1], Xa[:, 2], c=rows, cmap="twilight", s=28)
ax3.set_title("aligned 3-D truncation:\nthe cylinder")
fig.suptitle("Clifford-torus structure of the wrap_grid free-energy geometry "
             "(four equal eigenvalues; Procrustes-aligned)")
fig.tight_layout()
fig.savefig(Path(__file__).parent / "figs" / "F-wrap-mds-clifford-aligned.png"
            if "__file__" in dir() else "figs/F-wrap-mds-clifford-aligned.png",
            dpi=150, bbox_inches="tight")
plt.show()
r12 = np.hypot(Xa[:, 0], Xa[:, 1]); r34 = np.hypot(Xa[:, 2], Xa[:, 3])
print(f"aligned radius CV: (1,2) {r12.std()/r12.mean():.3f}  (3,4) {r34.std()/r34.mean():.3f}")

# %% [markdown]
# ## Discernments (2026-07-09)
#
# - **The cylinder is truncation, not structure.**  The wrap_grid
#   free-energy geometry at beta=1 is a flat square torus: classical MDS
#   gives FOUR exactly equal leading eigenvalues (557.46 x4; the next is
#   17x smaller) -- the signature of the Clifford torus S^1 x S^1 in R^4
#   with equal radii.  A 3-D embedding must discard one of four equal
#   coordinates: what remains is circle x interval -- a cylinder.  The
#   fourth dimension closes the cylinder's open direction into the
#   second circle.
#
# - The exact four-fold degeneracy also certifies flatness + squareness:
#   unequal torus aspect would split the pairs (two eigenvalues per
#   circle, different sizes); curvature would break the equality within
#   pairs.  This gives the infodesic paper a clean quantitative handle
#   for the curvature sweep: the eigenvalue SPLITTING of the top-4 MDS
#   spectrum measures how far a world's free-energy geometry is from the
#   flat torus.
#
# - After Procrustes alignment to the row/column harmonics the two
#   circles separate EXACTLY (radius CV 0.000 on both) -- at beta=1 the
#   free-energy geometry of the square wrap grid is a Clifford torus to
#   numerical precision; even det=0.97 does not perturb it.
