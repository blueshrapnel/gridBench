# %% [markdown]
# # Goal-geometry series, 03 — drift vs the policy-switching cost
#
# The claim from 01/02: the skew part of the uniform-prior free-energy
# matrix — rendered as drift — is the geometric object behind the cost
# of switching goals against the prevailing policy.  This notebook makes
# the switching cost concrete and tests whether drift predicts it.
#
# **Definition.**  The agent sits at goal A having just achieved it; its
# policy is still pi_A (the goal-A optimal policy).  The cost of
# switching to B is the free energy of reaching B computed with pi_A as
# the REFERENCE policy: the information term charges KL(pi || pi_A)
# per state — deviation from what you are currently doing, not from a
# neutral prior:
#
#     SC(A -> B) = F_B^{ref=pi_A}(A)
#
# **Hypotheses.**
#   H1: the antisymmetric part of SC correlates strongly with the skew
#       of the cheap uniform-prior matrix D (drift predicts switching
#       asymmetry without computing any reference-policy solves);
#   H2: the policy-aware asymmetry is LARGER than the uniform-prior skew
#       (pi_A is more opinionated than the uniform prior);
#   H3: per-goal mean switching asymmetry reproduces the sink field.
#
# **Method note.**  The fixed-reference solver iterates
#     pi(a|s) prop ref(a|s) * 2^(beta Q_F(s,a)),   F = -log2(Z)/beta
# with no marginal update (the reference is fixed), implemented in
# LOG-DOMAIN — incidentally the prototype of the low-beta formulation
# gridCore#1 needs.  References are floored (mix 1e-6 uniform) so
# near-deterministic pi_A rows cannot forbid actions outright.

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
from gridcore.geometry import drift_vectors, gower_decompose
from gridcore.info import DecisionInformation

ENV, SHAPE, DET, BETA, THETA = "four_rooms", (9, 9), 0.97, 1.0, 1e-5
W = SHAPE[1]
nS = SHAPE[0] * SHAPE[1]
identity = np.tile(np.arange(4, dtype=np.int64), (nS, 1))


def build_env(goal):
    cfg = EvalConfig(env_id=ENV, shape=SHAPE, goal=int(goal), beta=BETA,
                     determinism=DET, manhattan=True, theta=THETA,
                     state_dist="uniform")
    return build_twisted_env_from_sigma(identity, cfg)


# uniform-prior solves: D matrix AND each goal's optimal policy pi_g
env0 = build_env(0)
goals = [int(s) for s in env0.available_states]
n = len(goals)
gindex = {g: k for k, g in enumerate(goals)}

D = np.zeros((n, n))
policies = {}
for j, g in enumerate(goals):
    env = build_env(g)
    di = DecisionInformation(env, _state_dist_class("uniform")(env), THETA,
                             max_iterations=200_000, max_info_iterations=10_000)
    pi, _, F = di.get_opt_policy_Z_free_vector(BETA)
    D[:, j] = F[goals]
    policies[g] = np.asarray(pi, dtype=float)
np.fill_diagonal(D, 0.0)
sym_D, skew_D = gower_decompose(D)
print(f"uniform-prior matrix done: {n} goals, skew/sym "
      f"{np.linalg.norm(skew_D)/np.linalg.norm(sym_D):.3f}")

# %% [markdown]
# ## The fixed-reference (log-domain) solver

# %%
REF_FLOOR = 1e-6


def free_energy_with_reference(env, ref_policy, beta=BETA, theta=THETA,
                               max_iterations=200_000):
    """F for env's goal with a FIXED reference policy (log-domain)."""
    T = np.asarray(env.get_T(), dtype=float)          # (nS*nA, nS)
    rs = np.asarray(env.get_rs(), dtype=float)
    n_states, n_actions = env.nS, env.nA
    walls = np.asarray(env.walls_flat, dtype=int)
    qf_const = (T * rs).sum(axis=1)

    ref = ref_policy * (1 - REF_FLOOR) + REF_FLOOR / n_actions
    ref = ref / ref.sum(axis=1, keepdims=True)
    log_ref = np.log2(ref)

    F = np.zeros(n_states)
    for it in range(max_iterations):
        Q = (qf_const - T @ F).reshape(n_states, n_actions)
        logits = log_ref + beta * Q
        m = logits.max(axis=1, keepdims=True)
        log_Z = m.ravel() + np.log2(np.exp2(logits - m).sum(axis=1))
        if walls.size:
            log_Z[walls] = 0.0
        F_new = -log_Z / beta
        if np.max(np.abs(F_new - F)) < theta:
            return F_new, it + 1, True
        F = F_new
    return F, max_iterations, False


# %% [markdown]
# ## The switching-cost matrix SC(A -> B)

# %%
SC = np.zeros((n, n))
iters_total = 0
for a_k, A in enumerate(goals):
    ref = policies[A]
    for b_k, B in enumerate(goals):
        if A == B:
            continue
        env = build_env(B)
        F, its, ok = free_energy_with_reference(env, ref)
        assert ok, (A, B)
        SC[a_k, b_k] = F[A]
        iters_total += its
print(f"SC matrix done: {n*(n-1)} ordered pairs, "
      f"{iters_total/(n*(n-1)):.0f} iters/solve mean")
sym_SC, skew_SC = gower_decompose(SC)

# %% [markdown]
# ## H1-H3: does drift predict the switching asymmetry?

# %%
iu = np.triu_indices(n, k=1)
x = skew_D[iu]
y = skew_SC[iu]
r = float(np.corrcoef(x, y)[0, 1])
slope = float(np.polyfit(x, y, 1)[0])
print(f"H1  corr(skew_D, skew_SC) = {r:.3f}, slope = {slope:.2f}")
print(f"H2  ||skew_SC|| / ||skew_D|| = {np.linalg.norm(skew_SC)/np.linalg.norm(skew_D):.2f} "
      f"(policy-aware asymmetry vs uniform-prior skew)")
sink_D = -skew_D.mean(axis=0)
sink_SC = -skew_SC.mean(axis=0)
r_sink = float(np.corrcoef(sink_D, sink_SC)[0, 1])
print(f"H3  corr(sink field from D, sink field from SC) = {r_sink:.3f}")

# drift projection: does the embedding-free grid drift predict pairwise skew_SC?
pos = np.array([(g % W, -(g // W)) for g in goals], dtype=float)
grid_drift = drift_vectors(pos, skew_D)
diff = pos[None, :, :] - pos[:, None, :]
dist = np.sqrt((diff**2).sum(-1))
with np.errstate(divide="ignore", invalid="ignore"):
    unit = np.where(dist[..., None] > 0, diff / dist[..., None], 0.0)
proj = ((grid_drift[:, None, :] - grid_drift[None, :, :]) * unit).sum(-1) / 2
r_proj = float(np.corrcoef(proj[iu], y)[0, 1])
print(f"H1b corr(drift-projection difference, skew_SC) = {r_proj:.3f}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.9), dpi=150)
axes[0].scatter(x, y, s=8, alpha=0.4, color="#4477aa")
axes[0].set_xlabel("skew of uniform-prior D [bits]")
axes[0].set_ylabel("antisymmetric part of SC [bits]")
axes[0].set_title(f"H1: r = {r:.3f}, slope = {slope:.2f}")
axes[1].scatter(proj[iu], y, s=8, alpha=0.4, color="#228833")
axes[1].set_xlabel("drift-projection difference (grid space)")
axes[1].set_ylabel("antisymmetric part of SC [bits]")
axes[1].set_title(f"H1b: r = {r_proj:.3f}")
axes[2].scatter(sink_D, sink_SC, s=22, color="#cc3311")
axes[2].set_xlabel("sink strength from D [bits]")
axes[2].set_ylabel("mean switching asymmetry per goal [bits]")
axes[2].set_title(f"H3: r = {r_sink:.3f}")
fig.suptitle("Drift (cheap, uniform-prior) as a predictor of the policy-aware "
             f"switching cost ({ENV} {SHAPE[0]}x{SHAPE[1]}, $\\beta={BETA}$)",
             fontsize=12)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Discernments (2026-07-06 run, gridcore d432c4b)
#
# - **H1 partial** (r = 0.52, slope 7.3): the ambient drift predicts a
#   real but minority share (~27% of variance) of the policy-aware
#   switching asymmetry.  **H2 emphatic**: the policy-aware asymmetry is
#   14.5x LARGER than the uniform-prior skew — the prevailing policy,
#   not the environment's ambient current, dominates the cost of
#   switching.  **H3 moderate** (r = 0.68): the sink field survives the
#   change of reference in shape.  H1b weak (0.26): projecting drift
#   into grid space loses most of the pairwise signal.
#
# - **The better model this suggests**: decompose switching asymmetry
#   into an AMBIENT term (the environment's current — what drift shows,
#   goal-independent, cheap: one uniform-prior matrix) plus a
#   POLICY-SPECIFIC term (disagreement between pi_A's flow and the
#   A->B route system — the dominant part, requiring the reference
#   solves).  Drift is the floor every switch pays; the policy term is
#   the price of the particular habit being abandoned.  Notebook 04
#   candidate: characterise the residual skew_SC - 7.3*skew_D field.
#
# - **Log-domain first data point**: the fixed-reference solver (no
#   marginal update, logits in log2 space) converged in 84 iters/solve
#   mean — same order as the marginal-prior kernel (~90).  The low-beta
#   formulation costs nothing at beta=1; its value is where float64
#   linear-domain fails.
