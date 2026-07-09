# %% [markdown]
# # Goal-geometry series, 09 — the switching-cost conjecture, tested
#
# The RSOS cognitive-geometry paper (Archer, Catenacci Volpi, Bröker,
# Polani 2022) closes its Future Work with an unproven conjecture: the
# free-energy triangle inequality, which policy switching breaks, can be
# restored by charging the switch —
#
#     F_C(A)  <=  F_B(A) + F_C(B) + C_switching
#
# and (rebuttal round 2, overall comment 4) C_switching is "most aptly
# modeled" NOT as a one-off per switch but as the CUMULATED cost of
# checking the currently active policy at every single decision,
# "weighted with a suitable information cost of the likelihood of having
# to use one policy over another in a particular state" — accruing
# exactly on states that belong to the support of two segments.  The
# supplementary theorem proves the inequality holds when supports are
# DISJOINT; overlap is the entire source of breakage.  A proof of the
# relaxed form was declared outstanding.  This notebook is the empirical
# test the conjecture never got.
#
# Machinery: notebook 04's D matrix gives gap(A,B,C) = D[A,C] -
# (D[A,B] + D[B,C]) — the measured violation.  New here: per-segment
# expected-visit counts nu(s) from the fundamental matrix of each goal
# policy's Markov chain, and three candidate cost models per triple:
#
#   C_oneoff = 1 bit                       (reviewer's model: one binary
#                                           decision at the switch state)
#   C_id     = sum_s (nu1+nu2)(s) H_b(w(s))    "check the active policy
#             every decision": w(s) = nu1/(nu1+nu2) is the likelihood a
#             decision at s belongs to segment 1 — the paper's literal
#             conjecture
#   C_act    = sum_s (nu1+nu2)(s) JS_w(pi_B(.|s), pi_C(.|s))
#             actionable identification: mutual information between the
#             segment variable and the ACTION at s.  You only pay to know
#             which segment you are in where the two policies disagree.
#             C_act <= C_id always (JS <= H_b); it vanishes on agreement
#             states — notebook 05's doorways, where the switch rate is
#             zero because the basins flow through.
#
# Units: in the free-energy Bellman recursion an informational cost of
# X bits enters F as X / beta (F accumulates cost + KL/beta).  The
# relaxed inequality is therefore tested as  gap <= C / beta.
#
# Hypotheses.
#   H1 (disjoint-support theorem, contrapositive): every violating triple
#      has nonzero support overlap.
#   H2 (the conjecture): the per-decision models cover the gap —
#      gap <= C_id/beta for all triples; ideally C_act already suffices.
#   H3 (reviewer's model fails): C_oneoff does not cover the large gaps,
#      confirming the rebuttal's rejection of one-off costing.
#   H4 (the violation IS the switching cost): among violating triples,
#      gap correlates with C_act/beta — the round-1 claim that "the
#      relaxation contribution ... reflects the informational switching
#      cost between the different segments".

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
ENVS = ["four_rooms", "corr_1d_ring", "open_grid"]
BETAS = [0.3, 1.0, 3.0]
TOL = 1e-6
nS = SHAPE[0] * SHAPE[1]
identity = np.tile(np.arange(4, dtype=np.int64), (nS, 1))


def entropy_bits(p, axis=-1):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log2(p), 0.0)
    return -t.sum(axis=axis)


def binary_entropy_bits(w):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(w > 0, w * np.log2(w), 0.0) + np.where(
            w < 1, (1 - w) * np.log2(1 - w), 0.0)
    return -t


def solve_env(env_id, beta):
    """Per goal: free energy column, optimal policy, expected-visit matrix.

    V[j][A_idx, s] = expected number of decisions taken AT state s en
    route from A to goal_j under goal_j's optimal policy (fundamental
    matrix of the policy chain with the goal absorbing; the goal itself
    counts no decision, the start state counts one).
    """
    def build(goal):
        cfg = EvalConfig(env_id=env_id, shape=SHAPE, goal=int(goal), beta=beta,
                         determinism=DET, manhattan=True, theta=THETA,
                         state_dist="uniform")
        return build_twisted_env_from_sigma(identity, cfg)

    env0 = build(0)
    goals = [int(s) for s in env0.available_states]
    n = len(goals)
    D = np.zeros((n, n))
    policies = np.zeros((n, nS, 4))
    V = np.zeros((n, n, nS))
    p_hat = np.zeros((n, 4))
    for j, g in enumerate(goals):
        env = build(g)
        di = DecisionInformation(env, _state_dist_class("uniform")(env), THETA,
                                 max_iterations=200_000,
                                 max_info_iterations=10_000)
        pi, _, F = di.get_opt_policy_Z_free_vector(beta)
        assert di.converged, (env_id, beta, g)
        D[:, j] = np.asarray(F, dtype=float)[goals]
        pi = np.asarray(pi, dtype=float)
        policies[j] = pi
        p_s = np.asarray(di.state_dist.set_ps(pi), dtype=float)
        p_hat[j] = pi.T @ p_s          # the converged action marginal (prior)
        T = np.asarray(env.get_T(), dtype=float).reshape(nS, 4, nS)
        P = np.einsum("sa,sat->st", pi, T)
        tr = [s for s in goals if s != g]
        N = np.linalg.inv(np.eye(len(tr)) - P[np.ix_(tr, tr)])
        tr_pos = {s: k for k, s in enumerate(tr)}
        for ai, A in enumerate(goals):
            if A != g:
                V[j, ai, tr] = N[tr_pos[A]]
    np.fill_diagonal(D, 0.0)
    return D, goals, policies, V, p_hat


def switching_costs(policies, V):
    """C_id, C_act, and overlap mass for every ordered triple (A, B, C).

    Segment 1: A -> B under pi_B, visits nu1 = V[B][A].
    Segment 2: B -> C under pi_C, visits nu2 = V[C][B].
    """
    n = V.shape[0]
    H_pol = entropy_bits(policies)                     # (n, nS)
    C_id = np.zeros((n, n, n))
    C_act = np.zeros((n, n, n))
    OV = np.zeros((n, n, n))
    for bk in range(n):
        nu1_all = V[bk]                                # (n, nS)
        piB = policies[bk]
        for ck in range(n):
            if ck == bk:
                continue
            nu2 = V[ck][bk]                            # (nS,)
            tot = nu1_all + nu2[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                w = np.where(tot > 0, nu1_all / tot, 0.0)
            C_id[:, bk, ck] = (tot * binary_entropy_bits(w)).sum(axis=1)
            mix = w[:, :, None] * piB[None] + (1 - w)[:, :, None] * policies[ck][None]
            js = entropy_bits(mix) - w * H_pol[bk][None] - (1 - w) * H_pol[ck][None]
            C_act[:, bk, ck] = (tot * np.clip(js, 0.0, None)).sum(axis=1)
            OV[:, bk, ck] = np.minimum(nu1_all, nu2[None, :]).sum(axis=1)
    return C_id, C_act, OV


def triple_gaps(D):
    """gap[A, B, C] = D[A, C] - (D[A, B] + D[B, C]) and the validity mask."""
    n = D.shape[0]
    gap = D[:, None, :] - (D[:, :, None] + D[None, :, :])
    A, B, C = np.ogrid[:n, :n, :n]
    valid = (A != B) & (B != C) & (A != C)
    return gap, valid


# %% [markdown]
# ## The sweep: does the relaxed inequality hold, and under which model?

# %%
results = {}
print(f"{'env':14s} {'beta':5s} {'viol':>6s}   "
      f"{'oneoff':>7s} {'C_act':>7s} {'C_id':>7s}   "
      f"{'max residual (act, id)':>24s}")
for env_id in ENVS:
    for beta in BETAS:
        D, goals, policies, V, p_hat = solve_env(env_id, beta)
        C_id, C_act, OV = switching_costs(policies, V)
        gap, valid = triple_gaps(D)
        viol = valid & (gap > TOL)
        nviol = int(viol.sum())
        res = {"D": D, "goals": goals, "gap": gap, "valid": valid,
               "viol": viol, "C_id": C_id, "C_act": C_act, "OV": OV,
               "policies": policies, "V": V, "p_hat": p_hat}
        if nviol:
            cover = {}
            for name, C in (("oneoff", np.ones_like(gap)),
                            ("act", C_act), ("id", C_id)):
                cover[name] = float((gap[viol] <= C[viol] / beta + TOL).mean())
            r_act = float(np.max(gap[viol] - C_act[viol] / beta))
            r_id = float(np.max(gap[viol] - C_id[viol] / beta))
            res["cover"] = cover
            print(f"{env_id:14s} {beta:<5} {nviol/valid.sum():6.1%}   "
                  f"{cover['oneoff']:7.1%} {cover['act']:7.1%} {cover['id']:7.1%}   "
                  f"{r_act:10.3f}, {r_id:10.3f}")
        else:
            print(f"{env_id:14s} {beta:<5} {0:6.1%}   (no violations: relaxed "
                  f"inequality vacuous)")
        results[(env_id, beta)] = res

# %% [markdown]
# ## H1: violations live on support overlap.  H4: gap tracks the cost.

# %%
for env_id in ENVS:
    for beta in BETAS:
        res = results[(env_id, beta)]
        viol, valid = res["viol"], res["valid"]
        if not viol.any():
            continue
        ov_v = res["OV"][viol]
        ov_ok = res["OV"][valid & ~viol]
        g = res["gap"][viol]
        c = res["C_act"][viol] / beta
        r = float(np.corrcoef(g, c)[0, 1])
        print(f"{env_id:14s} beta={beta:<4} min overlap on violating triples "
              f"{ov_v.min():8.4f} (median non-violating {np.median(ov_ok):7.4f})  "
              f"H4 corr(gap, C_act/beta) = {r:.3f}")

# %% [markdown]
# ## Figures

# %%
fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8), dpi=150)

# (1) the conjecture plot: gap vs C_act/beta, four_rooms, all betas
ax = axes[0]
colors = {0.3: "#cc3311", 1.0: "#4477aa", 3.0: "#228833"}
for beta in BETAS:
    res = results[("four_rooms", beta)]
    viol = res["viol"]
    ax.scatter(res["C_act"][viol] / beta, res["gap"][viol], s=6, alpha=0.35,
               color=colors[beta], label=f"$\\beta$={beta}")
lim = ax.get_xlim()[1]
ax.plot([0, lim], [0, lim], "--", color="black", lw=0.9)
ax.set_xlabel("$C_\\mathrm{act}/\\beta$  (actionable identification cost)")
ax.set_ylabel("gap = direct $-$ segmented free energy")
ax.set_title("violating triples vs the conjectured bound\n"
             "(below the line = relaxed inequality holds)")
ax.legend(fontsize=8)

# (2) coverage by model, four_rooms
ax = axes[1]
models = ["oneoff", "act", "id"]
width = 0.25
for mi, m in enumerate(models):
    vals = [results[("four_rooms", b)]["cover"][m] for b in BETAS]
    ax.bar(np.arange(len(BETAS)) + (mi - 1) * width, vals, width,
           label={"oneoff": "one-off (1 bit)", "act": "$C_\\mathrm{act}$ (JS)",
                  "id": "$C_\\mathrm{id}$ ($H_b$)"}[m])
ax.set_xticks(range(len(BETAS)), [f"$\\beta$={b}" for b in BETAS])
ax.set_ylim(0, 1.05)
ax.axhline(1.0, color="black", lw=0.6, ls=":")
ax.set_ylabel("fraction of violating triples covered")
ax.set_title("which cost model restores the\ntriangle inequality (four_rooms)")
ax.legend(fontsize=8)

# (3) where C_act accrues: per-state contribution over violating triples,
#     four_rooms beta=1 (second pass, targeted)
ax = axes[2]
res = results[("four_rooms", 1.0)]
policies, V, viol = res["policies"], res["V"], res["viol"]
n = V.shape[0]
H_pol = entropy_bits(policies)
contrib = np.zeros(nS)
for bk in range(n):
    nu1_all, piB = V[bk], policies[bk]
    for ck in range(n):
        if ck == bk or not viol[:, bk, ck].any():
            continue
        sel = viol[:, bk, ck]
        nu2 = V[ck][bk]
        tot = nu1_all[sel] + nu2[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(tot > 0, nu1_all[sel] / tot, 0.0)
        mix = w[:, :, None] * piB[None] + (1 - w)[:, :, None] * policies[ck][None]
        js = entropy_bits(mix) - w * H_pol[bk][None] - (1 - w) * H_pol[ck][None]
        contrib += (tot * np.clip(js, 0.0, None)).sum(axis=0)
grid = np.full(nS, np.nan)
grid[res["goals"]] = contrib[res["goals"]]
im = ax.imshow(grid.reshape(SHAPE), cmap="magma")
ax.set_title("where the switching cost accrues\n"
             "(summed $C_\\mathrm{act}$ contribution, violating triples, "
             "$\\beta$=1)", fontsize=10)
ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle("Testing the RSOS switching-cost conjecture: "
             f"gap $\\leq C_\\mathrm{{switching}}/\\beta$  "
             f"(Cartesian identity, det={DET})")
fig.tight_layout()
fig.savefig(Path(__file__).parent / "figs" / "F-switching-cost-conjecture.png"
            if "__file__" in dir() else "figs/F-switching-cost-conjecture.png",
            dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Tightness: how much slack does each model leave?

# %%
for beta in BETAS:
    res = results[("four_rooms", beta)]
    viol = res["viol"]
    if not viol.any():
        continue
    g = res["gap"][viol]
    for name in ("act", "id"):
        c = res["C_" + name][viol] / beta
        ratio = g / np.where(c > 0, c, np.nan)
        print(f"four_rooms beta={beta:<4} gap/(C_{name}/beta): "
              f"median {np.nanmedian(ratio):.3f}  "
              f"p90 {np.nanpercentile(ratio, 90):.3f}  "
              f"max {np.nanmax(ratio):.3f}")

# %% [markdown]
# ## Decomposition: is the violation switching cost or PRIOR SPECIALISATION?
#
# First-pass verdict (see the sweep table): the conjecture's literal
# reading FAILS — identification cost does not cover the gap, and the
# worst violations sit on triples whose segments barely overlap.  The
# alternative mechanism: each leg's free energy is computed against its
# own self-consistent action marginal, while the direct route must make
# ONE marginal serve the whole composite journey.  Segmentation's profit
# would then be prior specialisation, not any runtime cost of knowing
# which policy is active.
#
# Test: recompute leg 1 (A -> B) with the prior FIXED at the direct
# problem's converged marginal p_hat_C (notebook 03's log-domain
# fixed-reference solver, reference = the state-independent marginal).
# Leg 2 under p_hat_C is the direct solve's own fixed point, so it stays
# D[B,C] (asserted below).  Then
#
#     gap_shared(A,B,C) = D[A,C] - F_B^{p_hat_C}(A) - D[B,C]
#     PS(A,B,C)         = F_B^{p_hat_C}(A) - D[A,B]   >= 0
#     gap               = gap_shared + PS
#
# If sharing the prior kills the violations (gap_shared <= 0, or covered
# by C_id/beta), the triangle inequality's breakage is fully accounted
# for by prior specialisation and the "switching cost" story needs
# recasting.

# %%
REF_FLOOR = 1e-6


def free_energy_fixed_prior(env, prior, beta, theta=THETA,
                            max_iterations=200_000):
    """F for env's goal with the prior FIXED at a state-independent
    action marginal (log-domain; notebook-03 solver, tiled reference)."""
    T = np.asarray(env.get_T(), dtype=float)
    rs = np.asarray(env.get_rs(), dtype=float)
    n_states, n_actions = env.nS, env.nA
    walls = np.asarray(env.walls_flat, dtype=int)
    qf_const = (T * rs).sum(axis=1)
    ref = prior * (1 - REF_FLOOR) + REF_FLOOR / n_actions
    ref = ref / ref.sum()
    log_ref = np.log2(ref)[None, :]
    F = np.zeros(n_states)
    for _ in range(max_iterations):
        Q = (qf_const - T @ F).reshape(n_states, n_actions)
        logits = log_ref + beta * Q
        m = logits.max(axis=1, keepdims=True)
        log_Z = m.ravel() + np.log2(np.exp2(logits - m).sum(axis=1))
        if walls.size:
            log_Z[walls] = 0.0
        F_new = -log_Z / beta
        if np.max(np.abs(F_new - F)) < theta:
            return F_new, True
        F = F_new
    return F, False


def decompose(env_id, beta):
    res = results[(env_id, beta)]
    D, goals, p_hat = res["D"], res["goals"], res["p_hat"]
    n = len(goals)

    def build(goal):
        cfg = EvalConfig(env_id=env_id, shape=SHAPE, goal=int(goal), beta=beta,
                         determinism=DET, manhattan=True, theta=THETA,
                         state_dist="uniform")
        return build_twisted_env_from_sigma(identity, cfg)

    # sanity: leg 2 under the shared prior IS the direct solve
    fp_dev = 0.0
    for ck in range(0, n, max(n // 5, 1)):
        env = build(goals[ck])
        Fc, ok = free_energy_fixed_prior(env, p_hat[ck], beta)
        assert ok
        fp_dev = max(fp_dev, float(np.max(np.abs(Fc[goals] - D[:, ck]))))
    print(f"  fixed-point check (F^p_hat_C_C vs D[:,C]): max dev {fp_dev:.4f}")

    F_shared = np.zeros((n, n, n))          # (A, B, C)
    for bk in range(n):
        env_B = build(goals[bk])
        for ck in range(n):
            if ck == bk:
                continue
            Fs, ok = free_energy_fixed_prior(env_B, p_hat[ck], beta)
            assert ok, (env_id, beta, bk, ck)
            F_shared[:, bk, ck] = Fs[goals]
    PS = F_shared - res["D"][:, :, None]     # prior-specialisation benefit
    gap_shared = res["gap"] - PS
    return gap_shared, PS, fp_dev


# %%
DECOMP = [("four_rooms", b) for b in BETAS] + [("corr_1d_ring", 1.0)]
for env_id, beta in DECOMP:
    print(f"{env_id} beta={beta}:")
    gap_shared, PS, fp_dev = decompose(env_id, beta)
    res = results[(env_id, beta)]
    res["gap_shared"], res["PS"] = gap_shared, PS
    valid, viol, gap = res["valid"], res["viol"], res["gap"]
    # genuine survivors: above the solver's own fixed-point noise floor
    noise = max(2 * fp_dev, 10 * THETA)
    res["noise"] = noise
    viol_shared = valid & (gap_shared > noise)
    g, ps = gap[viol], PS[viol]
    r_ps = float(np.corrcoef(g, ps)[0, 1])
    print(f"  original violations {int(viol.sum())} -> shared-prior "
          f"violations above solver noise ({noise:.4f}): "
          f"{int(viol_shared.sum())}")
    print(f"  max gap_shared {float(gap_shared[valid].max()):.4f} "
          f"(was {float(gap[valid].max()):.3f});  min PS on violating "
          f"triples {float(ps.min()):.4f}")
    print(f"  H4-redux: corr(gap, PS) on violating triples = {r_ps:.3f}; "
          f"median PS share of gap "
          f"{float(np.median(ps / g)):.2f}")

# %% [markdown]
# ## Decomposition figure

# %%
fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8), dpi=150)

ax = axes[0]
for beta in BETAS:
    res = results[("four_rooms", beta)]
    viol = res["viol"]
    ps = np.clip(res["PS"][viol], 1e-4, None)
    ax.scatter(ps, res["gap"][viol], s=6, alpha=0.35,
               color=colors[beta], label=f"$\\beta$={beta}")
ax.set_xscale("log"); ax.set_yscale("log")
lo, hi = 1e-4, 2e2
ax.plot([lo, hi], [lo, hi], "--", color="black", lw=0.9)
ax.set_xlim(lo, hi); ax.set_ylim(1e-3, 2e1)
ax.set_xlabel("PS = prior-specialisation benefit of leg 1")
ax.set_ylabel("gap (violation size)")
ax.set_title("does prior specialisation explain the gap?\n"
             "(at/below the line = fully explained)")
ax.legend(fontsize=8)

ax = axes[1]
res = results[("four_rooms", 1.0)]
viol = res["viol"]
ax.hist(res["gap"][viol], bins=60, histtype="step", color="#4477aa",
        label="gap (own priors)")
ax.hist(res["gap_shared"][viol], bins=60, histtype="step", color="#cc3311",
        label="gap under shared prior")
ax.axvline(0, color="black", lw=0.8)
ax.set_yscale("log")
ax.set_xlabel("free-energy gap on originally-violating triples")
ax.set_title("four_rooms $\\beta$=1: sharing the prior\n"
             "collapses the violation")
ax.legend(fontsize=8)

ax = axes[2]
for beta in BETAS:
    res = results[("four_rooms", beta)]
    valid = res["valid"]
    frac0 = float(res["viol"].sum()) / float(valid.sum())
    frac1 = float((valid & (res["gap_shared"] > res["noise"])).sum()) \
        / float(valid.sum())
    ax.plot([0, 1], [frac0, frac1], "o-", color=colors[beta],
            label=f"$\\beta$={beta}")
ax.set_xticks([0, 1], ["own priors\n(as published)", "shared prior\n(PS removed)"])
ax.set_ylabel("violating fraction of ordered triples")
ax.set_title("violation rate before/after removing\nprior specialisation\n"
             "(shared-prior count above solver noise)")
ax.legend(fontsize=8)

fig.suptitle("Decomposing the triangle-inequality violation: "
             "switching cost vs prior specialisation")
fig.tight_layout()
fig.savefig(Path(__file__).parent / "figs" / "F-switching-cost-decomposition.png"
            if "__file__" in dir() else "figs/F-switching-cost-decomposition.png",
            dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Discernments (2026-07-09 run, gridcore d432c4b)
#
# - **The published conjecture fails its first empirical test.**  No
#   switching-cost model restores the triangle inequality: the literal
#   per-decision identification cost C_id covers only 17-88% of
#   violating triples (worse as beta rises), the actionable JS variant
#   covers 3-32%, and the reviewer's one-off bit covers the bulk
#   trivially but misses the entire tail (max residuals ~ the full gap).
#   Decisively, the WORST violations sit on triples whose segments
#   barely overlap (min overlap 0.011-0.025 expected visits, vs median
#   ~1-2 on non-violating triples), and gap does not correlate with any
#   identification cost (|r| < 0.12).  Whatever the violation is, it is
#   not the runtime cost of knowing which policy is active.
#
# - **It is prior specialisation, completely.**  Repricing leg 1 under
#   the direct problem's own converged action marginal (fixed-reference
#   log-domain solve, prior = p_hat_C) collapses EVERY violation to
#   solver noise: max gap_shared 0.000-0.001 vs original gaps of 1.14 /
#   3.43 / 11.05 (four_rooms beta 3/1/0.3) and 2.32 (corr_1d_ring
#   beta 1).  Median PS share of the gap is 1.07 (beta 1) and 1.01
#   (beta 3): the specialisation benefit IS the gap.  At beta 0.3 PS
#   overshoots (median share 2.04) — the mismatched prior hurts leg 1
#   more than segmentation gained, so gap_shared goes deeply negative.
#
# - **Empirical restatement: free energy with a SHARED action prior is
#   a quasimetric here, support overlap or not.**
#       F_C(A) <= F_B^{p_hat_C}(A) + F_C(B)
#   held for every ordered triple tested (three environments, three
#   betas, ~178k triples).  The supplementary theorem's advertised
#   assumption is disjoint supports, but its proof writes a single
#   p_hat(a) across both legs and the glued policy — the shared prior is
#   the load-bearing assumption, and empirically it needs no
#   disjointness at all in these worlds.  Conjecture for the paper: the
#   glueing argument goes through with overlap once the prior is fixed,
#   because the mixture-policy KL against a COMMON reference is convex;
#   the identification cost then bounds the glued policy's excess, and
#   here that excess never exceeded solver noise.
#
# - **Consequence for the RSOS Future-Work programme**: the missing term
#   in F_C(A) <= F_B(A) + F_C(B) + C is not a switching cost but a
#   PRIOR-MISMATCH credit: C(A,B,C) = F_B^{p_hat_C}(A) - F_B(A), i.e.
#   what leg 1 saves by abandoning the journey's marginal for its own.
#   This is computable (one fixed-reference solve), exactly zero when
#   the legs want the same marginal, and by construction tight.
#   "Switching cost" in the 2022 sense — checking the active policy —
#   is real but second-order: it never showed above solver noise once
#   priors were shared.
#
# - **Twist connection (for the infodesic paper's switching section)**:
#   segmentation pays exactly where route pieces want DIFFERENT action
#   marginals — walls create marginal-incompatible legs (flat worlds:
#   none).  A twist relabels per state, letting a SINGLE state-blind
#   marginal serve legs that are marginal-incompatible under Cartesian
#   labels; the notebook-04 result (the exemplar amplifies violations
#   2.4x while cutting mean cost) reads as the GA re-partitioning WHICH
#   leg-pairs are prior-compatible, in favour of its own habit legs.
#   Follow-up: compute PS under the exemplar twist — prediction: PS
#   concentrates on twist-incoherent leg pairs and vanishes along the
#   home-cycle flow.  Plus the beta-screen harvest: does evolving AT low
#   beta shrink Cartesian-frame PS (the twist absorbing the prior
#   specialisation the geometry affords)?
