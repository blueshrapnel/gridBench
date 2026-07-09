# %% [markdown]
# # Goal-geometry series, 10 — prior specialisation under the exemplar twist
#
# Notebook 09 settled the Cartesian question: the free-energy triangle
# violation is prior specialisation (PS), completely — repricing leg 1
# under the direct problem's converged action marginal collapses every
# violation to solver noise, and no identification/switching cost is
# needed.  Notebook 04 left the twist-side puzzle: the GA exemplar
# AMPLIFIES violations 2.4x while lowering mean cost.  This notebook
# computes PS in the twisted frame and asks where it lives.
#
# Hypotheses.
#   H1 (frame independence): the shared-prior quasimetric property
#      survives the twist — gap_shared collapses to solver noise under
#      sigma too.  The twist bends WHICH legs are marginal-compatible,
#      not the accounting.
#   H2 (amplification = marginal divergence): the 2.4x violation
#      amplification is carried by MORE DIVERGENT per-goal marginals —
#      pairwise KL(p_hat_B || p_hat_C) grows under the twist relative to
#      Cartesian.
#   H3 (the workbench section-7 formula): the first-order estimate
#          PS_est(A,B,C) = (1/beta) * sum_s nu1(s) *
#                          E_{pi_B(.|s)}[log2 p_hat_B(a)/p_hat_C(a)]
#      (swap the reference prior, keep the policy) upper-bounds PS and
#      predicts it — the empirical face of C_switch = n * KL(marginals).
#   H4 (nb09's prediction): PS vanishes along the home-cycle flow —
#      goal pairs whose marginals both concentrate on the twist's
#      dominant label are marginal-compatible, so their legs carry
#      little specialisation benefit; PS concentrates on twist-
#      incoherent leg pairs.
#
# Cartesian reference numbers (nb09, four_rooms 7x7): viol 1.0-1.1% of
# triples, max gap 1.14 / 3.43 / 11.05 at beta 3 / 1 / 0.3, shared-prior
# survivors 0 at every beta.

# %%
import pickle
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

ENV, SHAPE, DET, THETA = "four_rooms", (7, 7), 0.97, 1e-5
BETAS = [0.3, 1.0, 3.0]
TOL = 1e-6
REF_FLOOR = 1e-6
nS = SHAPE[0] * SHAPE[1]
identity = np.tile(np.arange(4, dtype=np.int64), (nS, 1))

EXEMPLAR_HASH_PREFIX = "0e5cb0bf"   # paper-1 four_rooms exemplar (cov 0.85)
SCHEMA_EXPORT = Path("/media/merlin/grid-twist/data-schema-10/multi/schema10-export")


def load_exemplar_sigma():
    root = SCHEMA_EXPORT / "shape=7x7" / "env_id=four_rooms"
    for p in sorted(root.rglob("*.pickle")):
        blob = pickle.load(open(p, "rb"))
        prov = blob.get("provenance", {}) if isinstance(blob.get("provenance"), dict) else {}
        if str(prov.get("sigma_hash", "")).startswith(EXEMPLAR_HASH_PREFIX):
            return np.asarray(blob["sigma"], dtype=int)
    raise FileNotFoundError(EXEMPLAR_HASH_PREFIX)


def build(sigma, goal, beta):
    cfg = EvalConfig(env_id=ENV, shape=SHAPE, goal=int(goal), beta=beta,
                     determinism=DET, manhattan=True, theta=THETA,
                     state_dist="uniform")
    return build_twisted_env_from_sigma(sigma, cfg)


def solve_frame(sigma, beta):
    """Per goal: F column, policy, visit matrix, converged marginal."""
    env0 = build(sigma, 0, beta)
    goals = [int(s) for s in env0.available_states]
    n = len(goals)
    D = np.zeros((n, n))
    policies = np.zeros((n, nS, 4))
    V = np.zeros((n, n, nS))
    p_hat = np.zeros((n, 4))
    for j, g in enumerate(goals):
        env = build(sigma, g, beta)
        di = DecisionInformation(env, _state_dist_class("uniform")(env), THETA,
                                 max_iterations=200_000,
                                 max_info_iterations=10_000)
        pi, _, F = di.get_opt_policy_Z_free_vector(beta)
        assert di.converged, (beta, g)
        D[:, j] = np.asarray(F, dtype=float)[goals]
        pi = np.asarray(pi, dtype=float)
        policies[j] = pi
        p_s = np.asarray(di.state_dist.set_ps(pi), dtype=float)
        p_hat[j] = pi.T @ p_s
        T = np.asarray(env.get_T(), dtype=float).reshape(nS, 4, nS)
        P = np.einsum("sa,sat->st", pi, T)
        tr = [s for s in goals if s != g]
        N = np.linalg.inv(np.eye(len(tr)) - P[np.ix_(tr, tr)])
        tr_pos = {s: k for k, s in enumerate(tr)}
        for ai, A in enumerate(goals):
            if A != g:
                V[j, ai, tr] = N[tr_pos[A]]
    np.fill_diagonal(D, 0.0)
    return {"D": D, "goals": goals, "policies": policies, "V": V,
            "p_hat": p_hat}


def free_energy_fixed_prior(env, prior, beta, theta=THETA,
                            max_iterations=200_000):
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


def decompose_frame(sigma, beta, fr):
    """PS + gap_shared + the section-7 first-order estimate."""
    D, goals, p_hat = fr["D"], fr["goals"], fr["p_hat"]
    policies, V = fr["policies"], fr["V"]
    n = len(goals)

    ref = p_hat * (1 - REF_FLOOR) + REF_FLOOR / 4
    ref = ref / ref.sum(axis=1, keepdims=True)
    log_ref = np.log2(ref)                       # (n, 4)

    fp_dev = 0.0
    for ck in range(0, n, max(n // 5, 1)):
        env = build(sigma, goals[ck], beta)
        Fc, ok = free_energy_fixed_prior(env, p_hat[ck], beta)
        assert ok
        fp_dev = max(fp_dev, float(np.max(np.abs(Fc[goals] - D[:, ck]))))

    F_shared = np.zeros((n, n, n))
    PS_est = np.zeros((n, n, n))
    for bk in range(n):
        env_B = build(sigma, goals[bk], beta)
        # per-state expected log-ratio under pi_B, for every prior source C
        lr = policies[bk] @ (log_ref[bk][:, None] - log_ref.T)   # (nS, n)
        est = V[bk] @ lr / beta                                  # (n, n): A x C
        for ck in range(n):
            if ck == bk:
                continue
            Fs, ok = free_energy_fixed_prior(env_B, p_hat[ck], beta)
            assert ok, (beta, bk, ck)
            F_shared[:, bk, ck] = Fs[goals]
            PS_est[:, bk, ck] = est[:, ck]
    gap = D[:, None, :] - (D[:, :, None] + D[None, :, :])
    A, B, C = np.ogrid[:n, :n, :n]
    valid = (A != B) & (B != C) & (A != C)
    PS = F_shared - D[:, :, None]
    return {"gap": gap, "valid": valid, "PS": PS, "PS_est": PS_est,
            "gap_shared": gap - PS, "fp_dev": fp_dev}


# %% [markdown]
# ## H1/H2 verdicts across beta (twist), with the Cartesian beta=1 reference

# %%
sigma_ex = load_exemplar_sigma()
runs = {}
for label, sigma, betas in (("twist", sigma_ex, BETAS),
                            ("cartesian", identity, [1.0])):
    for beta in betas:
        fr = solve_frame(sigma, beta)
        dec = decompose_frame(sigma, beta, fr)
        runs[(label, beta)] = {**fr, **dec}
        gap, valid = dec["gap"], dec["valid"]
        viol = valid & (gap > TOL)
        noise = max(2 * dec["fp_dev"], 10 * THETA)
        survivors = int((valid & (dec["gap_shared"] > noise)).sum())
        g, ps = gap[viol], dec["PS"][viol]
        print(f"{label:10s} beta={beta:<4} viol {viol.sum()/valid.sum():5.1%}  "
              f"max gap {g.max():6.3f}  shared-prior survivors (>{noise:.4f}): "
              f"{survivors}  median PS share {np.median(ps / g):5.2f}")

# %%
# H2: pairwise marginal divergence, twist vs cartesian (beta = 1)
def pairwise_marginal_kl(p_hat):
    ref = p_hat * (1 - REF_FLOOR) + REF_FLOOR / 4
    ref = ref / ref.sum(axis=1, keepdims=True)
    lg = np.log2(ref)
    return (ref[:, None, :] * (lg[:, None, :] - lg[None, :, :])).sum(-1)


for label in ("cartesian", "twist"):
    KL = pairwise_marginal_kl(runs[(label, 1.0)]["p_hat"])
    off = KL[~np.eye(KL.shape[0], dtype=bool)]
    print(f"{label:10s} pairwise KL(p_hat_B||p_hat_C): "
          f"median {np.median(off):.4f}  p90 {np.percentile(off, 90):.4f}  "
          f"max {off.max():.4f} bits")

# %% [markdown]
# ## H3: the section-7 first-order formula against measured PS

# %%
for label in ("cartesian", "twist"):
    r = runs[(label, 1.0)]
    valid = r["valid"]
    ps, est = r["PS"][valid], r["PS_est"][valid]
    bound_ok = float((ps <= est + 1e-3).mean())
    corr = float(np.corrcoef(ps, est)[0, 1])
    print(f"{label:10s} beta=1: corr(PS, PS_est) = {corr:.3f};  "
          f"PS <= PS_est holds for {bound_ok:.1%} of triples;  "
          f"median tightness PS/PS_est "
          f"{np.median(ps[est > 1e-6] / est[est > 1e-6]):.2f}")

# %% [markdown]
# ## H4: where PS lives — the home-cycle flow

# %%
def dominant_label_anatomy(sigma):
    """(dominant label by basin, its largest-basin cycle cells).

    Intended-move successors, goal-free-exact: the absorbing goal only
    corrupts its own T row, so we take successors from the goal-0 env
    and patch row 0 from a second env with a different goal.
    """
    env0 = build(sigma, 0, 1.0)
    goals = [int(s) for s in env0.available_states]
    T = np.asarray(env0.get_T(), dtype=float).reshape(nS, 4, nS)
    succ = T.argmax(axis=2)
    env1 = build(sigma, goals[1], 1.0)
    T1 = np.asarray(env1.get_T(), dtype=float).reshape(nS, 4, nS)
    succ[0] = T1.argmax(axis=2)[0]
    best = None
    for li in range(4):
        cycle_id, cyc_cells = {}, {}
        for s0 in goals:
            if s0 in cycle_id:
                continue
            path, seen, s = [], {}, s0
            while True:
                if s in cycle_id:
                    cid = cycle_id[s]
                    break
                if s in seen:
                    cyc = path[seen[s]:]
                    cid = min(cyc)
                    cyc_cells[cid] = set(cyc)
                    for c in cyc:
                        cycle_id[c] = cid
                    break
                seen[s] = len(path)
                path.append(s)
                s = int(succ[s, li])
            for q in path:
                if q not in cycle_id:
                    cycle_id[q] = cid
        counts = {}
        for s in goals:
            counts[cycle_id[s]] = counts.get(cycle_id[s], 0) + 1
        cid_star, size = max(counts.items(), key=lambda kv: kv[1])
        if best is None or size > best[1]:
            best = (li, size, cyc_cells.get(cid_star, {cid_star}))
    assert best is not None
    return best[0], best[2], best[1] / len(goals)


dom_label, cycle_cells, dom_cov = dominant_label_anatomy(sigma_ex)
print(f"dominant label {dom_label}, coverage {dom_cov:.3f}, "
      f"cycle cells {sorted(cycle_cells)}")

r_tw = runs[("twist", 1.0)]
goals = r_tw["goals"]
n = len(goals)

# per-goal alignment with the flow: marginal mass on the dominant label
align = r_tw["p_hat"][:, dom_label]
KL_tw = pairwise_marginal_kl(r_tw["p_hat"])

# median PS per (B, C) leg pair over start states A
PS_pair = np.median(r_tw["PS"], axis=0)                      # (B, C)
pair_mask = ~np.eye(n, dtype=bool)
both_on = (align[:, None] > 0.5) & (align[None, :] > 0.5) & pair_mask
mixed = pair_mask & ~both_on
print(f"legs with BOTH goals on-flow (marginal mass on label {dom_label} "
      f"> 0.5): {int(both_on.sum())} pairs, median PS "
      f"{np.median(PS_pair[both_on]):.3f} bits-equiv")
print(f"remaining leg pairs: {int(mixed.sum())}, median PS "
      f"{np.median(PS_pair[mixed]):.3f}")
r_flow = float(np.corrcoef(KL_tw[pair_mask], PS_pair[pair_mask])[0, 1])
print(f"corr(KL(p_hat_B||p_hat_C), median-A PS) over leg pairs = {r_flow:.3f}")

# %% [markdown]
# ## Figures

# %%
colors = {0.3: "#cc3311", 1.0: "#4477aa", 3.0: "#228833"}
fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8), dpi=150)

ax = axes[0]
for beta in BETAS:
    r = runs[("twist", beta)]
    viol = r["valid"] & (r["gap"] > TOL)
    ps = np.clip(r["PS"][viol], 1e-4, None)
    ax.scatter(ps, r["gap"][viol], s=6, alpha=0.35, color=colors[beta],
               label=f"$\\beta$={beta}")
ax.set_xscale("log"); ax.set_yscale("log")
lo, hi = 1e-4, 3e2
ax.plot([lo, hi], [lo, hi], "--", color="black", lw=0.9)
ax.set_xlim(lo, hi); ax.set_ylim(1e-3, 3e1)
ax.set_xlabel("PS (prior-specialisation benefit)")
ax.set_ylabel("gap (violation size)")
ax.set_title("twist frame: violations vs PS\n(at/below the line = explained)")
ax.legend(fontsize=8)

ax = axes[1]
r1 = runs[("twist", 1.0)]
ps, est = r1["PS"][r1["valid"]], r1["PS_est"][r1["valid"]]
sel = np.random.default_rng(0).choice(len(ps), size=min(6000, len(ps)),
                                      replace=False)
ax.scatter(est[sel], ps[sel], s=5, alpha=0.25, color="#4477aa")
lim = max(np.percentile(est, 99.5), np.percentile(ps, 99.5))
ax.plot([0, lim], [0, lim], "--", color="black", lw=0.9)
ax.set_xlim(-0.1 * lim, lim); ax.set_ylim(-0.1 * lim, lim)
ax.set_xlabel("PS_est: $(1/\\beta)\\,\\sum_s \\nu_1(s)\\,"
              "E_{\\pi_B}[\\log_2 \\hat p_B/\\hat p_C]$")
ax.set_ylabel("measured PS")
ax.set_title("the section-7 formula bounds and\npredicts PS (twist, $\\beta$=1)")

ax = axes[2]
H, W = SHAPE
grid = np.full(nS, np.nan)
mid_ps = np.median(PS_pair, axis=1)          # per midpoint B, median over C
for k, s in enumerate(goals):
    grid[s] = mid_ps[k]
im = ax.imshow(grid.reshape(SHAPE), cmap="magma")
for c in cycle_cells:
    ax.plot(c % W, c // W, "o", mfc="none", mec="#66ccee", mew=2.2, ms=13)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"median PS by midpoint goal B (twist, $\\beta$=1)\n"
             f"rings = dominant-label home cycle")
fig.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle("Prior specialisation under the exemplar twist "
             f"({ENV} {SHAPE[0]}x{SHAPE[1]}, det={DET})")
fig.tight_layout()
fig.savefig(Path(__file__).parent / "figs" / "F-twist-prior-specialisation.png"
            if "__file__" in dir() else "figs/F-twist-prior-specialisation.png",
            dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Discernments (2026-07-09 run, gridcore d432c4b)
#
# - **H1 confirmed — the shared-prior quasimetric is frame-independent.**
#   Under the exemplar twist the violation rate rises to 2.9-4.0% of
#   triples (vs Cartesian ~1%) and the max gap doubles, exactly as
#   notebook 04 found; yet repricing leg 1 under the direct problem's
#   marginal leaves ZERO violations above solver noise at every beta.
#   The twist bends which legs are marginal-compatible; it does not
#   touch the accounting.  (Median PS share exceeds 1 more strongly in
#   the twist frame — 1.48 at beta=1 vs 1.07 Cartesian — the mismatched
#   prior punishes twisted legs harder than segmentation gains.)
#
# - **H2 confirmed — amplification is marginal divergence.**  Pairwise
#   KL(p_hat_B || p_hat_C) between per-goal converged marginals:
#   Cartesian median 0.79 bits (max 10.4), twist median 1.24 bits (max
#   18.6).  Evolution buys mean cheapness by letting most goals ride one
#   concentrated marginal — and thereby makes the marginals of the
#   remaining goals MORE different from each other, which is where the
#   extra profitable midpoints come from.
#
# - **H3 confirmed — the section-7 formula works.**  The first-order
#   swap-the-prior estimate PS_est = (1/beta) sum_s nu1(s)
#   E_pi_B[log2 p_hat_B/p_hat_C] upper-bounds measured PS on 100.0% of
#   triples in BOTH frames and predicts it with r = 0.995 (Cartesian) /
#   0.934 (twist); median tightness 0.86 / 0.58 (re-optimisation under
#   the foreign prior recovers more in the twisted frame).  This is the
#   empirical validation of the workbench cognitive-geometry-kl-control
#   section-7 C_switch, on both the untwisted and evolved geometry.
#
# - **H4 confirmed — PS vanishes along the home-cycle flow.**  Goal
#   pairs whose marginals both put > 0.5 mass on the dominant label
#   (210 of 1,560 leg pairs) carry median PS 0.58 bits-equivalent; all
#   other leg pairs carry 6.06 — a 10x separation — and
#   corr(KL(p_hat_B||p_hat_C), median-A PS) = 0.92 across leg pairs.
#   nb09's prediction lands: the habit flow is a marginal-compatibility
#   class.  Legs that ride it segment for free; the specialisation
#   benefit (and hence the triangle violation) concentrates on legs
#   that mix twist-incoherent marginals.
#
# - **Reading for the infodesic paper (section 5-6 material)**: in the
#   2024 meeting language, the twist manufactures a shared "memory
#   marginal" for most of the space — Daniel's different-marginal-per-
#   memory-state saving, compiled into the labelling so no memory is
#   needed.  The price is a hard marginal boundary: off-flow goals need
#   strongly specialised priors, and that boundary is where the twisted
#   geometry's extra violations (= extra profitable subgoals) live.
#
# - Follow-ups: replicate on 2-3 more GA run-bests (one exemplar here);
#   beta-screen harvest — do twists EVOLVED at beta 0.5/0.3 widen or
#   narrow the marginal-divergence distribution vs this beta=1 exemplar?
