# %% [markdown]
# # Goal-geometry series, 12 — PS decompose on the saturated flat-world twists
#
# nb11's headline: twists evolved at low beta MANUFACTURE triangle
# violations on flat geometry.  Cartesian wrap_grid/helical have zero
# violations and identical per-goal marginals (KL = 0 by torus
# symmetry); their evolved twists saturate (dominant-label coverage
# 1.00) yet violate on 3.3-3.9% of triples with marginal KL medians of
# 5.6-8.3 bits.  This notebook runs the nb10 prior-specialisation
# decomposition on those four twists (wrap/helical x beta 0.5/0.3), at
# their own beta.
#
# Hypotheses.
#   H1 (frame independence, hardest test yet): the shared-prior
#      quasimetric collapse holds on twists whose violations were
#      created from NOTHING -- zero survivors above solver noise, and
#      the one-sided bound gap <= PS <= PS_est holds throughout.
#   H2 (pure marginal-boundary effect): with coverage 1.00 the home
#      flow reaches everywhere, so if violations survive there must be
#      goals whose marginals leave the dominant vocabulary -- PS should
#      track pairwise marginal KL (as everywhere else), and the
#      midpoint-PS map should organise around the home cycle rather
#      than any wall (there are none).
#   H3 (who leaves the vocabulary): the per-goal alignment
#      p_hat_g(dominant label) splits the goals; on-flow pairs carry
#      ~zero PS, and the violation mass sits on legs crossing the
#      vocabulary boundary.

# %%
import re
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

SCREEN = Path("/media/merlin/grid-twist/gridtwist-outputs/core-beta-screen-09-07")
SHAPE, DET, THETA, TOL = (7, 7), 0.97, 1e-5, 1e-6
REF_FLOOR = 1e-6
nS = SHAPE[0] * SHAPE[1]

TARGETS = [("wrap_grid", 0.5), ("wrap_grid", 0.3),
           ("helical", 0.5), ("helical", 0.3)]

RUN_RE = re.compile(
    r"b(?P<beta>\d+)-free-core-ga-(?P<env>[a-z-]+)-7x7-s314")


def load_target(env_id, beta):
    tok = {0.3: "b03", 0.5: "b05"}[beta]
    env_tok = env_id.replace("_", "-")
    for d in SCREEN.iterdir():
        if tok in d.name and env_tok + "-7x7" in d.name:
            return np.asarray(np.load(next(d.glob("*multi-all.sigma.npy"))),
                              dtype=int)
    raise FileNotFoundError((env_id, beta))


def build(env_id, sigma, goal, beta):
    cfg = EvalConfig(env_id=env_id, shape=SHAPE, goal=int(goal), beta=beta,
                     determinism=DET, manhattan=True, theta=THETA,
                     state_dist="uniform")
    return build_twisted_env_from_sigma(sigma, cfg)


def solve_frame(env_id, sigma, beta):
    env0 = build(env_id, sigma, 0, beta)
    goals = [int(s) for s in env0.available_states]
    n = len(goals)
    D = np.zeros((n, n))
    policies = np.zeros((n, nS, 4))
    V = np.zeros((n, n, nS))
    p_hat = np.zeros((n, 4))
    for j, g in enumerate(goals):
        env = build(env_id, sigma, g, beta)
        di = DecisionInformation(env, _state_dist_class("uniform")(env), THETA,
                                 max_iterations=200_000,
                                 max_info_iterations=10_000)
        pi, _, F = di.get_opt_policy_Z_free_vector(beta)
        assert di.converged, (env_id, beta, g)
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
    return dict(D=D, goals=goals, policies=policies, V=V, p_hat=p_hat)


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


def decompose(env_id, sigma, beta, fr):
    D, goals, p_hat = fr["D"], fr["goals"], fr["p_hat"]
    policies, V = fr["policies"], fr["V"]
    n = len(goals)
    ref = p_hat * (1 - REF_FLOOR) + REF_FLOOR / 4
    ref = ref / ref.sum(axis=1, keepdims=True)
    log_ref = np.log2(ref)
    fp_dev = 0.0
    for ck in range(0, n, max(n // 5, 1)):
        env = build(env_id, sigma, goals[ck], beta)
        Fc, ok = free_energy_fixed_prior(env, p_hat[ck], beta)
        assert ok
        fp_dev = max(fp_dev, float(np.max(np.abs(Fc[goals] - D[:, ck]))))
    F_shared = np.zeros((n, n, n))
    PS_est = np.zeros((n, n, n))
    for bk in range(n):
        env_B = build(env_id, sigma, goals[bk], beta)
        lr = policies[bk] @ (log_ref[bk][:, None] - log_ref.T)
        est = V[bk] @ lr / beta
        for ck in range(n):
            if ck == bk:
                continue
            Fs, ok = free_energy_fixed_prior(env_B, p_hat[ck], beta)
            assert ok, (env_id, beta, bk, ck)
            F_shared[:, bk, ck] = Fs[goals]
            PS_est[:, bk, ck] = est[:, ck]
    gap = D[:, None, :] - (D[:, :, None] + D[None, :, :])
    A, B, C = np.ogrid[:n, :n, :n]
    valid = (A != B) & (B != C) & (A != C)
    PS = F_shared - D[:, :, None]
    return dict(gap=gap, valid=valid, PS=PS, PS_est=PS_est,
                gap_shared=gap - PS, fp_dev=fp_dev)


def dominant_label_anatomy(env_id, sigma):
    env0 = build(env_id, sigma, 0, 1.0)
    goals = [int(s) for s in env0.available_states]
    T = np.asarray(env0.get_T(), dtype=float).reshape(nS, 4, nS)
    succ = T.argmax(axis=2)
    env1 = build(env_id, sigma, goals[1], 1.0)
    succ[0] = np.asarray(env1.get_T(), dtype=float).reshape(nS, 4, nS).argmax(axis=2)[0]
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


def pairwise_marginal_kl(p_hat):
    ref = p_hat * (1 - REF_FLOOR) + REF_FLOOR / 4
    ref = ref / ref.sum(axis=1, keepdims=True)
    lg = np.log2(ref)
    return (ref[:, None, :] * (lg[:, None, :] - lg[None, :, :])).sum(-1)


# %% [markdown]
# ## The four decompositions

# %%
results = {}
for env_id, beta in TARGETS:
    sigma = load_target(env_id, beta)
    fr = solve_frame(env_id, sigma, beta)
    dec = decompose(env_id, sigma, beta, fr)
    dl, cyc, cov = dominant_label_anatomy(env_id, sigma)
    results[(env_id, beta)] = {**fr, **dec, "sigma": sigma,
                               "dom": dl, "cycle": cyc, "cov": cov}
    gap, valid = dec["gap"], dec["valid"]
    PS, PS_est = dec["PS"], dec["PS_est"]
    viol = valid & (gap > TOL)
    noise = max(2 * dec["fp_dev"], 10 * THETA)
    surv = int((valid & (dec["gap_shared"] > noise)).sum())
    ps_v, est_v = PS[valid], PS_est[valid]
    align = fr["p_hat"][:, dl]
    n = len(fr["goals"])
    pm = ~np.eye(n, dtype=bool)
    PSp = np.median(PS, axis=0)
    on = (align[:, None] > 0.5) & (align[None, :] > 0.5) & pm
    KL = pairwise_marginal_kl(fr["p_hat"])
    r_flow = float(np.corrcoef(KL[pm], PSp[pm])[0, 1])
    print(f"{env_id:10s} b{beta}  cov {cov:.2f} (label {dl}, cycle "
          f"{len(cyc)} cells)  viol {viol.sum()/valid.sum():5.1%}  "
          f"survivors(>{noise:.4f}): {surv}")
    print(f"    PS share median {np.median(PS[viol]/gap[viol]):.2f}  "
          f"bound {float((ps_v <= est_v + 1e-3).mean()):6.1%}  "
          f"corr(PS,PS_est) {float(np.corrcoef(ps_v, est_v)[0,1]):.3f}")
    print(f"    goals on-flow (align>0.5): {int((align > 0.5).sum())}/{n}  "
          f"on-flow pairs {int(on.sum())}  "
          f"PS on/off {np.median(PSp[on]) if on.any() else float('nan'):.2f}"
          f"/{np.median(PSp[pm & ~on]):.2f}  corr(KL, PS) {r_flow:.3f}")

# %% [markdown]
# ## Figures

# %%
fig, axes = plt.subplots(1, 4, figsize=(18.6, 4.6), dpi=150)

ax = axes[0]
cols = {("wrap_grid", 0.5): "#88ccee", ("wrap_grid", 0.3): "#4477aa",
        ("helical", 0.5): "#eeaa66", ("helical", 0.3): "#cc3311"}
for key, r in results.items():
    viol = r["valid"] & (r["gap"] > TOL)
    ps = np.clip(r["PS"][viol], 1e-4, None)
    ax.scatter(ps, r["gap"][viol], s=5, alpha=0.3, color=cols[key],
               label=f"{key[0]} b{key[1]}")
ax.set_xscale("log"); ax.set_yscale("log")
lo, hi = 1e-4, 3e2
ax.plot([lo, hi], [lo, hi], "--", color="black", lw=0.9)
ax.set_xlim(lo, hi); ax.set_ylim(1e-3, 2e1)
ax.set_xlabel("PS"); ax.set_ylabel("gap")
ax.set_title("flat-world twists: violations vs PS\n(at/below line = explained)")
ax.legend(fontsize=7)

ax = axes[1]
for key, r in results.items():
    align = r["p_hat"][:, r["dom"]]
    ax.hist(align, bins=np.linspace(0, 1, 21), histtype="step",
            color=cols[key], label=f"{key[0]} b{key[1]}")
ax.axvline(0.5, color="black", lw=0.7, ls=":")
ax.set_xlabel("per-goal marginal mass on the dominant label")
ax.set_ylabel("goals")
ax.set_title("who leaves the vocabulary\n(saturated flow, divergent marginals)")
ax.legend(fontsize=7)

for ax, key in ((axes[2], ("wrap_grid", 0.3)), (axes[3], ("helical", 0.3))):
    r = results[key]
    PSp = np.median(r["PS"], axis=0)
    grid = np.full(nS, np.nan)
    mid = np.median(PSp, axis=1)
    for k, s in enumerate(r["goals"]):
        grid[s] = mid[k]
    im = ax.imshow(grid.reshape(SHAPE), cmap="magma")
    for c in r["cycle"]:
        ax.plot(c % SHAPE[1], c // SHAPE[1], "o", mfc="none", mec="#66ccee",
                mew=2.2, ms=13)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"median PS by midpoint, {key[0]} b{key[1]}\n"
                 f"rings = home cycle (no walls anywhere)")
    fig.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle("Prior specialisation where the world is flat: the violation "
             "structure is entirely the twist's")
fig.tight_layout()
fig.savefig(Path(__file__).parent / "figs" / "F-flat-world-ps.png"
            if "__file__" in dir() else "figs/F-flat-world-ps.png",
            dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Discernments (2026-07-10 run, gridcore d432c4b)
#
# - **H1 passes its hardest test.**  On twists whose violations were
#   manufactured from flat geometry, the shared-prior repricing still
#   collapses every violation: zero survivors above solver noise in all
#   four decompositions, and the one-sided bound gap <= PS <= PS_est
#   holds on 100.0% of triples.  The corrected relaxed inequality is
#   not a walled-world fact; it is a fact about the accounting.
#
# - **The anatomy surprise: saturation does not mean a shared
#   vocabulary.**  Coverage is 1.00 — the dominant label drains every
#   cell — yet only 11-14 of 49 goals keep marginal mass > 0.5 on that
#   label, and the alignment histogram is BIMODAL with its biggest
#   spike at ZERO.  Reaching a specific goal against the global drain
#   requires pressing against it: the twist creates one home vocabulary
#   plus a fan of per-goal counter-vocabularies, and THAT split is the
#   manufactured violation structure.  (Also why nb11's saturated flat
#   twists had the largest marginal KL medians.)
#
# - **PS overshoots hard at low beta**: median PS/gap 4.3-7.0 (vs
#   1.07-1.48 at beta=1 on four_rooms).  The marginal-divergence credit
#   is far larger than the segmentation profit it licenses — at low
#   beta the mismatched prior is brutal, so the bound is loose though
#   never violated (corr(PS, PS_est) drops to 0.64-0.69, the
#   re-optimisation slack growing as beta falls).
#
# - **H2 confirmed — the home cycle is the low-PS spine even with no
#   walls.**  In the midpoint maps the cycle cells sit in the darkest
#   region for both worlds at both betas, and corr(marginal KL, PS) =
#   0.76-0.80.  On flat worlds there is nothing else for the structure
#   to organise around: the violation geography is entirely the
#   twist's own marginal-compatibility boundary, i.e. the geometry of
#   the habit.
#
# - Follow-up (paper section 6 material): the per-goal alignment split
#   (14/49 on-flow) versus the paper-1 escape-role decomposition —
#   are the off-vocabulary goals the directed-escape targets?
