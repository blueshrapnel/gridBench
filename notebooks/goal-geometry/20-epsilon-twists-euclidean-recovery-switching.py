# %% [markdown]
# # Goal-geometry series, 20 — ε-twists: Euclidean recovery against switching costs
#
# The 2023 progression report (phd/writing/progression-report, §"relabelled")
# twisted an 11×11 Manhattan grid by relabelling the actions of each state with
# probability ε, solved every goal's free-energy policy under a *uniform* state
# reference, and projected the symmetrised free-energy matrix with MDS. The
# finding was that "the underlying Euclidean [grid] is still prominent" up to
# ε = 0.65 at β = 1, with local distortions, and that at ε = 0.95 all states are
# similarly distorted. The question here: when a twist leaves the free-energy
# geometry near-Euclidean, what happens to the switching costs — is the triangle
# inequality upheld, how large are slack, advantage, bound and gap, and does
# Euclidean recovery predict any of it?
#
# Method. ε ∈ {0, 0.05, 0.15, 0.35, 0.5, 0.65, 0.85, 0.95}, three seeds each,
# 11×11 Manhattan, deterministic moves, β ∈ {1, 0.3}, uniform and pooled live
# references, every goal solved with gridCore's nested fixed-point solver
# (gridcore.info.fixed_point, certified). Per twist and setting:
#
#   recovery   Procrustes error (root form, scale removed) of the 2-D classical
#              MDS of the symmetrised free-energy matrix against the grid layout,
#              and the same for the goals' action priors (sqrt JS);
#   switching  for every ordered triple: defect, shared-prior slack and prior
#              advantage (report eq. prior-slack), the re-scoring bound and the
#              disagreement gap; violation share, pairs helped, vacuous bounds,
#              zero-slack violators, bound ÷ violation, where the best interim
#              goals sit (edge, corner, interior).
#
# Run (the fixed-point solver lives on the gridCore branch feature/stage2-fixed-point-solver):
#   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#   PYTHONPATH=/media/merlin/phd-marlyn/gridCore-fixed-point/src python 20-...py

# %%
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import itertools, json, subprocess
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

import gridcore
from gridcore.envs import GridRoom
from gridcore.info.fixed_point import absorb, kl_rows, solve_fixed_prior, solve_self_consistent, transition_tensor

repo = Path(gridcore.__file__).resolve().parents[2]
try:
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    commit = "unknown"
print(f"gridcore: {repo.name} @ {commit}")

OUT = Path("/media/merlin/Dropbox/workbench/topics/twisted-switching")
FIG, RES = OUT / "figures", OUT / "results"
FIG.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)
CACHE = Path("/media/merlin/fixed-point-sweep/epsilon-twists"); CACHE.mkdir(parents=True, exist_ok=True)

SIDE = 11
EPS = [0.0, 0.05, 0.15, 0.35, 0.5, 0.65, 0.85, 0.95]
SEEDS = [1, 2, 3]
BETAS = [1.0, 0.3]
REFS = ["uniform", "pooled"]
TWISTS = [(0.0, 0)] + [(e, s) for e in EPS[1:] for s in SEEDS]

# %% [markdown]
# ## Twists, geometry helpers

# %%
def make_env(eps: float, seed: int) -> GridRoom:
    """GridRoom twists at construction when epsilon > 0: each state's labels are
    shuffled with probability eps (the 2023 convention), from the seeded rng.
    Do not call twist_world again: that would compose a second, unseeded twist."""
    return GridRoom({"shape": (SIDE, SIDE), "goals": [0], "manhattan": True, "determinism": 1.0, "epsilon": eps, "twist_seed": seed})


def chi_twist(sigma: np.ndarray) -> float:
    """Twist strength: 1 minus the best global alignment of labels with physical actions."""
    C = np.zeros((4, 4), int)
    for row in sigma:
        for b in range(4):
            C[row[b], b] += 1
    best = max(sum(C[p[b], b] for b in range(4)) for p in itertools.permutations(range(4)))
    return 1 - best / (4 * len(sigma))


def classical_mds(D: np.ndarray, k: int = 2) -> np.ndarray:
    n = len(D); J = np.eye(n) - 1 / n
    w, v = np.linalg.eigh(-0.5 * J @ (D ** 2) @ J)
    o = np.argsort(w)[::-1][:k]
    return v[:, o] * np.sqrt(np.maximum(w[o], 0))


def procrustes_root(X: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray]:
    """Root Procrustes error after centring, optimal rotation and scale, and the aligned X."""
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    U, S, Vt = np.linalg.svd(Xc.T @ Yc)
    err = float(np.sqrt(max(0.0, 1 - S.sum() ** 2 / ((Xc ** 2).sum() * (Yc ** 2).sum()))))
    R = U @ Vt
    scale = S.sum() / (Xc ** 2).sum()
    return err, scale * Xc @ R + Y.mean(0)


def js_matrix(Q: np.ndarray) -> np.ndarray:
    def H(p):
        return -(np.where(p > 0, p * np.log2(np.maximum(p, 1e-300)), 0)).sum(-1)
    M = 0.5 * (Q[:, None, :] + Q[None, :, :])
    return np.maximum(H(M) - 0.5 * (H(Q)[:, None] + H(Q)[None, :]), 0)


LAYOUT = np.array([[s % SIDE, s // SIDE] for s in range(SIDE * SIDE)], dtype=float)
r_, c_ = LAYOUT[:, 1], LAYOUT[:, 0]
ON_EDGE = (r_ == 0) | (r_ == SIDE - 1) | (c_ == 0) | (c_ == SIDE - 1)
CORNER = ((r_ == 0) | (r_ == SIDE - 1)) & ((c_ == 0) | (c_ == SIDE - 1))

# %% [markdown]
# ## One twist, one setting: solve every goal, then every triple

# %%
def analyse(args):
    eps, seed, ref, beta = args
    key = f"eps-{eps:g}-seed-{seed}-{ref}-b-{beta:g}"
    f = CACHE / f"{key}.npz"
    env = make_env(eps, seed)
    T, states = transition_tensor(env); n = len(states); idx = np.arange(n)
    sigma = env.sigma
    if f.exists():
        d = dict(np.load(f)); F, Q, pol = d["F"], d["Q"], d["pol"]
        row = json.loads(str(d["row"]))
        return row, F, Q
    sols = [solve_self_consistent(T, g, beta, ref) for g in range(n)]
    # a few twisted goals at beta 0.3 do not reach the certificate in 20,000 prior
    # updates (residuals ~1e-4); record them rather than stop, but refuse anything worse
    uncertified = [g for g, s in enumerate(sols) if not s.certified()]
    max_bellman = max(s.bellman_residual for s in sols); max_marginal = max(s.marginal_residual for s in sols)
    assert max_bellman < 1e-2 and max_marginal < 1e-3, (key, max_bellman, max_marginal)
    F = np.column_stack([s.F for s in sols]); Q = np.array([s.prior for s in sols]); pol = np.array([s.policy for s in sols])
    # geometry: Euclidean recovery of the free-energy and prior maps
    err_F, _ = procrustes_root(classical_mds((F + F.T) / 2), LAYOUT)
    err_Q, _ = procrustes_root(classical_mds(np.sqrt(js_matrix(Q))), LAYOUT)
    # switching: per interim goal
    off = ~np.eye(n, dtype=bool)
    dead = Q < 1e-6; lq = np.where(dead, 0.0, np.log2(np.maximum(Q, 1e-300)))
    viol = 0; vac = 0; vac_v = 0; n_tri = 0; worst = 0.0; min_slack = np.inf; bound_margin = np.inf
    sav_l, slack_l, adv_l, C_l, D_l = [], [], [], [], []
    best = np.full((n, n), np.inf); arg = np.full((n, n), -1)
    for sp in range(n):
        s = sols[sp]; pi1 = s.policy; N1 = s.occupancy
        dF = F[:, [sp]] + F[[sp], :] - F
        mask = (idx[:, None] != sp) & (idx[None, :] != sp) & off
        L = (pi1 @ lq[sp])[:, None] - pi1 @ lq.T
        C = (N1 @ L) / beta
        C = np.where((N1 @ (pi1 @ (dead & ~dead[sp][None, :]).T)) > 1e-9, np.inf, C)
        with np.errstate(divide="ignore", invalid="ignore"):
            K = np.stack([kl_rows(pi1, pol[g]) for g in range(n)], 1)
        K = np.where(np.isfinite(K), K, 1e300); D = (N1 @ K) / beta; D = np.where(D > 1e200, np.inf, D)
        Fq = np.full((n, n), np.inf)
        for g in range(n):
            if g != sp:
                Fq[:, g] = solve_fixed_prior(T, sp, Q[g], beta, warm_policy=pi1).F
        A = Fq - F[:, [sp]]; S = Fq + F[[sp], :] - F
        v = mask & (dF < -1e-7)
        n_tri += int(mask.sum()); viol += int(v.sum()); vac += int((mask & ~np.isfinite(C)).sum()); vac_v += int((v & ~np.isfinite(C)).sum())
        if v.any():
            worst = max(worst, float(-dF[v].min()))
            sav_l.append(-dF[v]); slack_l.append(S[v]); adv_l.append(A[v]); C_l.append(C[v]); D_l.append(D[v])
        # an uncertified goal (Bellman residual r) can push slack and bound below zero by O(r)
        slack_tol = 1e-7 + 10 * max_bellman
        min_slack = min(min_slack, float(S[mask & np.isfinite(S)].min()))
        assert min_slack > -slack_tol, (key, "negative slack", min_slack, max_bellman)
        fin = mask & np.isfinite(C)
        bound_margin = min(bound_margin, float((C[fin] + dF[fin]).min()))
        assert bound_margin > -slack_tol, (key, "bound fails", bound_margin, max_bellman)
        better = mask & (dF < best); arg[better] = sp; best = np.where(better, dF, best)
    helped = (best < -1e-7) & off
    b = arg[helped]
    row = dict(eps=eps, seed=seed, reference=ref, beta=beta, chi=chi_twist(sigma),
               uncertified_goals=len(uncertified), max_bellman_residual=max_bellman, max_marginal_residual=max_marginal,
               min_slack=min_slack, min_bound_margin=bound_margin,
               eps_actual=float(np.mean([not np.array_equal(row_, np.arange(4)) for row_ in sigma])),
               mean_F=float(F[off].mean()), mean_info=float(np.mean([s.info[off[:, g]].mean() for g, s in enumerate(sols)])),
               recovery_error_F=err_F, recovery_error_prior=err_Q,
               mean_sqrt_js=float(np.sqrt(js_matrix(Q))[np.triu_indices(n, 1)].mean()),
               triples=n_tri, violating=viol, violating_share=viol / n_tri, worst_violation=worst,
               pairs_helped_share=float(helped[off].mean()),
               vacuous_share=vac / n_tri, vacuous_violators_share=(vac_v / viol) if viol else 0.0,
               best_on_edge_share=float(ON_EDGE[states[b]].mean()) if b.size else float("nan"),
               best_in_corner_share=float(CORNER[states[b]].mean()) if b.size else float("nan"),
               edge_share_of_states=float(ON_EDGE.mean()), corner_share_of_states=float(CORNER.mean()))
    if viol:
        sav, S, A, C, D = (np.concatenate(x) for x in (sav_l, slack_l, adv_l, C_l, D_l))
        finC = np.isfinite(C)
        top = sav >= np.percentile(sav, 99)
        row.update(saving_median=float(np.median(sav)), slack_median=float(np.median(S)), advantage_median=float(np.median(A)),
                   bound_median=float(np.median(C[finC])) if finC.any() else float("nan"), gap_median=float(np.median(D[np.isfinite(D)])),
                   zero_slack_share=float((S < 1e-6).mean()),
                   ratio_bound_median=float(np.median(C[finC] / sav[finC])) if finC.any() else float("nan"),
                   ratio_bound_top1pct_median=float(np.median((C / sav)[top & finC])) if (top & finC).any() else float("nan"),
                   ratio_advantage_median=float(np.median(A / sav)))
    np.savez_compressed(f, F=F, Q=Q, pol=pol, row=json.dumps(row))
    return row, F, Q

# %%
jobs = [(e, s, ref, b) for (e, s) in TWISTS for ref in REFS for b in BETAS]
rows = []; maps = {}
with Pool(10) as pool:
    for (row, F, Q), job in zip(pool.imap(analyse, jobs, chunksize=1), jobs):
        rows.append(row); maps[job] = (F, Q)
        print(f"eps {row['eps']:<5g} seed {row['seed']} {row['reference']:8s} β {row['beta']:<4g} χ {row['chi']:.2f} | recovery F {row['recovery_error_F']:.3f} prior {row['recovery_error_prior']:.3f} "
              f"| viol {100*row['violating_share']:.2f}% worst {row['worst_violation']:.2f} helped {100*row['pairs_helped_share']:.0f}% vacuous {100*row['vacuous_share']:.0f}% "
              f"| zero-slack {100*row.get('zero_slack_share', float('nan')):.0f}% ratio top1% {row.get('ratio_bound_top1pct_median', float('nan')):.2f}", flush=True)
(RES / "epsilon-twists.json").write_text(json.dumps(rows, indent=1) + "\n")

# %% [markdown]
# ## Figure 1: the MDS of the free-energy matrix, one seed per ε (uniform reference, β = 1)

# %%
BLUE = "#2a78d6"; INK = "#1f2328"; EDGE = "#9aa1ab"
edges = [(s, t) for s in range(SIDE * SIDE) for t in (s + 1, s + SIDE) if t < SIDE * SIDE and not (t == s + 1 and t % SIDE == 0)]
for ref in REFS:
    fig, axes = plt.subplots(2, len(EPS), figsize=(2.6 * len(EPS), 5.6), dpi=170)
    for c, eps in enumerate(EPS):
        seed = 0 if eps == 0 else SEEDS[0]
        for r, beta in enumerate(BETAS):
            F, Q = maps[(eps, seed, ref, beta)]
            row = next(x for x in rows if x["eps"] == eps and x["seed"] == seed and x["reference"] == ref and x["beta"] == beta)
            X = classical_mds((F + F.T) / 2); err, Xa = procrustes_root(X, LAYOUT)
            ax = axes[r, c]
            for i, j in edges:
                ax.plot(Xa[[i, j], 0], Xa[[i, j], 1], color=EDGE, lw=0.5, zorder=1)
            ax.scatter(Xa[:, 0], Xa[:, 1], s=9, color=BLUE, edgecolors="white", linewidths=0.3, zorder=2)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.invert_yaxis()
            for sp_ in ax.spines.values():
                sp_.set_color("#d0d4da")
            ax.set_title((f"ε = {eps:g}, χ = {row['chi']:.2f}\n" if r == 0 else "") + f"β = {beta:g}: error {err:.2f}, {100*row['violating_share']:.1f}% violate", fontsize=8)
    fig.suptitle(f"11×11 Manhattan, {ref} reference: MDS of the symmetrised free energy, aligned to the grid", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"mds-recovery-{ref}.png", bbox_inches="tight"); fig.savefig(FIG / f"mds-recovery-{ref}.pdf", bbox_inches="tight")
    plt.close(fig)

# %% [markdown]
# ## Figure 2: does Euclidean recovery predict the switching costs?

# %%
EPSCOL = LinearSegmentedColormap.from_list("eps", ["#0d366b", "#2a78d6", "#9ec5f4", "#f2a99b", "#d6453a", "#7a1410"])
MARK = {"uniform": "o", "pooled": "s"}
metrics = [("violating_share", "violating triples", 100.0), ("pairs_helped_share", "start–goal pairs with a cheaper interim goal, %", 100.0),
           ("ratio_bound_top1pct_median", "bound ÷ violation, largest 1 % of violations", 1.0), ("vacuous_share", "triples with an infinite bound, %", 100.0),
           ("zero_slack_share", "violators with zero slack, %", 100.0), ("gap_median", "violators' median gap C_disagree/β", 1.0)]
fig, axes = plt.subplots(2, 3, figsize=(13.5, 8), dpi=170, gridspec_kw=dict(wspace=0.32, hspace=0.32))
for ax, (k, lab, scale) in zip(axes.ravel(), metrics):
    for beta, alpha in ((1.0, 1.0), (0.3, 0.45)):
        for ref in REFS:
            pts = [x for x in rows if x["beta"] == beta and x["reference"] == ref and k in x]
            ax.scatter([x["recovery_error_F"] for x in pts], [scale * x[k] for x in pts], c=[x["eps"] for x in pts], cmap=EPSCOL, vmin=0, vmax=1,
                       marker=MARK[ref], s=38, alpha=alpha, edgecolors=INK, linewidths=0.4, label=f"{ref}, β = {beta:g}")
    ax.set_xlabel("recovery error of the free-energy map (0 = the grid)", fontsize=9); ax.set_ylabel(lab, fontsize=9)
    ax.grid(True, color="#e8e8e8", lw=0.6); ax.set_axisbelow(True)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
axes[0, 0].legend(frameon=False, fontsize=7.5)
sm = plt.cm.ScalarMappable(cmap=EPSCOL, norm=plt.Normalize(0, 1)); cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02); cb.set_label("ε"); cb.outline.set_visible(False)
fig.savefig(FIG / "recovery-vs-switching.png", bbox_inches="tight"); fig.savefig(FIG / "recovery-vs-switching.pdf", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Figure 3: the spectrum over violators against ε

# %%
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), dpi=170)
for ax, (k, lab) in zip(axes, (("slack_median", "shared-prior slack"), ("advantage_median", "prior advantage"), ("bound_median", "bound C_rescoring/β"), ("gap_median", "gap C_disagree/β"))):
    for beta, ls in ((1.0, "-"), (0.3, "--")):
        for ref, col in (("uniform", "#2a78d6"), ("pooled", "#d6453a")):
            xs, ys, lo, hi = [], [], [], []
            for eps in EPS:
                vals = [x[k] for x in rows if x["eps"] == eps and x["beta"] == beta and x["reference"] == ref and k in x and np.isfinite(x[k])]
                if vals:
                    xs.append(eps); ys.append(np.median(vals)); lo.append(min(vals)); hi.append(max(vals))
            ax.plot(xs, ys, ls=ls, color=col, lw=1.6, marker="o", ms=3, label=f"{ref}, β = {beta:g}")
            ax.fill_between(xs, lo, hi, color=col, alpha=0.12, lw=0)
    ax.set_title(f"violators' median {lab}", fontsize=9.5, loc="left"); ax.set_xlabel("ε"); ax.set_yscale("log")
    ax.grid(True, color="#e8e8e8", lw=0.6); ax.set_axisbelow(True)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
axes[0].set_ylabel("free-energy units (bits at β = 1)"); axes[0].legend(frameon=False, fontsize=7.5)
fig.tight_layout()
fig.savefig(FIG / "spectrum-vs-epsilon.png", bbox_inches="tight"); fig.savefig(FIG / "spectrum-vs-epsilon.pdf", bbox_inches="tight")
plt.close(fig)
print("figures written to", FIG)
