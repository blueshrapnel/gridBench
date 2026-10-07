# %% [markdown]
# # Goal-geometry series, 19 — slack, advantage, bound and gap under evolved twists
#
# The switching-costs report (workbench/topics/switching-costs, October 2026)
# decomposes the triangle defect of a two-leg journey s -> s' -> g into
#
#     defect  =  shared-prior slack  -  prior advantage of the first leg      (report eq. prior-slack)
#     saving  =  -defect  <=  C_rescoring / beta                               (the bound)
#     C_rescoring - C_disagree  =  beta * saving    (no early arrival)        (report eq. exact-defect)
#
# and Figures 4, 5 and 7 of the report show these on the untwisted 13x13 four
# rooms with deterministic moves.  This notebook runs the same verifications on
# the twisted four rooms of the twists work (determinism 0.97, pooled live
# reference, beta 1 and 0.3): the untwisted labelling, and the two most evolved
# stored twists, which carry habit cycles (the warm-started 1000-generation twist,
# chi 0.58, four-cell home cycle, dominant-label coverage 0.79; and the highest-chi
# twist, chi 0.64, six-cell home cycle).  Fixed points are the certified ones
# stored by fieldInfo's twist_compare.py on 2 October 2026.
#
# Questions.
#   Q1  Does the bound hold on every finite triple, and is the slack never negative,
#       under a twist?  (Theorems; this tests the implementation on twisted dynamics.)
#   Q2  Is the spectrum of slack, advantage, bound and gap different under a twist:
#       more violations, larger advantages, tighter or looser bounds?
#   Q3  Where do the useful interim goals sit under a twist: still at doorways, or on
#       the habit cycle of the dominant label?
#   Q4  With stochastic moves (0.97) early arrival is always possible; how large is the
#       residual of the no-early-arrival identity, and does it track p_early?
#
# Outputs (figures and JSON) go to the report's figure folder,
# workbench/topics/switching-costs/figures/twisted-worlds/; the per-triple arrays
# are cached outside Dropbox in /media/merlin/fixed-point-sweep/switching-twisted/.
#
# Run:  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=gridCore/src:gridBench/src python 19-...py
# (the fieldInfo stage2 solver spawns 20 BLAS threads otherwise; see memory
#  feedback_blas_threads_fieldinfo).

# %%
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import glob, json, subprocess, sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, TwoSlopeNorm, LinearSegmentedColormap
import numpy as np

import gridcore
repo = Path(gridcore.__file__).resolve().parents[2]
try:
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    commit = "unknown"
print(f"gridcore: {repo.name} @ {commit}")

FIELDINFO = "/media/merlin/Dropbox/phd/code/fieldInfo/stage2/ba_comparison"      # solver and the 2 Oct twist worlds
REPORT_FIGS = "/media/merlin/Dropbox/workbench/topics/switching-costs/figure-code"  # the report's grid-drawing style
for p in (FIELDINFO, REPORT_FIGS):
    if p not in sys.path:
        sys.path.insert(0, p)
from compare import absorb, occupancy, fixed_prior_pi, backup    # noqa: E402
from sweep import DEFAULT_OUT                                     # noqa: E402
from twist_compare import base_world, twisted, chi_twist, rooms, EVOLVED, ROOT, SIDE   # noqa: E402
from make_waypoint_intro_figure import draw_grid, xy, SEQ, ROOM_TINTS, room_of        # noqa: E402
from make_slack_advantage_figure import REDS, BLUES, DEFECT, NEGATIVE                   # noqa: E402
from journey_marks import mark_journey, OUTLINE                                        # noqa: E402
from gridbench.functional_graph.probe_env import build_goal_free_probe_env             # noqa: E402
from gridbench.functional_graph.label_graphs import label_graphs, walls_and_nonwalls   # noqa: E402

OUT = Path("/media/merlin/Dropbox/workbench/topics/switching-costs/figures/twisted-worlds")
CACHE = Path("/media/merlin/fixed-point-sweep/switching-twisted")
OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
BETAS = (1.0, 0.3)
WORLDS = [("untwisted", None), ("random twist", "random twist"), ("evolved, lowest F", "evolved, lowest F"),
          ("evolved, high χ, warm", "evolved, high χ, warm"), ("evolved, highest χ", "evolved, highest χ")]
TAG = {"untwisted": "untwisted", "random twist": "random_twist", "evolved, lowest F": "evolved_lowest_F",
       "evolved, high χ, warm": "evolved_high_chi_warm", "evolved, highest χ": "evolved_highest_chi"}
START, GOAL = (12, 0), (6, 0)          # the report's Figure 1 journey

# %% [markdown]
# ## Worlds, stored fixed points, habit cycles

# %%
T0, walk = base_world(); n = len(walk); pos = {int(s): i for i, s in enumerate(walk)}; idx = np.arange(n)
lab, near_door = rooms(walk)
env = build_goal_free_probe_env("four_rooms", (SIDE, SIDE), 0.97)
_, nonwall = walls_and_nonwalls(env)


def habit_cycle(sigma):
    """Cells of the dominant label's home cycle (largest basin's attractor), as in notebook 51."""
    graphs = label_graphs(env, sigma)
    cov = [int(np.asarray(g.fg.basin_sizes).max()) / len(nonwall) if len(g.fg.basin_sizes) else 0.0 for g in graphs]
    dom = int(np.argmax(cov)); fg = graphs[dom].fg
    b = int(np.argmax(np.asarray(fg.basin_sizes)))
    cyc = sorted(set(int(x) for x in fg.cycles[b]) & set(int(x) for x in nonwall))
    basin = sorted(int(x) for x in np.flatnonzero(np.asarray(fg.basin_id) == b) if int(x) in set(int(y) for y in nonwall))
    return dict(dominant=dom, coverage=cov, cycle=cyc, basin=basin)


worlds = {}
for name, key in WORLDS:
    if key is None:
        T, sigma, chi, cyc = T0, None, 0.0, dict(dominant=None, coverage=None, cycle=[], basin=[])
    else:
        if key == "random twist":   # the 2 October draw: twist_compare.py's rng seed 20261002
            rng = np.random.default_rng(20261002)
            sigma = np.array([rng.permutation(4) for _ in range(SIDE * SIDE)])
        else:
            f = glob.glob(ROOT + EVOLVED[key]); assert len(f) == 1, f
            sigma = np.load(f[0])
        T = twisted(T0, walk, sigma); chi = chi_twist(sigma, walk); cyc = habit_cycle(sigma)
    worlds[name] = dict(T=T, sigma=sigma, chi=chi, **cyc)
    print(f"{name:24s} χ {chi:.2f}  dominant label {cyc['dominant']}  home cycle {[(c // SIDE, c % SIDE) for c in cyc['cycle']]}")


def load_solution(name, beta):
    f = DEFAULT_OUT / f"four_rooms-13x13-det-0.97-{TAG[name]}-pooled-b-{beta:g}.npz"
    d = np.load(f)
    # the report's certificate is 1e-8 / 1e-6; one evolved goal at beta 1 stops at 1e-7 / 1e-5,
    # which is still far below anything the triple statistics can see
    assert d["marginal_residual"].max() < 1e-5 and d["bellman_residual"].max() < 1e-3, (name, beta)
    n_uncert = int(((d["marginal_residual"] > 1e-8) | (d["bellman_residual"] > 1e-6)).sum())
    if n_uncert:
        print(f"  {name} β {beta:g}: {n_uncert} goal(s) below the report certificate", flush=True)
    return d

# %% [markdown]
# ## Per-triple quantities
#
# For one interim goal s' (all starts s and goals g at once): the defect, the shared-prior
# slack and prior advantage (one stochastic fixed-prior solve per goal prior), the bound
# C_rescoring/beta (vacuous where the goal prior lacks a label the first leg uses), the
# policy-disagreement gap C_disagree/beta, and the early-arrival probability.

# %%
def kl_policy(p, q):
    """KL(p(.|x) || q(.|x)) per state, bits; inf where p uses an action q lacks."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * (np.log2(np.maximum(p, 1e-300)) - np.log2(q)), 0.0)
    return t.sum(-1)


def fixed_prior_solve(Tsp, sp, q, beta, pi0):
    """Optimal first-leg cost at the prior q, for stochastic dynamics.

    Soft policy iteration warm-started from the first-leg policy restricted to q's
    support (the solver's default start, the prior as a policy, can have hitting
    times of 1e20 and returns garbage from the evaluation solve); soft value
    iteration as the fallback, inf if it diverges (the interim goal cannot be
    reached with the labels q allows)."""
    pi = np.where(q[None, :] > 0, pi0, 0.0); rs = pi.sum(1, keepdims=True)
    pi = np.where(rs > 0, pi / np.maximum(rs, 1e-300), q[None, :])
    try:
        F, _ = fixed_prior_pi(Tsp, sp, q, beta, pi=pi, tol=1e-10, iters=500)
        if np.all(np.isfinite(F)) and F.min() >= -1e-6 and F.max() < 1e7:
            return F
    except (RuntimeError, np.linalg.LinAlgError, FloatingPointError, ValueError):
        pass
    F = np.zeros(len(Tsp))
    for _ in range(100000):
        Fb, _ = backup(Tsp, sp, q, F, beta)
        if np.max(np.abs(Fb - F)) < 1e-9 * max(1.0, float(Fb.max())):
            return Fb
        if Fb.max() > 1e7:
            break
        F = Fb
    return np.full(len(Tsp), np.inf)


def per_interim_goal(args):
    T, pol, Q, F, beta, sp = args
    Tsp = absorb(T, sp); pi1 = pol[sp]
    P1 = np.einsum("sa,saj->sj", pi1, Tsp)
    N1 = occupancy(P1, sp)                                   # [s, x]
    T1 = N1.sum(1)
    dF = F[:, [sp]] + F[[sp], :] - F                         # [s, g]
    dead = Q < 1e-6
    lq = np.where(dead, 0.0, np.log2(np.maximum(Q, 1e-300)))
    L = (pi1 @ lq[sp])[:, None] - pi1 @ lq.T                 # [x, g]: E_pi1 log q_sp / q_g
    C = (N1 @ L) / beta
    vac = (N1 @ (pi1 @ (dead & ~dead[sp][None, :]).T)) > 1e-9
    C = np.where(vac, np.inf, C)
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.stack([kl_policy(pi1, pol[g]) for g in range(n)], 1)   # [x, g]
    K = np.where(np.isfinite(K), K, 1e300)
    D = (N1 @ K) / beta; D = np.where(D > 1e200, np.inf, D)
    Fq = np.full((n, n), np.inf); okq = np.zeros(n, bool)
    for g in range(n):
        if g == sp:
            continue
        Fq[:, g] = fixed_prior_solve(Tsp, sp, Q[g], beta, pi1); okq[g] = bool(np.isfinite(Fq[:, g]).all())
    A = Fq - F[:, [sp]]                                       # prior advantage [s, g]
    slack = Fq + F[[sp], :] - F                               # shared-prior slack [s, g]
    early = np.zeros((n, n))
    for g in range(n):
        if g == sp:
            continue
        tr = (idx != sp) & (idx != g)
        h = np.linalg.solve(np.eye(tr.sum()) - P1[np.ix_(tr, tr)], P1[tr, g])
        early[tr, g] = h
    return sp, dict(dF=dF, C=C, D=D, A=A, slack=slack, early=early, T1=T1, okq=okq)


def compute(name, beta, pool):
    f = CACHE / f"{TAG[name]}-pooled-b-{beta:g}.npz"
    if f.exists():
        return dict(np.load(f))
    d = load_solution(name, beta); T = worlds[name]["T"]
    pol, Q, F = d["policies"], d["q"], d["F"]
    arrs = {k: np.zeros((n, n, n)) for k in ("dF", "C", "D", "A", "slack", "early")}   # [sp, s, g]
    T1 = np.zeros((n, n)); okq = np.zeros((n, n), bool)
    for sp, r in pool.imap_unordered(per_interim_goal, [(T, pol, Q, F, beta, sp) for sp in range(n)], chunksize=4):
        for k in arrs:
            arrs[k][sp] = r[k]
        T1[sp] = r["T1"]; okq[sp] = r["okq"]
    out = dict(F=F, Q=Q, T1=T1, okq=okq, **arrs)
    np.savez_compressed(f, **out)
    return out


def triple_mask(sp):
    return (idx[:, None] != sp) & (idx[None, :] != sp) & (idx[:, None] != idx[None, :])


MASK = np.stack([triple_mask(sp) for sp in range(n)])        # [sp, s, g]

# %%
results = {}
with Pool(10) as pool:
    for beta in BETAS:
        for name, _ in WORLDS:
            results[(name, beta)] = compute(name, beta, pool)
            print("computed", name, beta, flush=True)

# %% [markdown]
# ## Checks and summary statistics

# %%
def summarise(name, beta, r):
    m = MASK
    dF, C, D, A, S, E = (r[k][m] for k in ("dF", "C", "D", "A", "slack", "early"))
    saving = -dF
    viol = saving > 1e-7
    fin = np.isfinite(C)
    finA = np.isfinite(A)
    o = dict(world=name, chi=worlds[name]["chi"], beta=beta, triples=int(m.sum()),
             violating=int(viol.sum()), violating_share=float(viol.mean()), worst_violation=float(saving.max()),
             bound_vacuous_share=float((~fin).mean()),
             bound_fails=int(((C < saving - 1e-7) & fin).sum()),
             advantage_unsolved_share=float((~finA).mean()),
             slack_min=float(S[finA].min()), slack_negative=int(((S < -1e-7) & finA).sum()),
             identity_prior_slack_max_residual=float(np.abs(dF[finA] - (S[finA] - A[finA])).max()),
             early_median=float(np.median(E)), early_p90=float(np.percentile(E, 90)),
             early_violators_median=float(np.median(E[viol])) if viol.any() else float("nan"))
    ok = fin & np.isfinite(D)
    res = np.abs(saving - (C - D))[ok]          # C and D already carry the 1/beta
    o["exact_defect_residual_median"] = float(np.median(res)); o["exact_defect_residual_p99"] = float(np.percentile(res, 99))
    o["exact_defect_residual_corr_with_early"] = float(np.corrcoef(res, E[ok])[0, 1]) if ok.sum() > 2 else float("nan")

    def q(x, ps=(10, 50, 90)):
        x = x[np.isfinite(x)]
        return [float(np.percentile(x, p)) for p in ps] if x.size else [float("nan")] * len(ps)
    o["spectrum_all"] = dict(slack=q(S), advantage=q(A), bound=q(C), gap=q(D), saving=q(saving))
    o["spectrum_violators"] = dict(slack=q(S[viol]), advantage=q(A[viol]), bound=q(C[viol]), gap=q(D[viol]), saving=q(saving[viol]))
    if viol.any():
        v = viol & fin; rb = C[v] / saving[v]; ra = A[viol & finA] / saving[viol & finA]
        top = saving[v] >= np.percentile(saving[v], 99)
        o["ratio_bound_median"] = float(np.median(rb)); o["ratio_bound_top1pct_median"] = float(np.median(rb[top]))
        o["ratio_advantage_median"] = float(np.median(ra))
        o["violators_bound_vacuous_share"] = float((~fin[viol]).mean())
    # best interim goal per start-goal pair
    best = np.full((n, n), np.inf); arg = np.full((n, n), -1)
    for sp in range(n):
        d = np.where(MASK[sp], r["dF"][sp], np.inf)
        better = d < best; arg[better] = sp; best = np.where(better, d, best)
    off = ~np.eye(n, dtype=bool); helped = (best < -1e-7) & off
    o["pairs_helped_share"] = float(helped[off].mean())
    cyc = set(pos[c] for c in worlds[name]["cycle"] if c in pos)
    bas = set(pos[c] for c in worlds[name]["basin"] if c in pos)
    b = arg[helped]
    o["best_at_doorway_share"] = float(near_door[b].mean()) if b.size else float("nan")
    o["best_on_habit_cycle_share"] = float(np.isin(b, list(cyc)).mean()) if b.size and cyc else float("nan")
    o["best_in_dominant_basin_share"] = float(np.isin(b, list(bas)).mean()) if b.size and bas else float("nan")
    o["habit_cycle_share_of_states"] = len(cyc) / n; o["dominant_basin_share_of_states"] = len(bas) / n
    o["doorway_share_of_states"] = float(near_door.mean())
    # share of journeys cheaper via each state (Figure 1 left)
    share = np.array([100.0 * ((r["dF"][sp] < -1e-7) & MASK[sp]).sum() / ((n - 1) * (n - 2)) for sp in range(n)])
    return o, share, arg


summary = {}; shares = {}; args_best = {}
for (name, beta), r in results.items():
    o, share, arg = summarise(name, beta, r)
    summary[(name, beta)] = o; shares[(name, beta)] = share; args_best[(name, beta)] = arg
    print(f"{name:24s} β {beta:<4g} viol {100*o['violating_share']:.2f}% worst {o['worst_violation']:.2f} | bound fails {o['bound_fails']} vacuous {100*o['bound_vacuous_share']:.0f}% "
          f"| slack min {o['slack_min']:.1e} neg {o['slack_negative']} | ratio bound med {o.get('ratio_bound_median', float('nan')):.2f} top1% {o.get('ratio_bound_top1pct_median', float('nan')):.2f} "
          f"| early med {o['early_median']:.1e} | best at door {100*o['best_at_doorway_share']:.0f}% on cycle {100*(o['best_on_habit_cycle_share'] or 0):.0f}%", flush=True)
(OUT / "twisted-summary.json").write_text(json.dumps([summary[k] for k in summary], indent=1) + "\n")

# %% [markdown]
# ## Figure A: where an interim goal helps, with the habit cycle marked

# %%
def grid_data():
    return dict(side=SIDE, walkable=[int(s) for s in walk])


gd = grid_data()
fig, axes = plt.subplots(1, len(WORLDS), figsize=(4.4 * len(WORLDS), 4.6), dpi=170)
top = max(shares[(name, 1.0)].max() for name, _ in WORLDS)
for ax, (name, _) in zip(axes, WORLDS):
    share = dict(zip(gd["walkable"], shares[(name, 1.0)]))
    draw_grid(ax, gd, lambda s: SEQ(share[s] / top))
    for c in worlds[name]["cycle"]:
        x, y = xy(gd, c)
        ax.add_patch(plt.Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, fill=False, edgecolor=OUTLINE, linewidth=2.2, zorder=20))
    o = summary[(name, 1.0)]
    ax.set_title(f"{name}, χ {worlds[name]['chi']:.2f}\n{100*o['violating_share']:.1f}% of triples violate", fontsize=9.5)
sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, top))
cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02); cb.set_label("% of journeys that are cheaper via this state", fontsize=9)
cb.outline.set_visible(False)
fig.savefig(OUT / "twisted-share-maps.png", bbox_inches="tight"); fig.savefig(OUT / "twisted-share-maps.pdf", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Figure B: the Figure 1 journey in each world — defect, slack, advantage, bound, gap

# %%
def most_likely_route(T, pol, a, b):
    succ = T.argmax(2); path = [a]
    while a != b and len(path) < 300:
        a = int(succ[a, int(pol[b][a].argmax())]); path.append(a)
    return path


s_i, g_i = pos[START[0] * SIDE + START[1]], pos[GOAL[0] * SIDE + GOAL[1]]
fig, axes = plt.subplots(len(WORLDS), 5, figsize=(14.5, 3.25 * len(WORLDS)), dpi=170, gridspec_kw=dict(wspace=0.08, hspace=0.3))
for r_, (name, _) in enumerate(WORLDS):
    r = results[(name, 1.0)]; T = worlds[name]["T"]; pol = load_solution(name, 1.0)["policies"]
    vals = {k: r[k][:, s_i, g_i].copy() for k in ("dF", "slack", "A", "C", "D")}      # indexed by sp
    for k in vals:
        vals[k][[s_i, g_i]] = np.nan
    w = int(np.nanargmin(vals["dF"]))
    route = most_likely_route(T, pol, s_i, w) + most_likely_route(T, pol, w, g_i)[1:]
    fin = lambda v: v[np.isfinite(v)]
    dmax = max(abs(np.nanmin(fin(vals["dF"]))), np.nanmax(fin(vals["dF"]))); dnorm = TwoSlopeNorm(0.0, -dmax, dmax)
    top2 = max(np.nanmax(fin(vals[k])) for k in ("slack", "A", "C", "D")); snorm = PowerNorm(0.5, 0.0, top2)
    panels = (("dF", DEFECT, dnorm, "defect via s′"), ("slack", BLUES, snorm, "shared-prior slack"),
              ("A", REDS, snorm, "prior advantage"), ("C", REDS, snorm, "bound C_rescoring/β"), ("D", BLUES, snorm, "gap C_disagree/β"))
    for c_, (k, cmap, norm, title) in enumerate(panels):
        ax = axes[r_, c_]; v = dict(zip(gd["walkable"], vals[k]))

        def colour(s, k=k, cmap=cmap, norm=norm, v=v):
            x = v[s]
            if s in (gd["walkable"][s_i], gd["walkable"][g_i]) or x is None or np.isnan(x):
                return "#d9d9d9"
            if not np.isfinite(x):
                return "#555555"            # vacuous bound or unsolved advantage
            if k != "dF" and x < 0:
                return NEGATIVE
            return cmap(norm(x if k == "dF" else max(x, 0.0)))
        draw_grid(ax, gd, colour)
        for s in gd["walkable"]:
            x = v[s]
            if k != "dF" and x is not None and np.isfinite(x) and x < 0:
                px, py = xy(gd, s); ax.add_patch(plt.Rectangle((px - .5, py - .5), 1, 1, fill=False, hatch="////", edgecolor="#7a7a7a", linewidth=0, zorder=3))
        for c in worlds[name]["cycle"]:
            x, y = xy(gd, c); ax.add_patch(plt.Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, fill=False, edgecolor=OUTLINE, linewidth=1.4, ls=":", zorder=19))
        xs, ys = zip(*[xy(gd, int(walk[q])) for q in route])
        ax.plot(xs, ys, color="white", lw=3.2, solid_capstyle="round", zorder=8); ax.plot(xs, ys, color="black", lw=1.3, solid_capstyle="round", zorder=9)
        mark_journey(ax, s=xy(gd, int(walk[s_i])), sp=xy(gd, int(walk[w])), g=xy(gd, int(walk[g_i])), size=0.8, fontsize=7.5)
        ax.set_title((f"{name}, χ = {worlds[name]['chi']:.2f}\n" if c_ == 0 else "") + title, fontsize=9.5)
    sav = -vals["dF"][w]
    axes[r_, 0].text(0.02, -0.06, f"best s′ saves {sav:.2f}; slack {vals['slack'][w]:.2f}, advantage {vals['A'][w]:.2f}, bound {vals['C'][w]:.2f}, gap {vals['D'][w]:.2f}",
                     transform=axes[r_, 0].transAxes, fontsize=8, va="top")
fig.text(0.01, 0.01, "grey cells: start and goal; dark grey: vacuous bound or unsolvable advantage (goal prior lacks a label the leg needs); hatched: negative; dotted gold: habit cycle", fontsize=8, color="#555")
fig.savefig(OUT / "twisted-journey-maps.png", bbox_inches="tight"); fig.savefig(OUT / "twisted-journey-maps.pdf", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Figure C: how loose the bound is (the report's Figure 7, per world)

# %%
BINS = np.logspace(-3, np.log10(200), 14); MINB = 20
BCOL = LinearSegmentedColormap.from_list("beta", ["#0d366b", "#d6453a"])


def binned(Dv, X):
    r = X / Dv; mids, med, lo, hi, mn = [], [], [], [], []
    for a, c in zip(BINS[:-1], BINS[1:]):
        m = (Dv >= a) & (Dv < c) & np.isfinite(r)
        if m.sum() < MINB:
            continue
        mids.append(np.sqrt(a * c)); med.append(np.median(r[m])); lo.append(np.percentile(r[m], 10)); hi.append(np.percentile(r[m], 90)); mn.append(r[m].min())
    return map(np.array, (mids, med, lo, hi, mn))


fig, axes = plt.subplots(len(WORLDS), 2, figsize=(8.6, 3.5 * len(WORLDS)), sharey=True, dpi=170)
for r_, (name, _) in enumerate(WORLDS):
    for i, beta in enumerate(sorted(BETAS)):
        r = results[(name, beta)]; m = MASK; sav = -r["dF"][m]; viol = sav > 1e-7
        col = BCOL(i / max(1, len(BETAS) - 1))
        for c_, (key, lab_) in enumerate((("C", "bound ÷ violation"), ("A", "advantage ÷ violation"))):
            X = r[key][m][viol]; Dv = sav[viol]
            mids, med, lo, hi, mn = binned(Dv, X); ax = axes[r_, c_]
            if len(mids):
                ax.fill_between(mids, lo, hi, color=col, alpha=0.18, lw=0); ax.plot(mids, med, color=col, lw=1.8, label=f"β = {beta:g}  ({viol.sum():,} violating)" if c_ == 0 else None)
                ax.plot(mids, mn, color=col, lw=0.8, ls=":")
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-3, 200); ax.set_ylim(0.7, 3e4)
            ax.axhline(1.0, color="#222", lw=0.9, ls="--"); ax.text(150, 1.15, "tight", ha="right", fontsize=8)
            ax.set_title(f"{name}, χ {worlds[name]['chi']:.2f}: {('the bound' if key == 'C' else 'the re-optimised advantage')}", fontsize=9.5, loc="left")
            ax.set_ylabel(lab_ if True else ""); ax.set_xlabel("size of the violation  −ΔF, free-energy units (bits at β = 1)" if r_ == len(WORLDS) - 1 else "")
            ax.grid(True, which="major", color="#e8e8e8", lw=0.6); ax.set_axisbelow(True)
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
    axes[r_, 0].legend(frameon=False, fontsize=7.5, loc="upper right")
fig.text(0.01, 0.005, f"line: median per bin; band: 10th to 90th percentile; dotted: minimum; bins with fewer than {MINB} triples omitted; vacuous (infinite) bounds excluded", fontsize=7.5, color="#555")
fig.tight_layout(rect=(0, 0.015, 1, 1))
fig.savefig(OUT / "twisted-bound-ratio.png"); fig.savefig(OUT / "twisted-bound-ratio.pdf")
plt.close(fig)

# %% [markdown]
# ## Figure D: the spectrum over violating triples at β = 1

# %%
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), dpi=170)
WCOL = {"untwisted": "#555555", "random twist": "#9aa1ab", "evolved, lowest F": "#b8860b", "evolved, high χ, warm": "#d6453a", "evolved, highest χ": "#2a78d6"}
for (k, title) in zip(("slack", "A", "C", "D"), ("shared-prior slack", "prior advantage", "bound C_rescoring/β", "gap C_disagree/β")):
    ax = axes[("slack", "A", "C", "D").index(k)]
    for name, _ in WORLDS:
        r = results[(name, 1.0)]; m = MASK; viol = (-r["dF"][m]) > 1e-7
        x = r[k][m][viol]; x = x[np.isfinite(x)]; x = np.sort(np.maximum(x, 1e-4))
        ax.plot(x, np.linspace(0, 1, len(x)), color=WCOL[name], lw=1.6, label=f"{name} (n = {viol.sum():,})")
    ax.set_xscale("log"); ax.set_xlabel(f"{title}, bits"); ax.set_title(title, fontsize=10, loc="left")
    ax.grid(True, color="#e8e8e8", lw=0.6); ax.set_axisbelow(True)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
axes[0].set_ylabel("share of violating triples below"); axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "twisted-spectra.png"); fig.savefig(OUT / "twisted-spectra.pdf")
plt.close(fig)
print("figures written to", OUT)
