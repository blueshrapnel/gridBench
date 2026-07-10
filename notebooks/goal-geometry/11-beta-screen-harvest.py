# %% [markdown]
# # Goal-geometry series, 11 — harvesting the beta screen
#
# The 09-07 beta screen evolved twists AT beta 0.5 and 0.3 (gridcore
# kernel, g200 pop-96, seed-matched to the beta=1 smoke anchors): five
# 7x7 worlds + four_rooms 9x9, thirteen run-bests.  Three questions,
# all queued by earlier notebooks:
#
#   Q1 (typology): does the coverage typology hold when evolution
#      happens at low beta — and does the asymmetric mode (dominant
#      coverage >= 0.70 plus a silenced label) appear more or less
#      often?
#   Q2 (triangle, nb04's live question): the Cartesian violation gap
#      grows as beta falls (1.1 -> 3.4 -> 11.0), and the beta=1
#      exemplar AMPLIFIES violations 2.4x.  Do beta-EVOLVED twists
#      amplify their own (beta-matched) geometry more, less, or the
#      same?
#   Q3 (marginals, nb10's open question): prior specialisation is the
#      whole triangle violation, and its cheap proxy is the pairwise
#      KL between per-goal action marginals (r = 0.92 with the credit).
#      Does evolving at low beta widen or narrow the marginal-
#      divergence distribution relative to the beta-matched Cartesian
#      frame?
#
# Everything is evaluated AT THE TWIST'S OWN EVOLUTION BETA, with fresh
# Cartesian baselines at the same beta — cross-beta comparisons are
# meaningless otherwise (marginals sharpen as beta rises regardless of
# the labelling).

# %%
import json
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
DET, THETA, TOL = 0.97, 1e-5, 1e-6
REF_FLOOR = 1e-6

RUN_RE = re.compile(
    r"b(?P<beta>\d+)-free-core-ga-(?P<env>[a-z-]+)-(?P<shape>\d+x\d+)"
    r"-s(?P<seed>\d+)-(?P<host>sp3011-\d+)$")


def parse_runs():
    out = []
    for d in sorted(SCREEN.iterdir()):
        m = RUN_RE.search(d.name)
        if not m:
            continue
        beta = {"03": 0.3, "05": 0.5}[m["beta"]]
        env_id = m["env"].replace("-", "_")
        hw = int(m["shape"].split("x")[0])
        sig = np.load(next(d.glob("*multi-all.sigma.npy")))
        summ = json.load(open(next(d.glob("*multi-all.summary.json"))))
        out.append(dict(dir=d.name, beta=beta, env_id=env_id,
                        shape=(hw, hw), seed=m["seed"],
                        sigma=np.asarray(sig, dtype=int), summary=summ))
    return out


RUNS = parse_runs()
print(f"{len(RUNS)} screen runs loaded")
for r in RUNS:
    s = r["summary"]
    print(f"  {r['env_id']:10s} {r['shape'][0]}x{r['shape'][0]} b{r['beta']} "
          f"s{r['seed'][:3]}  kernel={s.get('eval_kernel')}  "
          f"best={s.get('best_fitness', s.get('best_mean_free', float('nan')))}")

# %% [markdown]
# ## Machinery (nb04/nb08/nb10, parameterised by environment)

# %%
def build(env_id, shape, sigma, goal, beta):
    cfg = EvalConfig(env_id=env_id, shape=shape, goal=int(goal), beta=beta,
                     determinism=DET, manhattan=True, theta=THETA,
                     state_dist="uniform")
    return build_twisted_env_from_sigma(sigma, cfg)


def solve_frame(env_id, shape, sigma, beta):
    """D matrix, per-goal policies, per-goal converged marginals."""
    nS = shape[0] * shape[1]
    env0 = build(env_id, shape, sigma, 0, beta)
    goals = [int(s) for s in env0.available_states]
    n = len(goals)
    D = np.zeros((n, n))
    policies = np.zeros((n, nS, 4))
    p_hat = np.zeros((n, 4))
    failed = []
    for j, g in enumerate(goals):
        env = build(env_id, shape, sigma, g, beta)
        di = DecisionInformation(env, _state_dist_class("uniform")(env), THETA,
                                 max_iterations=200_000,
                                 max_info_iterations=10_000)
        pi, _, F = di.get_opt_policy_Z_free_vector(beta)
        if not di.converged:
            failed.append(g)
        D[:, j] = np.asarray(F, dtype=float)[goals]
        pi = np.asarray(pi, dtype=float)
        policies[j] = pi
        p_s = np.asarray(di.state_dist.set_ps(pi), dtype=float)
        p_hat[j] = pi.T @ p_s
    np.fill_diagonal(D, 0.0)
    return dict(D=D, goals=goals, policies=policies, p_hat=p_hat,
                failed=failed, nS=nS)


def triangle_stats(D):
    n = D.shape[0]
    gap = D[:, None, :] - (D[:, :, None] + D[None, :, :])
    A, B, C = np.ogrid[:n, :n, :n]
    valid = (A != B) & (B != C) & (A != C)
    viol = valid & (gap > TOL)
    return dict(frac=float(viol.sum() / valid.sum()),
                maxgap=float(gap[viol].max()) if viol.any() else 0.0)


def pairwise_marginal_kl(p_hat):
    ref = p_hat * (1 - REF_FLOOR) + REF_FLOOR / 4
    ref = ref / ref.sum(axis=1, keepdims=True)
    lg = np.log2(ref)
    KL = (ref[:, None, :] * (lg[:, None, :] - lg[None, :, :])).sum(-1)
    return KL[~np.eye(KL.shape[0], dtype=bool)]


def label_coverages(env_id, shape, sigma):
    """Largest-basin fraction per label, goal-free-exact successors."""
    nS = shape[0] * shape[1]
    env0 = build(env_id, shape, sigma, 0, 1.0)
    goals = [int(s) for s in env0.available_states]
    T = np.asarray(env0.get_T(), dtype=float).reshape(nS, 4, nS)
    succ = T.argmax(axis=2)
    env1 = build(env_id, shape, sigma, goals[1], 1.0)
    succ[0] = np.asarray(env1.get_T(), dtype=float).reshape(nS, 4, nS).argmax(axis=2)[0]
    fracs = []
    for li in range(4):
        cycle_id = {}
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
        fracs.append(max(counts.values()) / len(goals))
    return np.array(fracs)


def label_usage(policies, goals):
    """Greedy-decision share per label over all (state, goal) pairs."""
    usage = np.zeros(4)
    for j in range(len(goals)):
        greedy = policies[j].argmax(axis=1)
        for k, s in enumerate(goals):
            if k != j:
                usage[greedy[s]] += 1
    return usage / usage.sum()


# %% [markdown]
# ## The sweep: every screen twist at its own beta + Cartesian baselines

# %%
cart_cache = {}
rows = []
for r in RUNS:
    env_id, shape, beta = r["env_id"], r["shape"], r["beta"]
    nS = shape[0] * shape[1]
    identity = np.tile(np.arange(4, dtype=np.int64), (nS, 1))

    key = (env_id, shape, beta)
    if key not in cart_cache:
        fr_c = solve_frame(env_id, shape, identity, beta)
        cart_cache[key] = dict(tri=triangle_stats(fr_c["D"]),
                               kl=pairwise_marginal_kl(fr_c["p_hat"]),
                               failed=fr_c["failed"])
    cart = cart_cache[key]

    fr = solve_frame(env_id, shape, r["sigma"], beta)
    tri = triangle_stats(fr["D"])
    kl = pairwise_marginal_kl(fr["p_hat"])
    cov = label_coverages(env_id, shape, r["sigma"])
    usage = label_usage(fr["policies"], fr["goals"])
    order = np.argsort(usage)[::-1]
    asym = bool(cov.max() >= 0.70 and usage.min() <= 0.02
                and cov[np.argmin(usage)] <= 0.25)
    rows.append(dict(env=env_id, hw=shape[0], beta=beta, seed=r["seed"][:3],
                     cov_dom=float(cov.max()), cov_sorted=cov[order],
                     usage_sorted=usage[order], asym=asym,
                     viol_tw=tri["frac"], viol_ca=cart["tri"]["frac"],
                     gap_tw=tri["maxgap"], gap_ca=cart["tri"]["maxgap"],
                     kl_tw=float(np.median(kl)),
                     kl_ca=float(np.median(cart["kl"])),
                     failed=len(fr["failed"]) + len(cart["failed"])))
    x = rows[-1]
    print(f"{env_id:10s} {shape[0]}x{shape[0]} b{beta} s{x['seed']}  "
          f"cov {x['cov_dom']:.2f} asym={'Y' if x['asym'] else 'n'}  "
          f"viol {x['viol_tw']:5.1%} (cart {x['viol_ca']:5.1%})  "
          f"maxgap {x['gap_tw']:6.2f} ({x['gap_ca']:6.2f})  "
          f"KLmed {x['kl_tw']:6.3f} ({x['kl_ca']:6.3f})"
          + ("  [UNCONVERGED solves!]" if x["failed"] else ""))

# %% [markdown]
# ## Aggregates against the beta=1 anchors

# %%
# beta=1 reference points from earlier notebooks (four_rooms 7x7):
#   Cartesian viol 1.1%, max gap 3.43, KL median 0.79
#   GA exemplar (cov 0.85): viol 3.3%, max gap 6.59, KL median 1.24
for beta in (0.5, 0.3):
    sel = [x for x in rows if x["beta"] == beta]
    amp = [x["viol_tw"] / x["viol_ca"] for x in sel if x["viol_ca"] > 0]
    flat_made = [x for x in sel if x["viol_ca"] == 0 and x["viol_tw"] > 0]
    klr = [x["kl_tw"] / x["kl_ca"] for x in sel]
    print(f"beta={beta}: n={len(sel)}  asym {sum(x['asym'] for x in sel)}/{len(sel)}  "
          f"cov_dom median {np.median([x['cov_dom'] for x in sel]):.2f}  "
          f"amplification median {np.median(amp):.2f}x  "
          f"KL ratio median {np.median(klr):.2f}  "
          f"flat worlds given violations: {len(flat_made)}")

# %% [markdown]
# ## Figures

# %%
env_marks = {"four_rooms": "o", "wrap_grid": "s", "open_grid": "D",
             "pinwheel": "^", "helical": "v"}
beta_cols = {0.3: "#cc3311", 0.5: "#ee7733", 1.0: "#4477aa"}
fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8), dpi=150)

ax = axes[0]
for x in rows:
    ax.scatter(x["viol_ca"] * 100, x["viol_tw"] * 100,
               marker=env_marks[x["env"]], s=70 if x["hw"] == 9 else 45,
               color=beta_cols[x["beta"]], alpha=0.85)
ax.scatter([1.1], [3.3], marker="o", s=45, color=beta_cols[1.0])
lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([0, lim], [0, lim], "--", color="black", lw=0.8)
ax.set_xlabel("Cartesian violating triples (%)")
ax.set_ylabel("evolved-twist violating triples (%)")
ax.set_title("Q2: do beta-evolved twists amplify\ntheir own geometry? "
             "(above line = yes)")
from matplotlib.lines import Line2D  # noqa: E402

hnd = [Line2D([], [], marker=m, ls="", color="grey", label=e)
       for e, m in env_marks.items()]
hnd += [Line2D([], [], marker="o", ls="", color=c, label=f"$\\beta$={b}")
        for b, c in beta_cols.items()]
ax.legend(handles=hnd, fontsize=7, ncol=2)

ax = axes[1]
for x in rows:
    ax.scatter(x["kl_ca"], x["kl_tw"], marker=env_marks[x["env"]],
               s=70 if x["hw"] == 9 else 45, color=beta_cols[x["beta"]],
               alpha=0.85)
ax.scatter([0.79], [1.24], marker="o", s=45, color=beta_cols[1.0])
lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([0, lim], [0, lim], "--", color="black", lw=0.8)
ax.set_xlabel("Cartesian pairwise-marginal KL, median (bits)")
ax.set_ylabel("evolved-twist KL, median (bits)")
ax.set_title("Q3: marginal divergence,\ntwist vs beta-matched Cartesian")

ax = axes[2]
for x in rows:
    ax.scatter(x["beta"] + (0.012 if x["env"] == "four_rooms" else 0),
               x["cov_dom"], marker=env_marks[x["env"]],
               s=95 if x["asym"] else 45,
               facecolors="none" if not x["asym"] else beta_cols[x["beta"]],
               edgecolors=beta_cols[x["beta"]], linewidths=1.6)
ax.set_xticks([0.3, 0.5], ["$\\beta$=0.3", "$\\beta$=0.5"])
ax.set_xlim(0.2, 0.62)
ax.set_ylabel("dominant-label coverage")
ax.set_title("Q1: coverage typology at low beta\n(filled = asymmetric mode)")

fig.suptitle("Harvesting the beta screen: twists evolved at "
             "$\\beta \\in \\{0.5, 0.3\\}$, evaluated at their own beta")
fig.tight_layout()
fig.savefig(Path(__file__).parent / "figs" / "F-beta-screen-harvest.png"
            if "__file__" in dir() else "figs/F-beta-screen-harvest.png",
            dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Discernments (2026-07-10 run, gridcore d432c4b)
#
# - **Headline (Q2): low-beta evolution manufactures violations on FLAT
#   geometry.**  Cartesian wrap_grid and helical have zero violating
#   triples at every beta and, by torus symmetry, IDENTICAL per-goal
#   marginals (pairwise KL exactly 0).  Their evolved twists saturate
#   completely (coverage 1.00) -- and their geometry now violates on
#   3.3-3.9% of triples with gaps of 6-8.8 bits, marginal KL medians
#   5.6-8.3 bits.  nb04's "bent, not flattened" in the extreme: the
#   labelling alone creates the marginal-incompatibility structure that
#   walls create elsewhere.  Segmentation affordance is a property of
#   the REPRESENTATION, not just the world.
#
# - **Pinwheel shows the first ABSORPTION.**  Cartesian pinwheel at low
#   beta is the most violation-rich geometry measured (10.5-11.8% of
#   triples -- 10x four_rooms) and its evolved twists REDUCE the rate
#   (8.1%/9.6%).  Amplify-vs-absorb tracks how violation-rich the base
#   geometry already is: four_rooms amplifies (1.3-3.1x at own beta),
#   pinwheel partially absorbs, flat worlds get violations created.
#
# - **Q3 answered: evolving at low beta WIDENS marginal divergence,
#   dramatically.**  Walled-world twist/Cartesian KL ratios run ~3-7x
#   (four_rooms 7x7: 4.3-6.9x) against the beta=1 exemplar's 1.6x, and
#   the flat worlds go from exactly 0 to 5.6-8.3 bits.  Low beta makes
#   deviation from the prior expensive; evolution responds by giving
#   each goal region its own vocabulary -- prior specialisation as a
#   design strategy, exactly the mechanism nb09/nb10 identified.
#
# - **Q1 typology**: flat worlds fully saturate even at low beta
#   (cov 1.00); walled and pinwheel spread bimodally (0.32-0.85);
#   formal asymmetric mode only 1/13 (four_rooms 7x7 b0.3 s314,
#   cov 0.85).  Coverage median at beta=0.5 (0.64) sits BELOW beta=0.3
#   (0.82) -- a non-monotonicity hint, but at one seed per cell it
#   awaits the v2 seed expansion (dispatched 10-07) for confirmation.
#
# - **Caveat**: the four_rooms 9x9 b0.3 row had 2/69 forced-exit DI
#   solves in the twisted frame (residual_U ~0.6) -- the low-beta
#   float64 wall encroaching at 9x9.  Treat that row as provisional
#   until the log-domain solver (gridCore#1) exists.
#
# - Next: v2 seed expansion lands 2-3 seeds per cell -> redo the
#   aggregates with error bars; PS decompose (nb10 machinery) on the
#   wrap/helical saturated twists -- on-flow PS should be ~zero there
#   by construction, making the flat-world violations a pure
#   off-flow/marginal-boundary effect.
