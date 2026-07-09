# %% [markdown]
# # Goal-geometry series, 08 — a usage-weighted coverage fingerprint (EXPLORATION)
#
# Branch explore/weighted-coverage.  Karen's idea: paper-1's policy view
# (its figure 12) showed the label USAGE distribution -- 39/39/21/1% of
# greedy decisions across all goals -- while coverage weights every label
# equally and so counts the silenced label's anatomy that behaviour never
# exercises.  Candidates explored here, per twist:
#
#   cov      = max_l basin_frac(l)                       (current metric)
#   wcov     = sum_l usage(l) * basin_frac(l)            (usage-weighted)
#   cov3     = max over the 3 MOST-USED labels           (drop the retiree)
#   mean_use = sum_l usage(l) * basin_frac(l) / max ...  see below
#
# Everything is computed fresh in-notebook (gridcore solves + a local
# functional-graph decompose): NO gridFour fingerprint code, NO metric
# caches touched.  Cohort: four_rooms 7x7 GA run-bests (schema export,
# read-only) + Cartesian + a uniform-random null.

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
N_NULL = 24
RNG = np.random.default_rng(20260709)
SCHEMA_EXPORT = Path("/media/merlin/grid-twist/data-schema-10/multi/schema10-export")


def build(sigma, goal):
    cfg = EvalConfig(env_id="four_rooms", shape=SHAPE, goal=int(goal), beta=BETA,
                     determinism=DET, manhattan=True, theta=THETA,
                     state_dist="uniform")
    return build_twisted_env_from_sigma(sigma, cfg)


def ga_sigmas():
    out = []
    root = SCHEMA_EXPORT / "shape=7x7" / "env_id=four_rooms"
    for p in sorted(root.rglob("*beta=1/*/*.pickle")):
        blob = pickle.load(open(p, "rb"))
        prov = blob.get("provenance", {}) if isinstance(blob.get("provenance"), dict) else {}
        rn = (prov.get("run_name") or "").lower()
        if "fepm" in rn or "random" in rn or "rand-" in rn:
            continue
        out.append(np.asarray(blob["sigma"], dtype=int))
    return out


def random_sigma():
    out = np.tile(np.arange(4, dtype=np.int64), (nS, 1))
    for s in range(nS):
        out[s] = RNG.permutation(4)
    return out


def basin_fracs(succ_by_label, nonwall):
    """Largest-basin fraction per label from intended-move successor maps.

    Plain-numpy functional-graph decompose: follow orbits to find each
    state's terminal cycle; states sharing a cycle share a basin.
    """
    nonwall_set = set(nonwall)
    fracs = []
    for li in range(4):
        succ = succ_by_label[:, li]
        cycle_id = {}
        for s0 in nonwall:
            if s0 in cycle_id:
                continue
            path, seen = [], {}
            s = s0
            while True:
                if s in cycle_id:      # joins a known basin
                    cid = cycle_id[s]
                    break
                if s in seen:          # new cycle found
                    cyc = path[seen[s]:]
                    cid = min(cyc)
                    for c in cyc:
                        cycle_id[c] = cid
                    break
                seen[s] = len(path)
                path.append(s)
                s = int(succ[s])
                if s not in nonwall_set:   # stepped onto wall: self-loop guard
                    cid = path[-1]
                    break
            for q in path:
                if q not in cycle_id:
                    cycle_id[q] = cid
        counts = {}
        for s in nonwall:
            counts[cycle_id[s]] = counts.get(cycle_id[s], 0) + 1
        fracs.append(max(counts.values()) / len(nonwall))
    return np.array(fracs)


def evaluate_twist(sigma):
    """Per-label basin fractions + per-label greedy usage over all goals."""
    env0 = build(sigma, 0)
    goals = [int(s) for s in env0.available_states]
    T = np.asarray(env0.get_T(), dtype=float)
    succ = T.reshape(nS, 4, nS).argmax(axis=2)
    fracs = basin_fracs(succ, goals)

    usage = np.zeros(4)
    for g in goals:
        env = build(sigma, g)
        di = DecisionInformation(env, _state_dist_class("uniform")(env), THETA,
                                 max_iterations=200_000,
                                 max_info_iterations=10_000)
        pi, _, _ = di.get_opt_policy_Z_free_vector(BETA)
        greedy = np.argmax(np.asarray(pi, dtype=float), axis=1)
        for s in goals:
            if s != g:
                usage[greedy[s]] += 1
    usage = usage / usage.sum()
    order = np.argsort(usage)[::-1]
    return {
        "cov": float(fracs.max()),
        "wcov": float((usage * fracs).sum()),
        "cov3": float(fracs[order[:3]].max()),
        "wcov3": float((usage[order[:3]] * fracs[order[:3]]).sum()
                       / usage[order[:3]].sum()),
        "usage": usage[order],
        "fracs_by_usage": fracs[order],
    }


# %%
print("evaluating cohort...")
records = []
for kind, sigmas in (("GA", ga_sigmas()),
                     ("cartesian", [np.tile(np.arange(4, dtype=np.int64), (nS, 1))]),
                     ("random", [random_sigma() for _ in range(N_NULL)])):
    for sig in sigmas:
        r = evaluate_twist(sig)
        r["kind"] = kind
        records.append(r)
    print(f"  {kind}: {len(sigmas)} twists done")

# %% [markdown]
# ## Does usage correlate with anatomy anyway?

# %%
ga = [r for r in records if r["kind"] == "GA"]
u = np.concatenate([r["usage"] for r in ga])
f = np.concatenate([r["fracs_by_usage"] for r in ga])
r_uf = np.corrcoef(u, f)[0, 1]
print(f"corr(usage share, basin fraction) across GA cohort labels: {r_uf:.3f}")
print(f"mean usage by rank: {np.mean([r['usage'] for r in ga], axis=0)}")
print(f"mean basin frac by usage rank: {np.mean([r['fracs_by_usage'] for r in ga], axis=0)}")

# %% [markdown]
# ## The candidate metrics side by side

# %%
KIND_STYLE = {"GA": ("#2166ac", 46, "o"), "cartesian": ("#d73027", 130, "D"),
              "random": ("#999999", 22, "o")}

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), dpi=150)

ax = axes[0]
for kind, (c, s, m) in KIND_STYLE.items():
    rs = [r for r in records if r["kind"] == kind]
    ax.scatter([r["cov"] for r in rs], [r["wcov"] for r in rs], color=c, s=s,
               marker=m, alpha=0.8, label=kind)
lims = [0, 1]
ax.plot(lims, lims, "--", color="black", lw=0.7)
ax.set_xlabel("coverage (current)"); ax.set_ylabel("usage-weighted coverage")
ax.set_title("wcov vs cov: weighting separates\nanatomy exercised from anatomy idle")
ax.legend(fontsize=8)

ax = axes[1]
for kind, (c, s, m) in KIND_STYLE.items():
    rs = [r for r in records if r["kind"] == kind]
    ax.scatter([r["cov"] for r in rs], [r["cov3"] for r in rs], color=c, s=s,
               marker=m, alpha=0.8)
ax.plot(lims, lims, "--", color="black", lw=0.7)
ax.set_xlabel("coverage (current)"); ax.set_ylabel("top-3-usage coverage")
ax.set_title("cov3 vs cov: dropping the retired label\n(changes little when max is dominant)")

ax = axes[2]
# separation from null: standardised gap between GA and random per metric
txt = []
for key in ("cov", "wcov", "cov3", "wcov3"):
    g = np.array([r[key] for r in records if r["kind"] == "GA"])
    n = np.array([r[key] for r in records if r["kind"] == "random"])
    d = (g.mean() - n.mean()) / np.sqrt(0.5 * (g.var() + n.var()))
    txt.append((key, d, g.mean(), n.mean()))
    ax.bar(key, d, color="#4477aa")
ax.set_title("GA-vs-null separation (Cohen's d)\nper candidate metric")
ax.set_ylabel("d")
for key, d, gm, nm in txt:
    print(f"{key:6s} GA mean {gm:.3f}  null mean {nm:.3f}  Cohen's d {d:.2f}")

fig.suptitle("Usage-weighted coverage candidates (four_rooms 7x7 cohort, exploration)")
fig.tight_layout()
fig.savefig(Path(__file__).parent / "figs" / "F-weighted-coverage-explore.png"
            if "__file__" in dir() else "figs/F-weighted-coverage-explore.png",
            dpi=150, bbox_inches="tight")
plt.show()
