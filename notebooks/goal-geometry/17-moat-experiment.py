# %% [markdown]
# # 17 — The moat experiment: single-row mutant scans around the founders
#
# Mechanism test (2026-07-15): the ratchet-vs-moat account predicts a
# local fitness-landscape ANISOTROPY around the two generation-zero
# founders of the matched s173 runs:
#   - around the COMPASS founder (perm_balanced), mutants that reduce the
#     weakest label's policy usage are systematically deleterious -- the
#     moat that walls off the silenced optimum;
#   - around the SHUFFLE founder, a beneficial mutant set exists and is
#     enriched in dominant-basin-extending edits -- the ratchet's first
#     stair.
#
# Anchors: history[0].best_sigma of the two real runs (not resampled).
# Mutants: for each of the 40 walkable states, 6 random alternative row
# permutations -> 240 per anchor.  Per mutant: full-goal mean free energy
# (wall-masked), greedy usage shares, per-label basin coverage.

# %%
import json
import glob
import random
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
from gridcore.bridge import EvalConfig, _state_dist_class, build_twisted_env_from_sigma  # noqa: E402
from gridcore.info import DecisionInformation as GC_DI  # noqa: E402
from gridbench.functional_graph.decomposition import decompose, deterministic_successor  # noqa: E402
from gridbench.functional_graph.probe_env import build_goal_free_probe_env  # noqa: E402
from itertools import permutations  # noqa: E402

BASE = Path("/media/merlin/grid-twist/gridtwist-outputs")
WALLS = {4, 11, 25, 28, 29, 31, 32, 33, 38}
WALK = [s for s in range(49) if s not in WALLS]
PERMS = [list(p) for p in permutations(range(4))]
FIG_DIR = Path(__file__).resolve().parent / "figs"
N_MUT_PER_STATE = 6

probe = build_goal_free_probe_env("four_rooms", (7, 7), 0.97)
BASE_SUCC = np.stack([deterministic_successor(probe, a) for a in range(4)], axis=0)
IDX = np.arange(49)


def evaluate(sigma):
    """(mean_free wall-masked, usage shares, per-label coverage)."""
    usage = np.zeros(4)
    tot = 0.0
    for g in WALK:
        cfg = EvalConfig(env_id="four_rooms", shape=(7, 7), goal=int(g), beta=1.0,
                         determinism=0.97, manhattan=True, theta=1e-5, state_dist="uniform")
        e = build_twisted_env_from_sigma(sigma, cfg)
        di = GC_DI(e, _state_dist_class("uniform")(e), 1e-5,
                   max_iterations=200_000, max_info_iterations=10_000)
        pi, _, F = di.get_opt_policy_Z_free_vector(1.0)
        F = np.asarray(F, dtype=float)
        tot += float(F[WALK].mean())
        greedy = np.argmax(np.asarray(pi, dtype=float), axis=1)
        for s in WALK:
            if s != g:
                usage[greedy[s]] += 1
    usage /= usage.sum()
    sigma_inv = np.argsort(sigma, axis=1)
    cov = np.empty(4)
    for l in range(4):
        succ = BASE_SUCC[sigma_inv[:, l], IDX]
        bs = np.asarray(decompose(succ, walls=WALLS).basin_sizes, dtype=int)
        cov[l] = bs.max() / len(WALK) if bs.size else 0.0
    return tot / len(WALK), usage, cov


def founder(pattern):
    sj = glob.glob(str(BASE / pattern))[0]
    return np.asarray(json.load(open(sj))["history"][0]["best_sigma"], dtype=int)


import os

def run_best(pattern):
    import numpy as np, glob as g
    return np.load(g.glob(str(BASE / pattern))[0]).astype(int)

if os.environ.get("MOAT_ANCHORS") == "converged":
    # The moat test proper: the two g500 run-bests (local optima).
    ANCHORS = {
        "balanced trap floor (g500 run-best)":
            run_best("core-silence-hunt-10-07/*s173*/*-multi-all.sigma.npy"),
        "silenced peak (g500 run-best)":
            run_best("core-silence-hunt2-shuffle-10-07/*s173*/*-multi-all.sigma.npy"),
    }
    OUT_TAG = "converged"
else:
    ANCHORS = {
        "compass (perm_balanced founder)":
            founder("core-silence-hunt-10-07/*s173*/*-multi-all.summary.json"),
        "shuffle founder":
            founder("core-silence-hunt2-shuffle-10-07/*s173*/*-multi-all.summary.json"),
    }
    OUT_TAG = "founders"

results = {}
rng = random.Random(20260715)
for name, anchor in ANCHORS.items():
    F0, u0, c0 = evaluate(anchor)
    print(f"== {name}: F={F0:.4f} usage={np.round(u0,2)} cov={np.round(c0,2)}", flush=True)
    rows = []
    for s in WALK:
        cur = list(anchor[s])
        alts = [p for p in PERMS if p != cur]
        for p in rng.sample(alts, N_MUT_PER_STATE):
            m = anchor.copy()
            m[s] = p
            F, u, c = evaluate(m)
            rows.append({"state": s, "dF": F - F0,
                         "d_min_usage": float(u.min() - u0.min()),
                         "d_max_cov": float(c.max() - c0.max())})
        print(f"  state {s} done ({len(rows)} mutants)", flush=True)
    results[name] = {"F0": F0, "usage0": u0.tolist(), "cov0": c0.tolist(), "mutants": rows}
    (FIG_DIR / f"moat_results_{OUT_TAG}.json").write_text(json.dumps(results, indent=1))

# %% Figure
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), dpi=150, sharey=True)
for ax, (name, res) in zip(axes, results.items()):
    d = res["mutants"]
    x = [m["d_min_usage"] for m in d]
    y = [m["dF"] for m in d]
    ben = sum(1 for m in d if m["dF"] < 0)
    ben_ext = sum(1 for m in d if m["dF"] < 0 and m["d_max_cov"] > 0)
    del_red = sum(1 for m in d if m["d_min_usage"] < -0.01)
    del_red_bad = sum(1 for m in d if m["d_min_usage"] < -0.01 and m["dF"] > 0)
    ax.scatter(x, y, s=18, alpha=0.6,
               c=[m["d_max_cov"] for m in d], cmap="coolwarm", vmin=-0.15, vmax=0.15)
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8, ls=":")
    ax.set_title(f"{name}\nbeneficial {ben}/{len(d)} (basin-ext.\ {ben_ext})\n"
                 f"usage-reducing deleterious {del_red_bad}/{del_red}", fontsize=9)
    ax.set_xlabel("Δ min-label usage")
axes[0].set_ylabel("Δ full-goal mean free energy  (mutant − founder)")
sc = axes[1].collections[0]
fig.colorbar(sc, ax=axes, label="Δ dominant coverage", shrink=0.8)
out = FIG_DIR / f"F-moat-experiment-{OUT_TAG}.png"
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}")
