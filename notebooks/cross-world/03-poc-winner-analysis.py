# %% [markdown]
# # cross-world 03 — What the mean_z winner kept of the compass
#
# Analysis of the first cross-world run (grid-twist/cross-world-poc,
# g500 mean_z, perm_balanced init).  Two questions:
# 1. Per-world-restricted alignment across generations: did the winner
#    keep compass structure on the rows the bounded worlds use, while
#    rebuilding the torus structure?  (Alignment = best-of-24 ordering
#    match restricted to a world's walkable cells.)
# 2. Achievable-range (AR) re-score: position of winner and Cartesian
#    in [0,1] per world, 0 = provisional single-world floor, 1 = null
#    median.  Also writes data/provisional_floors.json for the
#    proportional-fairness objective.

# %%
import json
from itertools import permutations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gridbench.functional_graph.probe_env import build_goal_free_probe_env


def _nb_dir(default: str) -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        cwd = Path.cwd().resolve()
        return cwd if (cwd / Path(default).name).exists() \
            or cwd.name == Path(default).parent.name else Path(default).parent


NB_DIR = _nb_dir("/media/merlin/phd-marlyn/gridBench/notebooks/cross-world"
                 "/03-poc-winner-analysis.py")
DATA = NB_DIR / "data"
DATA.mkdir(exist_ok=True)
FIGS = NB_DIR / "figs"
POC = Path("/media/merlin/grid-twist/cross-world-poc/cross-world")
SUMMARY = POC / "cross-world-g500-meanz.summary.json"
CON = json.loads((NB_DIR / "results/seed20260718/constants_full2016.json")
                 .read_text())
ENVS = ["wrap_grid", "open_grid", "helical", "pinwheel", "four_rooms", "pillar_3"]
ORDS = np.array(list(permutations(range(4))))

# provisional floors: best single-world FE run-best per env (store + outputs)
import glob as _glob
FLOORS = {}
OUTS = ["core-silence-hunt-10-07", "core-silence-hunt2-shuffle-10-07",
        "core-silence-scale-env-warm-11-07", "core-torus-shuffle-15-07",
        "g400-pop-96-perm-bal-13-05-b1-free-existing-envs"]
for e in ENVS:
    best = np.inf
    for p in _glob.glob(f"/media/merlin/grid-twist/data-schema-11/multi/*/"
                        f"fitness_objective=free_energy/env_id={e}/shape=7x7/"
                        f"*/*/run_name=*/*-multi-all.summary.json"):
        f = json.load(open(p)).get("best_expected_free")
        if f:
            best = min(best, f)
    tag = "-" + e.replace("_", "-") + "-7x7-"
    for b in OUTS:
        for p in _glob.glob(f"/media/merlin/grid-twist/gridtwist-outputs/{b}/"
                            f"*{tag}*/*-multi-all.summary.json"):
            if "-b1-free-" not in p:
                continue
            f = json.load(open(p)).get("best_expected_free")
            if f:
                best = min(best, f)
    FLOORS[e] = best
(DATA / "provisional_floors.json").write_text(json.dumps(
    {e: {"floor_F": FLOORS[e], "mu": CON[e]["mu"], "s": CON[e]["s"],
         "source": "best single-world FE run-best, store+outputs 2026-07-18"}
     for e in ENVS}, indent=1))
print("floors:", {e: round(FLOORS[e], 3) for e in ENVS})

# %% Per-world walkable subsets
subsets = {}
for e in ENVS:
    env = build_goal_free_probe_env(e, (7, 7), 0.97)
    wf = getattr(env, "walls_flat", None)
    walls = set(int(w) for w in np.ravel(wf)) if wf is not None else set()
    subsets[e] = [s for s in range(49) if s not in walls]


def alignment(sigma, cells):
    rows = sigma[cells]
    return float((rows[None] == ORDS[:, None, :]).mean(axis=(1, 2)).max())


summ = json.load(open(SUMMARY))
hist = summ["history"]
gens = [h["gen"] for h in hist]
al = {e: [] for e in ENVS}
prev_hash, prev_vals = None, None
for h in hist:
    if h.get("best_sigma_hash") == prev_hash and prev_vals is not None:
        for e in ENVS:
            al[e].append(prev_vals[e])
        continue
    sigma = np.asarray(h["best_sigma"], dtype=int)
    prev_vals = {e: alignment(sigma, subsets[e]) for e in ENVS}
    prev_hash = h.get("best_sigma_hash")
    for e in ENVS:
        al[e].append(prev_vals[e])

# %% AR re-score of winner + Cartesian
zw = {e: hist[-1]["best_z_by_env"][e] for e in ENVS}
ar = {}
for e in ENVS:
    mu, s = CON[e]["mu"], CON[e]["s"]
    zf = (FLOORS[e] - mu) / s
    ar[e] = {"winner": (zw[e] - zf) / (0 - zf),
             "cartesian": (CON[e]["cartesian_z"] - zf) / (0 - zf),
             "z_floor": zf}
    print(f"{e:12} AR(winner)={ar[e]['winner']:.2f}  "
          f"AR(Cart)={ar[e]['cartesian']:.2f}  z_floor={zf:+.1f}")

# %% Figure
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.2, 5.2), dpi=150,
                             gridspec_kw={"width_ratios": [1.35, 1]})
COLS = dict(zip(ENVS, ["tab:blue", "tab:orange", "tab:green", "tab:red",
                       "tab:purple", "tab:brown"]))
for e in ENVS:
    ls = "--" if e in ("wrap_grid", "helical") else "-"
    a1.plot(gens, al[e], color=COLS[e], ls=ls, lw=1.6, label=e)
a1.axhline(0.32, color="grey", lw=0.8, ls=":", label="chance level")
a1.set_xlabel("generation")
a1.set_ylabel("alignment restricted to the world's walkable cells")
a1.set_title("The winner keeps the compass where boundaries need it",
             fontsize=11)
a1.set_ylim(0.25, 1.02)
a1.legend(fontsize=8)

xs = np.arange(len(ENVS))
a2.axhline(1.0, color="grey", lw=0.8, ls="--")
a2.axhline(0.0, color="black", lw=0.8)
a2.scatter(xs - 0.12, [ar[e]["cartesian"] for e in ENVS], marker="o", s=70,
           color="crimson", label="Cartesian")
a2.scatter(xs + 0.12, [ar[e]["winner"] for e in ENVS], marker="*", s=140,
           color="tab:blue", label="mean_z winner")
a2.set_xticks(xs)
a2.set_xticklabels(ENVS, rotation=25, ha="right", fontsize=8)
a2.set_ylabel("achievable-range position\n(0 = single-world floor, 1 = null median)")
a2.set_title("AR re-score: remaining headroom per world", fontsize=11)
a2.legend(fontsize=8)
fig.tight_layout()
out = FIGS / "poc-winner-alignment-and-AR.png"
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}")
print("final alignments:", {e: round(al[e][-1], 2) for e in ENVS})
