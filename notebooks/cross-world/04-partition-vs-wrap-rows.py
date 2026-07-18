# %% [markdown]
# # cross-world 04 — The contested rows: partition against wrap
#
# The wall is certified (warm-start walks back; two-world reproduces the
# pin; five regimes bind at four_rooms z ~ -11.5).  This notebook names
# the territory: per four_rooms-walkable cell, compare the row demanded
# by the four_rooms specialist, the row demanded by the wrap specialist,
# and the row the cross-world winners actually hold.  Output: a conflict
# map (agree / disagree cells; who the winner sides with) and counts.

# %%
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _nb_dir(default: str) -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        cwd = Path.cwd().resolve()
        return cwd if (cwd / Path(default).name).exists() \
            or cwd.name == Path(default).parent.name else Path(default).parent


NB_DIR = _nb_dir("/media/merlin/phd-marlyn/gridBench/notebooks/cross-world"
                 "/04-partition-vs-wrap-rows.py")
FIGS = NB_DIR / "figs"
OUT = "/media/merlin/grid-twist/gridtwist-outputs"
POC = "/media/merlin/grid-twist/cross-world-poc/cross-world"
FR_WALLS = {4, 11, 25, 28, 29, 31, 32, 33, 38}
WALK = [s for s in range(49) if s not in FR_WALLS]


def best_sigma(base_glob):
    best, sig = np.inf, None
    for p in glob.glob(base_glob):
        f = json.load(open(p)).get("best_expected_free", np.inf)
        if f < best:
            best = f
            sig = np.load(p.replace(".summary.json", ".sigma.npy")).astype(int)
    return sig, best


fr_spec, fr_F = best_sigma(f"{OUT}/core-silence-hunt2-shuffle-10-07/*/"
                           "*-multi-all.summary.json")
wrap_spec, wrap_F = best_sigma(f"{OUT}/core-torus-shuffle-15-07/"
                               "*wrap-grid*/*-multi-all.summary.json")
print(f"fr specialist F={fr_F:.3f}; wrap specialist F={wrap_F:.3f}")

winners = {}
for tag in ["cross-world-g200-nash-2world-fr-wrap",
            "cross-world-g200-nash-warmstart-fr",
            "cross-world-g500-nash-permbal"]:
    winners[tag.split("-")[-1]] = np.load(f"{POC}/{tag}.sigma.npy").astype(int)

# %% classify each four_rooms-walkable cell
def same(a, b):
    return np.array_equal(a, b)


cats = {}
for s in WALK:
    agree = same(fr_spec[s], wrap_spec[s])
    cats[s] = {"specialists_agree": agree}
    for name, w in winners.items():
        if same(w[s], fr_spec[s]) and same(w[s], wrap_spec[s]):
            side = "both"
        elif same(w[s], fr_spec[s]):
            side = "fr"
        elif same(w[s], wrap_spec[s]):
            side = "wrap"
        else:
            side = "neither"
        cats[s][name] = side

n_agree = sum(1 for s in WALK if cats[s]["specialists_agree"])
print(f"specialists agree on {n_agree}/40 walkable rows; "
      f"disagree on {40 - n_agree} (the contested set)")
for name in winners:
    from collections import Counter
    c = Counter(cats[s][name] for s in WALK)
    cc = Counter(cats[s][name] for s in WALK
                 if not cats[s]["specialists_agree"])
    print(f"{name:12} all rows: {dict(c)};  contested rows only: {dict(cc)}")

# %% conflict map figure (2-world winner as the exhibit)
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), dpi=150)
titles = ["specialists: agree vs contested",
          "2-world winner: row allegiance",
          "warm-start final: row allegiance"]
COL = {"agree": "#c7e9c0", "contested": "#fc9272", "both": "#c7e9c0",
       "fr": "#9e9ac8", "wrap": "#6baed6", "neither": "#fdd49e",
       "wall": "#252525"}
for ax, mode, ttl in zip(axes, ["spec", "fr-wrap", "warmstart"], titles):
    for s in range(49):
        r, c = divmod(s, 7)
        if s in FR_WALLS:
            col = COL["wall"]
        elif mode == "spec":
            col = COL["agree"] if cats[s]["specialists_agree"] else COL["contested"]
        else:
            key = "fr-wrap" if mode == "fr-wrap" else "warmstart-fr"
            key = [k for k in winners if (mode == "fr-wrap") == ("wrap" in k)][0]
            col = COL[cats[s][key]]
        ax.add_patch(plt.Rectangle((c, 6 - r), 1, 1, facecolor=col,
                                   edgecolor="white", lw=1.2))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(ttl, fontsize=10)
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COL[k]) for k in
           ["agree", "contested", "fr", "wrap", "neither"]]
fig.legend(handles, ["specialists agree", "contested",
                     "winner sides with four_rooms", "winner sides with wrap",
                     "winner sides with neither"],
           loc="lower center", ncol=5, fontsize=8, frameon=False)
fig.suptitle("Partition against wrap: the contested rows of the shared lattice",
             fontsize=12)
fig.tight_layout(rect=[0, 0.06, 1, 1])
out = FIGS / "partition-vs-wrap-contested-rows.png"
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}")
