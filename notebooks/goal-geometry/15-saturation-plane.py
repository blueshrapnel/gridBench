# %% [markdown]
# # 15 — The saturation plane: max vs min label coverage
#
# Question (2026-07-15, Karen): fp_min_label_coverage alone guarantees a
# floor ("all four labels are at least this") but a mid-range min is
# ambiguous about the top end and carries no run performance.  Pairing
# min with the max already used as the coverage axis completes the
# decomposition -- [min, max] bracket all four labels -- and colouring by
# mean free energy keeps the performance link:
#   top-right corner  = all four labels saturated (the switchboard);
#   diagonal          = symmetric twists (Cartesian tori sit low-left);
#   bottom-right      = one/two saturated + one silenced (retired regime);
#   low-left off-axis = fragmented walled twists.
#
# Cohorts: run-bests only (population clouds need the fp v8 per-label
# columns).  four_rooms 7x7 perm-bal + shuffle, the 11-07 shuffle env
# fan, existing-envs g400, wave1 wrap 13x13.

# %%
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gridbench.functional_graph.decomposition import decompose, deterministic_successor
from gridbench.functional_graph.probe_env import build_goal_free_probe_env

BASE = Path("/media/merlin/grid-twist/gridtwist-outputs")
def _nb_dir(default: str) -> Path:
    """Directory of this notebook: __file__ when run as a script, the
    jupytext/Jupyter-safe fallback otherwise (kernel cwd if it matches,
    else the canonical repo path)."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        cwd = Path.cwd().resolve()
        return cwd if (cwd / Path(default).name).exists() or cwd.name == Path(default).parent.name else Path(default).parent


FIG_DIR = _nb_dir("/media/merlin/phd-marlyn/gridBench/notebooks/goal-geometry/15-saturation-plane.py") / "figs"

RUNS = []  # (glob, marker, label)
GROUPS = [
    ("core-silence-hunt-10-07/*", "o", "four_rooms 7x7 perm-bal"),
    ("core-silence-hunt2-shuffle-10-07/*", "*", "four_rooms 7x7 shuffle"),
    ("core-silence-scale-env-warm-11-07/g500*", "s", "env fan / 9x9 shuffle"),
    ("g400-pop-96-perm-bal-13-05-b1-free-existing-envs/*", "^", "palette g400 perm-bal"),
    ("g500-pop-96-perm-bal-04-07-b1-free-K1-wave1-fans/*", "D", "wrap 13x13 perm-bal"),
]

_env_cache = {}


def cov4(env_id, shape, sigma):
    key = (env_id, tuple(shape))
    if key not in _env_cache:
        env = build_goal_free_probe_env(env_id, tuple(shape), 0.97)
        nS = shape[0] * shape[1]
        base = np.stack([deterministic_successor(env, a) for a in range(4)], axis=0)
        wf = getattr(env, "walls_flat", None)
        walls = set(int(w) for w in np.ravel(wf)) if wf is not None and len(np.ravel(wf)) else set()
        _env_cache[key] = (base, walls, nS - len(walls))
    base, walls, n_walk = _env_cache[key]
    sigma_inv = np.argsort(sigma, axis=1)
    idx = np.arange(sigma.shape[0])
    out = np.empty(4)
    for l in range(4):
        succ = base[sigma_inv[:, l], idx]
        bs = np.asarray(decompose(succ, walls=walls).basin_sizes, dtype=int)
        out[l] = bs.max() / n_walk if bs.size else 0.0
    return out


pts = []
for pat, marker, glabel in GROUPS:
    for run in sorted(glob.glob(str(BASE / pat))):
        sigs = glob.glob(run + "/*-multi-all.sigma.npy")
        sums = glob.glob(run + "/*-multi-all.summary.json")
        if not sigs or not sums:
            continue
        s = json.load(open(sums[0]))
        cfg = s["config"]
        sigma = np.load(sigs[0])
        if sigma.shape[0] != cfg["shape"][0] * cfg["shape"][1]:
            continue
        c = cov4(cfg["env_id"], cfg["shape"], sigma)
        pts.append({"max": c.max(), "min": c.min(), "free": s["best_expected_free"],
                    "marker": marker, "group": glabel, "env": cfg["env_id"]})

fig, ax = plt.subplots(figsize=(8.6, 7.2), dpi=150)
frees = [p["free"] for p in pts]
vmin, vmax = min(frees), max(frees)
for pat, marker, glabel in GROUPS:
    gp = [p for p in pts if p["group"] == glabel]
    sc = ax.scatter([p["max"] for p in gp], [p["min"] for p in gp],
                    c=[p["free"] for p in gp], cmap="viridis", vmin=vmin, vmax=vmax,
                    marker=marker, s=90, edgecolors="k", linewidths=0.4, label=glabel)
ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--")
ax.set_xlabel("max label coverage  (dominant basin)")
ax.set_ylabel("min label coverage  (weakest label)")
ax.set_xlim(0, 1.03); ax.set_ylim(0, 1.03)
ax.annotate("switchboard:\nall four saturated", (0.97, 0.97), ha="right", va="top", fontsize=9)
ax.annotate("retired regime", (0.97, 0.05), ha="right", fontsize=9)
ax.legend(fontsize=8, loc="upper left")
fig.colorbar(sc, ax=ax, label="best mean free energy")
ax.set_title("Saturation plane: run-bests, all cohorts")
out = FIG_DIR / "F-saturation-plane.png"
fig.savefig(out, bbox_inches="tight")
print(f"{len(pts)} run-bests -> {out}")
