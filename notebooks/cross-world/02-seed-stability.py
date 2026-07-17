# %% [markdown]
# # cross-world 02 — Seed stability of the standardisation constants
#
# IQR view across the five independent draw seeds: per world, boxes
# (median + IQR) with the five seed values overlaid, for the Cartesian
# z, the centre mu (as deviation from the cross-seed median, in bits),
# and the spread s.  Output: results/seed-stability-iqr.png

# %%
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
                 "/02-seed-stability.py")
RES = NB_DIR / "results"
ENVS = ["wrap_grid", "open_grid", "helical", "pinwheel", "four_rooms", "pillar_3"]

seeds = sorted(d.name for d in RES.iterdir() if d.name.startswith("seed"))
cons = {s: json.loads((RES / s / "constants_full2016.json").read_text())
        for s in seeds}
print(f"{len(seeds)} seeds: {seeds}")

vals = {stat: {e: [cons[s][e][key] for s in seeds] for e in ENVS}
        for stat, key in [("z", "cartesian_z"), ("mu", "mu"), ("s", "s")]}

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.6, 5.0), dpi=150)
xs = np.arange(len(ENVS))
rng = np.random.default_rng(3)


def box_with_points(ax, data_per_env, ylabel, title):
    ax.boxplot([data_per_env[e] for e in ENVS], positions=xs, widths=0.5,
               medianprops=dict(color="crimson", lw=1.6),
               boxprops=dict(color="black"), whiskerprops=dict(color="black"),
               capprops=dict(color="black"), showfliers=False)
    for i, e in enumerate(ENVS):
        ax.scatter(np.full(len(seeds), i) + rng.uniform(-0.10, 0.10, len(seeds)),
                   data_per_env[e], s=26, alpha=0.8, color="tab:blue", zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels(ENVS, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)


box_with_points(a1, vals["z"],
                "Cartesian $z$",
                "Cartesian z across seeds (boxes: median + IQR)")
a1.axhspan(-3, 3, color="grey", alpha=0.15)
a1.axhline(0, color="grey", lw=0.8)

mu_dev = {e: np.array(vals["mu"][e]) - np.median(vals["mu"][e]) for e in ENVS}
box_with_points(a2, mu_dev,
                r"$\mu_e$ deviation from cross-seed median (bits)",
                "Centre stability (max spread 0.013 bits)")
a2.axhline(0, color="grey", lw=0.8)

box_with_points(a3, vals["s"],
                r"spread $s_e$ (bits)",
                "Spread stability across seeds")

fig.suptitle("Standardisation constants across five independent 2,016-draw "
             "ensembles", fontsize=12)
fig.tight_layout()
out = RES / "seed-stability-iqr.png"
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}")
for e in ENVS:
    z = vals["z"][e]
    print(f"{e:12} z IQR = [{np.percentile(z, 25):+.2f}, {np.percentile(z, 75):+.2f}]"
          f"  full range {min(z):+.2f}..{max(z):+.2f}")
