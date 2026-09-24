# %% [markdown]
# # cross-world 05 — Exploratory four_rooms--wrap_grid Pareto front
#
# Chapter figure from the archived pre-fix NSGA-II run.  The points remain
# direct evaluations, but the old driver applied row mutation to every
# offspring instead of using the configured outer mutation probability of
# 0.3.  Treat the front as exploratory until the matched run is regenerated.

# %%
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _nb_dir(default: str) -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        cwd = Path.cwd().resolve()
        return cwd if (cwd / Path(default).name).exists() \
            or cwd.name == Path(default).parent.name else Path(default).parent


NB_DIR = _nb_dir("/media/merlin/phd-marlyn/gridBench/notebooks/cross-world"
                 "/05-front-figure.py")
FIGS = NB_DIR / "figs"
POC = "/media/merlin/grid-twist/cross-world-poc/cross-world"

df = pd.read_parquet(f"{POC}/cross-world-nsga2-A-fr-wrap.front.parquet")
df = df.sort_values("z_four_rooms")

# anchors: (z_fr, z_wrap, label, marker)
CON = json.loads(Path("/media/merlin/phd-marlyn/gridBench/notebooks/cross-world"
                      "/results/seed20260718/constants_full2016.json").read_text())
anchors = [(-11.04, 4.48, "Cartesian", "o", "cartesian")]
for tag, lab in [("cross-world-g200-nash-warmstart-fr", "warm-start final"),
                 ("cross-world-g200-nash-2world-fr-wrap", "2-world winner"),
                 ("cross-world-g500-nash-permbal", "nash g500 balanced"),
                 ("cross-world-g500-nash-shuffle", "nash g500 shuffle")]:
    s = json.load(open(f"{POC}/{tag}.summary.json"))
    z = s["history"][-1]["best_z_by_env"]
    anchors.append((z["four_rooms"], z["wrap_grid"], lab, "D", "scalar"))

LABEL_OFFSETS = {
    "Cartesian": (8, 5),
    "warm-start final": (-72, 8),
    "2-world winner": (-8, -18),
    "nash g500 balanced": (10, 17),
    "nash g500 shuffle": (10, -22),
}

fig, ax = plt.subplots(figsize=(8.8, 7.0), dpi=150)
# the empty region: fr <= -18 AND wrap <= -10
ax.add_patch(plt.Rectangle((-24, -24), 6, 14, facecolor="#fee0d2",
                           edgecolor="none", zorder=0))
ax.annotate("not reached in this run:\nfour_rooms $\\leq -18$\nwith wrap $\\leq -10$",
            xy=(-21.2, -16.5), fontsize=9, color="#a50f15", ha="center")
sc = ax.scatter(df.z_four_rooms, df.z_wrap_grid, c=df.alignment,
                cmap="viridis", vmin=0.30, vmax=1.0, s=55,
                edgecolors="k", linewidths=0.4, zorder=3,
                label=f"NSGA-II front (n={len(df)})")
ax.plot(df.z_four_rooms, df.z_wrap_grid, color="grey", lw=0.8, zorder=2)
front_points = df[["z_four_rooms", "z_wrap_grid"]].to_numpy()
for zf, zw, lab, m, kind in anchors:
    is_front = bool(np.any(np.all(np.isclose(front_points, [zf, zw], atol=1e-8), axis=1)))
    edge = "crimson" if kind == "cartesian" or is_front else "#d95f02"
    face = "crimson" if is_front else "none"
    ax.scatter([zf], [zw], marker=m, s=110, facecolors=face,
               edgecolors=edge, linewidths=1.8, zorder=4)
    ax.annotate(lab, (zf, zw), textcoords="offset points",
                xytext=LABEL_OFFSETS[lab],
                fontsize=8, color=edge)
ax.axhline(0, color="grey", lw=0.6, ls=":")
ax.axvline(0, color="grey", lw=0.6, ls=":")
ax.set_xlabel(r"four_rooms $z$  (lower = better)")
ax.set_ylabel(r"wrap_grid $z$  (lower = better)")
ax.set_title("Exploratory four_rooms--wrap_grid front\n"
             "(archived pre-fix NSGA-II run; front coloured by alignment)",
             fontsize=11)
cb = fig.colorbar(sc, ax=ax, label="alignment (1 = Cartesian, 0.32 = chance)")
cb.ax.axhline(0.32, color="crimson", lw=1.2)
cb.ax.annotate("chance", xy=(1.05, 0.32), xycoords=("axes fraction", "data"),
               fontsize=7, color="crimson", va="center")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
out = FIGS / "fr-wrap-pareto-front.png"
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}")
print(f"front alignment range: {df.alignment.min():.2f}-{df.alignment.max():.2f}")
# slope of the exchange in the compromise region
mid = df[(df.z_four_rooms > -18) & (df.z_four_rooms < -8)]
if len(mid) > 3:
    m, b = np.polyfit(mid.z_four_rooms, mid.z_wrap_grid, 1)
    print(f"compromise-region fitted slope: {m:.2f} z_wrap per z_fr (n={len(mid)})")
