# %% [markdown]
# # cross-world 05 — The wall's final form: the fr-wrap Pareto front
#
# Chapter figure: round-4 NSGA-II front in the (z_four_rooms, z_wrap)
# plane, coloured by alignment, anchors labelled, the empty region
# (fr <= -18 and wrap <= -10) marked.  Reads: the trade is a broad
# ~slope-1 exchange; the whole front is non-compass (alignment
# 0.32-0.41); Cartesian sits dominated in the interior.

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
anchors = [(-11.04, 4.48, "Cartesian", "o")]
for tag, lab in [("cross-world-g200-nash-warmstart-fr", "warm-start final"),
                 ("cross-world-g200-nash-2world-fr-wrap", "2-world winner"),
                 ("cross-world-g500-nash-permbal", "nash g500 balanced"),
                 ("cross-world-g500-nash-shuffle", "nash g500 shuffle")]:
    s = json.load(open(f"{POC}/{tag}.summary.json"))
    z = s["history"][-1]["best_z_by_env"]
    anchors.append((z["four_rooms"], z["wrap_grid"], lab, "D"))

fig, ax = plt.subplots(figsize=(8.8, 7.0), dpi=150)
# the empty region: fr <= -18 AND wrap <= -10
ax.add_patch(plt.Rectangle((-24, -24), 6, 14, facecolor="#fee0d2",
                           edgecolor="none", zorder=0))
ax.annotate("empty:\nno twist holds\nfour_rooms $\\leq -18$\nwith wrap $\\leq -10$",
            xy=(-21.2, -16.5), fontsize=9, color="#a50f15", ha="center")
sc = ax.scatter(df.z_four_rooms, df.z_wrap_grid, c=df.alignment,
                cmap="viridis", vmin=0.30, vmax=1.0, s=55,
                edgecolors="k", linewidths=0.4, zorder=3,
                label=f"NSGA-II front (n={len(df)})")
ax.plot(df.z_four_rooms, df.z_wrap_grid, color="grey", lw=0.8, zorder=2)
for zf, zw, lab, m in anchors:
    ax.scatter([zf], [zw], marker=m, s=110, facecolors="none",
               edgecolors="crimson", linewidths=1.8, zorder=4)
    ax.annotate(lab, (zf, zw), textcoords="offset points", xytext=(8, 5),
                fontsize=8, color="crimson")
ax.axhline(0, color="grey", lw=0.6, ls=":")
ax.axvline(0, color="grey", lw=0.6, ls=":")
ax.set_xlabel(r"four_rooms $z$  (lower = better)")
ax.set_ylabel(r"wrap_grid $z$  (lower = better)")
ax.set_title("Partition against wrap: the measured Pareto boundary\n"
             "(front coloured by alignment; the compass is inside it)",
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
    print(f"compromise-region exchange slope: {m:.2f} z_wrap per z_fr (n={len(mid)})")
