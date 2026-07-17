# %% [markdown]
# # cross-world 01 — Visualising the null-standardisation results
#
# Reads the frozen artefacts of cross-world/00 and draws:
#   figs/null-distributions-cartesian.png : per-world F-bar distributions
#     (shuffle vs perm_balanced) with the Cartesian marked; the z vector
#     made visible.
#   figs/cartesian-z-and-coupling.png     : (a) the Cartesian z lollipop
#     in palette order; (b) the null cross-world correlation of F-bar
#     over the shared draws (how coupled the worlds are before selection).

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
                 "/01-standardisation-plots.py")
DATA, FIGS = NB_DIR / "data", NB_DIR / "figs"
FIGS.mkdir(exist_ok=True)

ENVS = ["wrap_grid", "open_grid", "helical", "pinwheel", "four_rooms", "pillar_3"]
df = pd.read_parquet(DATA / "null_standardisation_full2016.parquet")
con = json.loads((DATA / "constants_full2016.json").read_text())

# %% Figure 1: distributions + Cartesian per world
fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.6), dpi=150)
for ax, e in zip(axes.ravel(), ENVS):
    sh = df[(df.env_id == e) & (df.ensemble == "shuffle")].mean_free
    pb = df[(df.env_id == e) & (df.ensemble == "perm_balanced")].mean_free
    lo = min(sh.min(), pb.min(), con[e]["cartesian_mean_free"]) - 0.15
    hi = max(sh.max(), pb.max()) + 0.15
    bins = np.linspace(lo, hi, 70)
    ax.hist(sh, bins=bins, density=True, alpha=0.55, color="tab:blue",
            label="shuffle null (2,016)")
    ax.hist(pb, bins=bins, density=True, alpha=0.45, color="tab:green",
            label="perm-balanced (2,016)")
    ax.axvline(con[e]["mu"], color="tab:blue", ls="--", lw=1.2,
               label="frozen centre $\\mu_e$")
    ax.axvline(con[e]["cartesian_mean_free"], color="crimson", lw=2.0,
               label="Cartesian")
    ax.annotate(f"$z(\\mathrm{{Cart}}) = {con[e]['cartesian_z']:+.1f}$",
                xy=(0.03, 0.92), xycoords="axes fraction", fontsize=10,
                color="crimson")
    ax.set_title(e, fontsize=11)
    ax.set_xlabel("per-pair mean free energy  $\\bar{F}_e$")
    if e == ENVS[0]:
        ax.legend(fontsize=7, loc="upper right")
fig.suptitle("Null ensembles and the Cartesian, per world "
             "(same 2,016 lattice twists projected everywhere)", fontsize=12)
fig.tight_layout()
fig.savefig(FIGS / "null-distributions-cartesian.png", bbox_inches="tight")
print("saved figure 1")

# %% Figure 2: z lollipop + null cross-world coupling
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.6, 5.2), dpi=150,
                             gridspec_kw={"width_ratios": [1.1, 1]})
zs = [con[e]["cartesian_z"] for e in ENVS]
xs = np.arange(len(ENVS))
a1.axhspan(-3, 3, color="grey", alpha=0.18, label="typical null range ($|z|\\leq 3$)")
a1.axhline(0, color="grey", lw=0.8)
a1.vlines(xs, 0, zs, color="crimson", lw=2)
a1.scatter(xs, zs, s=60, color="crimson", zorder=5)
for x, z in zip(xs, zs):
    a1.annotate(f"{z:+.1f}", (x, z), textcoords="offset points",
                xytext=(6, -4 if z < 0 else 4), fontsize=9)
mean_z, max_z = float(np.mean(zs)), float(np.max(zs))
a1.axhline(mean_z, color="black", ls=":", lw=1.2,
           label=f"mean $z$ = {mean_z:+.1f}")
a1.set_xticks(xs)
a1.set_xticklabels(ENVS, rotation=20, ha="right", fontsize=9)
a1.set_ylabel("Cartesian $z$ (negative = cheaper than typical null)")
a1.set_title("The Cartesian z vector: a specialist for boundedness",
             fontsize=11)
a1.legend(fontsize=8, loc="lower left")

piv = df[df.ensemble == "shuffle"].pivot(index="draw_id", columns="env_id",
                                         values="mean_free")[ENVS]
corr = piv.corr(method="spearman")
im = a2.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
a2.set_xticks(range(6)); a2.set_yticks(range(6))
a2.set_xticklabels(ENVS, rotation=35, ha="right", fontsize=8)
a2.set_yticklabels(ENVS, fontsize=8)
for i in range(6):
    for j in range(6):
        a2.annotate(f"{corr.iloc[i, j]:+.2f}", (j, i), ha="center",
                    va="center", fontsize=8,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
fig.colorbar(im, ax=a2, shrink=0.8, label="Spearman correlation")
a2.set_title("Null cross-world coupling of $\\bar{F}$ (same twists)",
             fontsize=11)
fig.tight_layout()
fig.savefig(FIGS / "cartesian-z-and-coupling.png", bbox_inches="tight")
print("saved figure 2")
print("coupling matrix (shuffle):")
print(corr.round(2).to_string())
