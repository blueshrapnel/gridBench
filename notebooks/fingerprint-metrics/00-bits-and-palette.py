# %% [markdown]
# # 00 — Daniel's bits axis, adopted; the palette axis, demonstrated
#
# Meeting prep 2026-07-18 (vault: daniel-bits-axis-meeting-prep-2026-07-18).
# Three claims, one notebook:
#
# 1. ADOPT THE UNITS: Daniel's −log2(F), F = C/(C+T) (22-05 meeting), is a
#    strictly monotone transform of the cycle/basin ratio axis the paper
#    plots — Spearman −1.0 by construction.  Panel 1 shows the same cloud
#    under both labellings: his framing ("free information from the
#    environment") is an interpretation of the existing axis, not a new one.
# 2. THE NEW DIRECTION IS THE PALETTE: union_coverage / pairwise-Jaccard /
#    dead-end candidates from gridFour nb 02/07 (definitions ported
#    verbatim) are weakly coupled to the existing axes and SEPARATE the
#    four_rooms initialisation families that the ratio plane cannot.
# 3. RANKED CANDIDATES: which metric best separates (a) GA from null,
#    (b) the balanced from the silenced family — the "better fingerprint"
#    shortlist.

# %%
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

from gridbench.functional_graph.decomposition import (
    decompose, deterministic_successor, per_label_stats,
)
from gridbench.functional_graph.probe_env import build_goal_free_probe_env

FIG_DIR = Path(__file__).resolve().parent / "figs"
FIG_DIR.mkdir(exist_ok=True)
ENV_ID, SHAPE, DET = "four_rooms", (7, 7), 0.97
OUT = "/media/merlin/grid-twist/gridtwist-outputs"
STORE = "/media/merlin/grid-twist/data-schema-11/multi"
CACHE = ("/media/merlin/grid-twist/data-schema-11/cache/functional_graph/"
         f"env_id={ENV_ID}/shape=7x7/det={DET}/beta=1/fingerprints.parquet")
SUPP_BASES = ["core-silence-hunt-10-07", "core-silence-hunt2-shuffle-10-07",
              "core-silence-scale-env-warm-11-07",
              "g400-pop-96-perm-bal-13-05-b1-free-existing-envs"]
N_NULL = 300

env = build_goal_free_probe_env(ENV_ID, SHAPE, DET)
WALLS = list(getattr(env, "walls_flat"))
N_WALK = int(env.nS) - len(WALLS)
BASE_SUCC = np.stack([deterministic_successor(env, a) for a in range(4)], axis=0)
IDX = np.arange(int(env.nS))


def per_label_successors(sigma):
    sigma_inv = np.argsort(sigma, axis=1)
    return [BASE_SUCC[sigma_inv[:, l], IDX] for l in range(4)]


def metrics(sigma):
    """Existing axes + Daniel bits + nb-02 candidates (ported verbatim) +
    min-label coverage."""
    dead_end = 0
    basin_H, cycle_sets, nbas, cbrs, covs = [], [], [], [], []
    for succ in per_label_successors(np.asarray(sigma, dtype=int)):
        fg = decompose(succ, walls=WALLS)
        st = per_label_stats(fg)
        nbas.append(st["n_basins"])
        cbrs.append(st["cycle_basin_ratio"])
        sizes = np.array(fg.basin_sizes, dtype=float)
        covs.append(sizes.max() / N_WALK if sizes.size else 0.0)
        for cyc, bs in zip(fg.cycles, fg.basin_sizes):
            if len(cyc) == 1 and bs == 1:
                dead_end += 1
        tot = sizes.sum()
        p = sizes[sizes > 0] / tot if tot > 0 else np.array([1.0])
        basin_H.append(float(-(p * np.log2(p)).sum()))
        s = set()
        for c in fg.cycles:
            s.update(c)
        cycle_sets.append(s)
    jac = [len(a & b) / len(a | b) if (a | b) else 0.0
           for i, a in enumerate(cycle_sets) for b in cycle_sets[i + 1:]]
    cbr = float(np.mean(cbrs))
    return {
        "fp_n_basins": float(np.mean(nbas)),
        "fp_cycle_basin_ratio": cbr,
        "daniel_bits": float(-np.log2(cbr)) if cbr > 0 else np.nan,
        "dead_end_fraction": dead_end / (4 * N_WALK),
        "basin_entropy": float(np.mean(basin_H)),
        "mean_pairwise_jaccard": float(np.mean(jac)) if jac else 0.0,
        "union_coverage": len(set().union(*cycle_sets)) / N_WALK,
        "min_label_coverage": float(min(covs)),
    }


# %% Cohorts: FE run-bests by init + fresh row-shuffle null
def runbest_sigmas():
    out, seen = [], set()
    pats = ([f"{STORE}/init_method=*/fitness_objective=free_energy/env_id={ENV_ID}/"
             "shape=7x7/beta=1/det=0.97/run_name=*/*-multi-all.sigma.npy"] +
            [f"{OUT}/{b}/*-four-rooms-7x7-*/*-multi-all.sigma.npy" for b in SUPP_BASES])
    for pat in pats:
        for sg in glob.glob(pat):
            name = Path(sg).parent.name.replace("run_name=", "")
            if name in seen or "gss" in name or re.search(r"k0\d\d", name):
                continue
            if OUT in sg and "-b1-free-" not in name:
                continue
            seen.add(name)
            init = "shuffle" if ("shuffle" in name or "prod-baseline" in name) \
                else "perm_balanced"
            out.append((np.load(sg).astype(int), init))
    return out


rows = []
for sigma, init in runbest_sigmas():
    rows.append({**metrics(sigma), "group": init})
rng = np.random.default_rng(20260718)
for _ in range(N_NULL):
    sigma = np.stack([rng.permutation(4) for _ in range(int(env.nS))])
    rows.append({**metrics(sigma), "group": "null"})
df = pd.DataFrame(rows)
print(df.group.value_counts().to_dict())

# %% Panel 1 — same cloud, two labellings of one axis
sub = pd.read_parquet(CACHE, columns=["fp_n_basins", "fp_cycle_basin_ratio",
                                      "run_type"])
sub = sub[(sub.run_type == "multi") & (sub.fp_cycle_basin_ratio > 0)].sample(
    20000, random_state=1)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.6), dpi=150, sharex=True)
a1.scatter(sub.fp_n_basins, sub.fp_cycle_basin_ratio, s=3, alpha=0.2,
           color="tab:blue", rasterized=True)
a1.set_ylabel("mean cycle/basin ratio  $F$")
a1.set_title("the paper's axis")
bits = -np.log2(sub.fp_cycle_basin_ratio)
a2.scatter(sub.fp_n_basins, bits, s=3, alpha=0.2, color="tab:green",
           rasterized=True)
a2.set_ylabel(r"$-\log_2 F$  (bits the environment gives for free)")
a2.set_title("Daniel's axis: the same axis, re-unitised")
rho, _ = spearmanr(sub.fp_cycle_basin_ratio, bits)
a2.annotate(f"Spearman vs ratio axis: {rho:+.3f}\n(monotone transform)",
            xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
            fontsize=9)
for a in (a1, a2):
    a.set_xlabel("mean basins per label")
fig.tight_layout()
fig.savefig(FIG_DIR / "bits-axis-is-the-ratio-axis.png", bbox_inches="tight")
print("panel 1 saved; Spearman(bits, ratio) =", round(rho, 4))

# %% Redundancy + separation tables
CANDS = ["daniel_bits", "dead_end_fraction", "basin_entropy",
         "mean_pairwise_jaccard", "union_coverage", "min_label_coverage"]
print(f"\n{'candidate':24} {'vs n_basins':>11} {'vs ratio':>9}   "
      f"{'AUC GA|null':>11} {'AUC bal|shuf':>12}")
ga = df[df.group != "null"]
bal, shf = df[df.group == "perm_balanced"], df[df.group == "shuffle"]
for c in CANDS:
    r_nb, _ = spearmanr(df[c], df.fp_n_basins)
    r_cb, _ = spearmanr(df[c], df.fp_cycle_basin_ratio)
    u1 = mannwhitneyu(ga[c], df[df.group == "null"][c]).statistic
    auc_null = u1 / (len(ga) * N_NULL)
    u2 = mannwhitneyu(bal[c], shf[c]).statistic
    auc_fam = u2 / (len(bal) * len(shf))
    print(f"{c:24} {r_nb:+11.2f} {r_cb:+9.2f}   "
          f"{max(auc_null, 1-auc_null):11.2f} {max(auc_fam, 1-auc_fam):12.2f}")

# %% Panel 2 — the palette plane separates what the ratio plane cannot
fig, (b1, b2) = plt.subplots(1, 2, figsize=(12.4, 5.2), dpi=150)
for ax, x, y, xl, yl, ttl in [
    (b1, "fp_n_basins", "fp_cycle_basin_ratio", "mean basins per label",
     "mean cycle/basin ratio", "ratio plane: families coincide"),
    (b2, "union_coverage", "mean_pairwise_jaccard",
     "union coverage of label cycles", "mean pairwise Jaccard of cycle sets",
     "palette plane: families separate"),
]:
    nul = df[df.group == "null"]
    ax.scatter(nul[x], nul[y], s=14, alpha=0.35, color="lightgrey",
               label=f"row-shuffle null (n={N_NULL})")
    ax.scatter(bal[x], bal[y], marker="*", s=170, facecolors="none",
               edgecolors="crimson", linewidths=1.4,
               label=f"balanced run-bests (n={len(bal)})")
    ax.scatter(shf[x], shf[y], marker="s", s=60, facecolors="none",
               edgecolors="black", linewidths=1.4,
               label=f"row-shuffle run-bests (n={len(shf)})")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(ttl, fontsize=11)
    ax.legend(fontsize=8, framealpha=0.9)
fig.tight_layout()
fig.savefig(FIG_DIR / "palette-plane-family-separation.png", bbox_inches="tight")
print("panel 2 saved")
