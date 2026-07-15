# %% [markdown]
# # 14 — Seed strips: four-label coverage across generations
#
# Question (2026-07-15, Karen): support the init-mechanism findings with
# per-seed strips of ALL FOUR labels' coverage over generations, and
# answer (a) how many generations until at least one label saturates in
# four_rooms, (b) how the saturation differs between perm_balanced and
# shuffle cohorts.
#
# Data: summary.json `history` carries `best_sigma` per generation for
# every synced run, so this computes entirely from the desktop mirror.
# Per generation: per-label basin coverage via gridbench decompose
# (paper-figure side, sigma_inv).
#
# Outputs:
#   figs/F-seed-strips-four-label.png  — heat-strip grid: one 4-row
#     band per seed (rows = labels N/E/S/W, x = generation, colour =
#     that label's coverage), perm_balanced left, shuffle right.
#   seed_strips_saturation.csv — per run: first generation where the
#     dominant label reaches 0.70 / 0.85, final per-label coverages,
#     min-label coverage (the "how many labels saturate" axis).
#
# Paper: supports the Methods initialisation paragraph + feeds the
# multi-label fingerprint discussion (min-label coverage proposal).

# %%
import csv
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gridbench.functional_graph.decomposition import decompose, deterministic_successor
from gridbench.functional_graph.probe_env import build_goal_free_probe_env

BASE = Path("/media/merlin/grid-twist/gridtwist-outputs")
COHORTS = {
    "perm_balanced": "core-silence-hunt-10-07",
    "shuffle": "core-silence-hunt2-shuffle-10-07",
}
ACT = "NESW"
FIG_DIR = Path(__file__).resolve().parent / "figs"
FIG_DIR.mkdir(exist_ok=True)

env = build_goal_free_probe_env("four_rooms", (7, 7), 0.97)
WALLS = {4, 11, 25, 28, 29, 31, 32, 33, 38}
N_WALK = 49 - len(WALLS)
BASE_SUCC = np.stack([deterministic_successor(env, a) for a in range(4)], axis=0)
IDX = np.arange(49)


def four_label_coverage(sigma: np.ndarray) -> np.ndarray:
    sigma_inv = np.argsort(sigma, axis=1)
    out = np.empty(4)
    for l in range(4):
        succ = BASE_SUCC[sigma_inv[:, l], IDX]
        bs = np.asarray(decompose(succ, walls=WALLS).basin_sizes, dtype=int)
        out[l] = bs.max() / N_WALK if bs.size else 0.0
    return out


# %% Compute per-generation strips (dedupe consecutive identical sigmas)
strips = {}   # (cohort, seed) -> (gens, 4) coverage array
stats = []
for cohort, base in COHORTS.items():
    for sj in sorted(glob.glob(str(BASE / base / "*" / "*-multi-all.summary.json"))):
        summ = json.load(open(sj))
        seed = f"s{str(summ['config']['seed'])[:3]}"
        hist = summ["history"]
        cov = np.empty((len(hist), 4))
        prev_key, prev_cov = None, None
        for i, h in enumerate(hist):
            key = h.get("best_sigma_hash")
            if key is not None and key == prev_key:
                cov[i] = prev_cov
            else:
                cov[i] = four_label_coverage(np.asarray(h["best_sigma"], dtype=int))
                prev_key, prev_cov = key, cov[i]
        strips[(cohort, seed)] = cov
        dom = cov.max(axis=1)
        first70 = int(np.argmax(dom >= 0.70)) if (dom >= 0.70).any() else -1
        first85 = int(np.argmax(dom >= 0.85)) if (dom >= 0.85).any() else -1
        final = cov[-1]
        stats.append({"cohort": cohort, "seed": seed,
                      "first_gen_cov70": first70, "first_gen_cov85": first85,
                      **{f"final_{ACT[l]}": round(float(final[l]), 3) for l in range(4)},
                      "final_min_label": round(float(final.min()), 3),
                      "n_labels_ge_070": int((final >= 0.70).sum())})
        print(f"{cohort:14} {seed}: cov70@gen {first70:>3}  cov85@gen {first85:>3}  "
              f"final {np.round(final, 2)}  min={final.min():.2f}", flush=True)

with open(FIG_DIR / "seed_strips_saturation.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(stats[0]))
    w.writeheader()
    w.writerows(stats)

# %% Heat-strip figure
fig, axes = plt.subplots(1, 2, figsize=(16, 10), dpi=150, sharey=False)
for ax, cohort in zip(axes, COHORTS):
    keys = sorted(k for k in strips if k[0] == cohort)
    n = len(keys)
    gens = max(strips[k].shape[0] for k in keys)
    img = np.full((n * 5 - 1, gens), np.nan)
    for r, k in enumerate(keys):
        img[r * 5:r * 5 + 4] = strips[k].T
    im = ax.imshow(img, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_yticks([r * 5 + 1.5 for r in range(n)])
    ax.set_yticklabels([k[1] for k in keys], fontsize=8)
    ax.set_xlabel("generation")
    ax.set_title(f"four_rooms 7x7 · {cohort} · rows per seed = labels N/E/S/W")
fig.colorbar(im, ax=axes, label="per-label coverage", shrink=0.6)
out = FIG_DIR / "F-seed-strips-four-label.png"
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}")
