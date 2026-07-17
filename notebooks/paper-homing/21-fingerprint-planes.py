# %% [markdown]
# # 21 — Fingerprint planes (ported from gridFour attractor-fingerprint-probe/21)
#
# Canonical home of the paper figure code (2026-07-17, per "all analysis
# in gridBench"); the gridFour copy is deprecated.
#
# Paper figure for twists-home-vectors: the population view of the
# Cartesian -> home-vector journey in the minimal fingerprint plane
# (fp_n_basins x fp_cycle_basin_ratio), coloured by mean_free.  One panel
# per toroidal env: wrap_grid (the full journey: Cartesian at (7, 1.0),
# run-bests at (~1, ~0.1)) and helical (the polar-star control: the seam
# pre-pays half the journey, Cartesian already at n_basins=4).
#
# Only the two tori are shown deliberately: the metric-selection analysis
# (attractor-fingerprint-report S3-4) found this fingerprint pair expressive
# exactly on toroidal/open geometry and compressed behind walls, where
# coverage carries the story instead.  The full per-env collection lives at
# the GitLab Pages expressive-range-fingerprint report (Christoph's request).
#
# Cohort safety: run_type == "multi" and fitness_objective in
# {decision_information, free_energy, NaN} (the licensed beta=1 pooling);
# no fepm rows exist in these caches.
#
# Output: figs/fingerprint_journey_two_panel.png

# %%
import sys
from pathlib import Path

try:
    _NB_PATH = Path(__file__).resolve()
except NameError:  # Jupyter cell context
    _NB_PATH = Path(
        "/media/merlin/phd-marlyn/gridBench/notebooks/paper-homing"
        "/21-fingerprint-planes.py"
    )

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from gridbench.functional_graph.decomposition import (
    decompose, deterministic_successor, per_label_stats,
)
from gridbench.functional_graph.probe_env import build_goal_free_probe_env

FIG_DIR = _NB_PATH.parent / "figs"
FIG_DIR.mkdir(exist_ok=True)

CACHE_ROOT = Path("/media/merlin/grid-twist/data-schema-11/cache/functional_graph")
DET = 0.97
SHAPE = (7, 7)
# Six environments on ONE fingerprint pair (2026-07-16): the ratio plane,
# grouped as in the paper's environment palette figure.  Coverage is out of
# the fingerprint (cluster analysis showed it mixes budget with structure);
# the seed-strips figure carries coverage instead.
PANELS_OPEN = ["wrap_grid", "open_grid", "helical"]
PANELS_WALLED = ["pinwheel", "four_rooms", "pillar_3"]

# Consistent colour treatment across panels (2026-07-12): every colourbar
# spans the SAME width in free-energy bits (per-env window centred on the
# panel median, width = the widest panel's robust range) and carries the
# same number of ticks, so colour distances compare across panels.
from matplotlib import ticker as _mticker


def _shared_clims(dfs, col="mean_free"):
    spans = [float(d[col].quantile(0.995) - d[col].quantile(0.005)) for d in dfs]
    width = max(spans)
    lims = []
    for d in dfs:
        mid = float(d[col].median())
        lims.append((mid - width / 2, mid + width / 2))
    return lims


def _uniform_colorbar(fig, sc, ax):
    cb = fig.colorbar(sc, ax=ax, label="mean free energy", shrink=0.85)
    cb.locator = _mticker.MaxNLocator(nbins=5)
    cb.update_ticks()
    return cb

COHORT_OBJECTIVES = {"decision_information", "free_energy"}  # NaN kept separately
N_NULL = 2000
CACHE_DIR = _NB_PATH.parent / "_cache"
CACHE_DIR.mkdir(exist_ok=True)


# %%
def cartesian_anchor(env_id: str) -> tuple[float, float]:
    """(mean n_basins, mean cycle/basin ratio) of the identity twist.
    Walls must be passed to decompose() as inert (matters for pillar_3)."""
    env = build_goal_free_probe_env(env_id, SHAPE, DET)
    walls = [s for s in range(int(env.nS)) if not env.T[s].any()]
    nb, cbr = [], []
    for a in range(int(env.nA)):
        st = per_label_stats(decompose(deterministic_successor(env, a), walls=walls))
        nb.append(st["n_basins"])
        cbr.append(st["cycle_basin_ratio"])
    return float(np.mean(nb)), float(np.mean(cbr))


def k_equals_one(run_names: pd.Series) -> pd.Series:
    """True for full-goal (K=1) runs.  Goal-subsampled campaigns are tagged
    'gss' or 'k0xx' in run_name (untagged runs predate subsampling and are
    K=1 by construction); K<1 runs are excluded throughout the paper."""
    return ~(run_names.str.contains("gss") | run_names.str.contains(r"k0\d\d", regex=True))


def load_cohort(env_id: str) -> pd.DataFrame:
    p = (CACHE_ROOT / f"env_id={env_id}/shape={SHAPE[0]}x{SHAPE[1]}"
         / f"det={DET}/beta=1/fingerprints.parquet")
    df = pd.read_parquet(p, columns=[
        "fp_n_basins", "fp_cycle_basin_ratio", "mean_free",
        "is_run_best", "run_type", "fitness_objective", "run_name",
        "init_method",
    ])
    keep = (df.run_type == "multi") & (
        df.fitness_objective.isin(COHORT_OBJECTIVES) | df.fitness_objective.isna()
    ) & k_equals_one(df.run_name)
    return df[keep]


def random_null_fp(env_id: str, n: int = N_NULL, seed: int = 20260704):
    """(mean n_basins, mean cycle/basin ratio, coverage) for n uniform-random
    twists — the no-selection null cloud on both fingerprint planes.
    Per-label successors are assembled explicitly as
    f_{sigma,label}(s) = base[sigma_inv[s, label], s]; walls passed to
    decompose() as inert.  Cached per env (delete the npz to recompute)."""
    cache = CACHE_DIR / f"random_null_fp_{env_id}_{SHAPE[0]}x{SHAPE[1]}_n{n}.npz"
    if cache.exists():
        c = np.load(cache)
        return c["nb"], c["cbr"], c["cov"]
    env = build_goal_free_probe_env(env_id, SHAPE, DET)
    nS, nA = int(env.nS), int(env.nA)
    walls = [s for s in range(nS) if not env.T[s].any()]
    n_nonwall = nS - len(walls)
    base = np.stack([deterministic_successor(env, a) for a in range(nA)])
    idx = np.arange(nS)
    rng = np.random.default_rng(seed)
    nb, cbr, cov = np.zeros(n), np.zeros(n), np.zeros(n)
    for k in range(n):
        sigma = np.stack([rng.permutation(nA) for _ in range(nS)])
        sigma_inv = np.argsort(sigma, axis=1)
        nbs, cbrs, covs = [], [], []
        for label in range(nA):
            fg = decompose(base[sigma_inv[:, label], idx], walls=walls)
            st = per_label_stats(fg)
            sizes = np.asarray(fg.basin_sizes, dtype=int)
            nbs.append(st["n_basins"])
            cbrs.append(st["cycle_basin_ratio"])
            covs.append(float(sizes.max()) / n_nonwall)
        nb[k] = float(np.mean(nbs))
        cbr[k] = float(np.mean(cbrs))
        cov[k] = float(np.max(covs))
    np.savez_compressed(cache, nb=nb, cbr=cbr, cov=cov)
    return nb, cbr, cov


def null_footprint(ax, xs, ys, label=None):
    """Draw the 95%/99% KDE footprint of a null cloud."""
    kde = gaussian_kde(np.vstack([xs, ys]))
    dens = kde(np.vstack([xs, ys]))
    xg = np.linspace(xs.min() - 0.6, xs.max() + 0.6, 140)
    yg = np.linspace(max(0.0, ys.min() - 0.06), min(1.02, ys.max() + 0.08), 140)
    XX, YY = np.meshgrid(xg, yg)
    ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    ax.contour(XX, YY, ZZ,
               levels=sorted([np.percentile(dens, 1), np.percentile(dens, 5)]),
               colors="#ee6677", linewidths=[1.0, 1.6], zorder=4)
    if label:
        ax.plot([], [], color="#ee6677", lw=1.4, label=label)


PRIOR_CSV = ("/media/merlin/phd-marlyn/gridBench/notebooks/twist-generation/"
             "_cache/initialiser-fingerprints-v2-7x7-n2016-seed20260716.csv.gz")
_PRIORS = pd.read_csv(PRIOR_CSV)
PRIOR_STYLE = {"row_shuffle": ("#4477aa", "gen-0 prior: row-shuffle"),
               "perm_balanced": ("#228833", "gen-0 prior: permutation-balanced")}


def prior_footprints(ax, env_id, xcol, ycol, bits=False):
    """Two generation-zero prior footprints (95/99% KDE), replacing the
    uniform null: fresh-uniform and row-shuffle are the same law over
    fingerprint rows (twist-generation RESULT 2026-07-16)."""
    thresholds = {}
    for gen, (col, lab) in PRIOR_STYLE.items():
        d = _PRIORS[(_PRIORS.env_id == env_id) & (_PRIORS.generator == gen)]
        yv = -np.log2(np.clip(d[ycol], 1e-6, None)) if bits else d[ycol]
        xy = np.vstack([d[xcol], yv])
        kde = gaussian_kde(xy)
        dens = kde(xy)
        t95, t99 = np.percentile(dens, 5), np.percentile(dens, 1)
        xg = np.linspace(d[xcol].min() - 1, d[xcol].max() + 1, 160)
        ylo, yhi = float(np.min(yv)), float(np.max(yv))
        pad = 0.05 * (yhi - ylo + 1e-9)
        yg = np.linspace(ylo - pad, yhi + pad, 160)
        XG, YG = np.meshgrid(xg, yg)
        Z = kde(np.vstack([XG.ravel(), YG.ravel()])).reshape(XG.shape)
        ax.contour(XG, YG, Z, levels=[t99, t95], colors=col,
                   linewidths=[0.9, 1.8])
        ax.plot([], [], color=col, lw=1.8, label=f"{lab} (95/99%)")
        thresholds[gen] = (kde, t99)
    return thresholds


def prior_overlap(thresholds, xs, ys):
    out = {}
    for gen, (kde, t99) in thresholds.items():
        dens = kde(np.vstack([xs, ys]))
        out[gen] = int((dens >= t99).sum())
    return out

# %% [markdown]
# ## Star cohort and the two grouped ratio-plane figures
#
# Run-bests filter to the production objective and evaluation (free-energy
# fitness, K=1); plotted as plain red stars (2026-07-16: no budget colours,
# no initialiser markers -- the init story lives in the seed strips).
# Synced-but-unimported July batches are supplemented from run-best sigmas.

# %%
import glob as _glob
import re as _re

OUT_MIRROR = "/media/merlin/grid-twist/gridtwist-outputs"
SUPP_BASES = [
    "core-silence-hunt-10-07",
    "core-silence-hunt2-shuffle-10-07",
    "core-silence-scale-env-warm-11-07",
    "g400-pop-96-perm-bal-13-05-b1-free-existing-envs",
    "core-torus-shuffle-15-07",
    "core-star-equalise-A-16-07",
]


def supplement_stars(env_id, known_runs):
    tag = "-" + env_id.replace("_", "-") + "-7x7-"
    env = build_goal_free_probe_env(env_id, SHAPE, DET)
    wf = getattr(env, "walls_flat", None)
    walls = set(int(w) for w in np.ravel(wf)) if wf is not None else set()
    base_succ = np.stack([deterministic_successor(env, a) for a in range(4)], axis=0)
    idx = np.arange(SHAPE[0] * SHAPE[1])
    rows = []
    for b in SUPP_BASES:
        for run in _glob.glob(f"{OUT_MIRROR}/{b}/*"):
            name = run.rsplit("/", 1)[1]
            if tag not in name or "-b1-free-" not in name or name in known_runs:
                continue
            sg = _glob.glob(run + "/*-multi-all.sigma.npy")
            if not sg:
                continue
            sigma = np.load(sg[0]).astype(int)
            sigma_inv = np.argsort(sigma, axis=1)
            nbas, cbrs = [], []
            for l in range(4):
                st = per_label_stats(decompose(base_succ[sigma_inv[:, l], idx],
                                               walls=walls))
                nbas.append(st["n_basins"])
                cbrs.append(st["cycle_basin_ratio"])
            rows.append({"run_name": name,
                         "fp_n_basins": float(np.mean(nbas)),
                         "fp_cycle_basin_ratio": float(np.mean(cbrs))})
    return pd.DataFrame(rows)


def star_cohort(df, env_id):
    rb = df[df.is_run_best & (df.fitness_objective == "free_energy")]
    rb = rb.drop_duplicates("run_name")[
        ["run_name", "fp_n_basins", "fp_cycle_basin_ratio"]]
    supp = supplement_stars(env_id, set(rb.run_name))
    return pd.concat([rb, supp], ignore_index=True) if len(supp) else rb


def _ycoord(vals, bits):
    v = np.asarray(vals, dtype=float)
    return -np.log2(np.clip(v, 1e-6, None)) if bits else v


def make_ratio_fig(panels, outname, bits=False):
    """The paper fingerprint figure; bits=True re-unitises the y axis as
    Daniel's free-information reading, -log2 F (2026-05-22 meeting), a
    monotone relabelling of the same geometry."""
    fig, axes = plt.subplots(1, 3, figsize=(18.6, 5.2), dpi=150)
    dfs = [load_cohort(e) for e in panels]
    lims = _shared_clims(dfs)
    for ax, env_id, df, (vmin, vmax) in zip(axes, panels, dfs, lims):
        sc = ax.scatter(df.fp_n_basins, _ycoord(df.fp_cycle_basin_ratio, bits),
                        c=df.mean_free,
                        s=4, alpha=0.25, cmap="viridis", rasterized=True,
                        vmin=vmin, vmax=vmax,
                        label=rf"all evaluated $\sigma$ (n={len(df):,})")
        prior_footprints(ax, env_id, "fp_n_basins", "fp_cycle_basin_ratio",
                         bits=bits)
        rb = star_cohort(df, env_id)
        ax.scatter(rb.fp_n_basins, _ycoord(rb.fp_cycle_basin_ratio, bits),
                   marker="*", s=210,
                   facecolors="none", edgecolors="crimson", linewidths=1.5,
                   label=f"GA run-bests, free-energy $K{{=}}1$ (n={len(rb)})",
                   zorder=5)
        print(f"{env_id}: stars n={len(rb)}  cbr med="
              f"{rb.fp_cycle_basin_ratio.median():.2f}  nb med="
              f"{rb.fp_n_basins.median():.2f}", flush=True)
        cx, cy = cartesian_anchor(env_id)
        ax.scatter([cx], [_ycoord([cy], bits)[0]], marker="o", s=70,
                   facecolors="none",
                   edgecolors="black", linewidths=1.8,
                   label="Cartesian identity", zorder=6)
        if bits:
            ax.set_ylim(-0.15, 4.4)
            ax.set_ylabel(r"free information  $-\log_2 F$  (bits)")
        else:
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("mean cycle/basin ratio  (fp_cycle_basin_ratio)")
        ax.set_xlabel("mean basins per label  (fp_n_basins)")
        ax.set_title(f"{env_id}  {SHAPE[0]}x{SHAPE[1]}  " + r"$\beta=1$",
                     fontsize=12)
        leg = ax.legend(fontsize=8, loc="best", framealpha=0.9)
        for h in leg.legend_handles:
            try:
                h.set_alpha(1.0)
            except AttributeError:
                pass
        _uniform_colorbar(fig, sc, ax)
    fig.tight_layout()
    out = FIG_DIR / outname
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


make_ratio_fig(PANELS_OPEN, "fingerprint_open_interiors.png")
make_ratio_fig(PANELS_WALLED, "fingerprint_walled_interiors.png")
# Candidate variants in Daniel's units (2026-07-17): same geometry, y in
# bits of free information.  Swap into the paper only if adopted.
make_ratio_fig(PANELS_OPEN, "fingerprint_open_interiors_bits.png", bits=True)
make_ratio_fig(PANELS_WALLED, "fingerprint_walled_interiors_bits.png", bits=True)

# %% sanity
for e in PANELS_OPEN + PANELS_WALLED:
    print(f"sanity: {e} Cartesian anchor =",
          tuple(round(v, 3) for v in cartesian_anchor(e)))


# %% [markdown]
# ## Round-up figure (2026-07-17): chi_twist of the run-bests, and the two
# four_rooms families on the ratio plane
#
# (a) Run-best chi_twist per environment, palette order, initialiser by
# marker, against the random-pool chance band and Cartesian zero: the
# compass-character evidence of sec:reach-recovered, drawn.
# (b) four_rooms ratio plane with the two initialisation families
# distinguished: where the row-shuffle run-bests sit relative to the
# permutation-balanced family.

# %%
from itertools import permutations as _perms
ORDS = np.array(list(_perms(range(4))))
STORE = "/media/merlin/grid-twist/data-schema-11/multi"


def chi_of(sigma, walls):
    rows = sigma[[s for s in range(sigma.shape[0]) if s not in walls]]
    return 1 - float((rows[None] == ORDS[:, None, :]).mean(axis=(1, 2)).max())


def runbest_chis(env_id):
    """(chi, init) for every FE full-goal 7x7 run-best: store + supplements."""
    env = build_goal_free_probe_env(env_id, SHAPE, DET)
    wf = getattr(env, "walls_flat", None)
    walls = set(int(w) for w in np.ravel(wf)) if wf is not None else set()
    out, seen = [], set()
    pats = ([f"{STORE}/init_method=*/fitness_objective=free_energy/env_id={env_id}/"
             f"shape=7x7/beta=1/det=0.97/run_name=*/*-multi-all.sigma.npy"] +
            [f"{OUT_MIRROR}/{b}/*{'-' + env_id.replace('_','-') + '-7x7-'}*/*-multi-all.sigma.npy"
             for b in SUPP_BASES])
    for pat in pats:
        for sg in _glob.glob(pat):
            name = Path(sg).parent.name.replace("run_name=", "")
            if name in seen or "gss" in name or _re.search(r"k0\d\d", name):
                continue
            if OUT_MIRROR in sg and "-b1-free-" not in name:
                continue
            seen.add(name)
            init = "shuffle" if ("shuffle" in name or "prod-baseline" in name) \
                else "perm_balanced"
            out.append((chi_of(np.load(sg).astype(int), walls), init))
    return out


def palette_metrics(sigma, walls, n_walk, base_succ, idx):
    """union coverage of label attractor-cycle cells + mean pairwise
    Jaccard between the labels' cycle sets (gridFour nb-02 definitions)."""
    sigma_inv = np.argsort(np.asarray(sigma, dtype=int), axis=1)
    cycle_sets = []
    for l in range(4):
        fg = decompose(base_succ[sigma_inv[:, l], idx], walls=walls)
        s = set()
        for c in fg.cycles:
            s.update(c)
        cycle_sets.append(s)
    jac = [len(a & b) / len(a | b) if (a | b) else 0.0
           for i, a in enumerate(cycle_sets) for b in cycle_sets[i + 1:]]
    return (len(set().union(*cycle_sets)) / n_walk,
            float(np.mean(jac)) if jac else 0.0)


def fourrooms_palette_cohorts(n_null=300):
    env_id = "four_rooms"
    env = build_goal_free_probe_env(env_id, SHAPE, DET)
    walls = set(int(w) for w in np.ravel(getattr(env, "walls_flat")))
    n_walk = SHAPE[0] * SHAPE[1] - len(walls)
    base_succ = np.stack([deterministic_successor(env, a) for a in range(4)],
                         axis=0)
    idx = np.arange(SHAPE[0] * SHAPE[1])
    rows, seen = [], set()
    pats = ([f"{STORE}/init_method=*/fitness_objective=free_energy/env_id={env_id}/"
             "shape=7x7/beta=1/det=0.97/run_name=*/*-multi-all.sigma.npy"] +
            [f"{OUT_MIRROR}/{b}/*-four-rooms-7x7-*/*-multi-all.sigma.npy"
             for b in SUPP_BASES])
    for pat in pats:
        for sg in _glob.glob(pat):
            name = Path(sg).parent.name.replace("run_name=", "")
            if name in seen or "gss" in name or _re.search(r"k0\d\d", name):
                continue
            if OUT_MIRROR in sg and "-b1-free-" not in name:
                continue
            seen.add(name)
            u, j = palette_metrics(np.load(sg), walls, n_walk, base_succ, idx)
            rows.append({"union": u, "jaccard": j,
                         "group": "shuffle" if ("shuffle" in name or
                                                "prod-baseline" in name)
                         else "perm_balanced"})
    rng = np.random.default_rng(20260718)
    for _ in range(n_null):
        sig = np.stack([rng.permutation(4) for _ in range(SHAPE[0] * SHAPE[1])])
        u, j = palette_metrics(sig, walls, n_walk, base_succ, idx)
        rows.append({"union": u, "jaccard": j, "group": "null"})
    return pd.DataFrame(rows)


PALETTE_ORDER = PANELS_OPEN + PANELS_WALLED
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(19.6, 5.4), dpi=150,
                                    gridspec_kw={"width_ratios": [1.2, 1, 1]})

# (a) chi strip
rng = np.random.default_rng(7)
for i, env_id in enumerate(PALETTE_ORDER):
    _chis = runbest_chis(env_id)
    for init in ("perm_balanced", "shuffle"):
        v = [c for c, ini in _chis if ini == init]
        if v:
            print(f"chi {env_id} {init}: n={len(v)} med={np.median(v):.2f} "
                  f"[{min(v):.2f}-{max(v):.2f}]", flush=True)
    for chi, init in _chis:
        m = "*" if init == "perm_balanced" else "s"
        s = 150 if init == "perm_balanced" else 55
        axA.scatter(i + rng.uniform(-0.16, 0.16), chi, marker=m, s=s,
                    facecolors="none", edgecolors="crimson", linewidths=1.3)
axA.axhspan(0.57, 0.73, color="grey", alpha=0.18,
            label="uniform-random pool (min-max)")
axA.axhline(0.68, color="grey", lw=1.0, ls="--", label="random median")
axA.axhline(0.0, color="black", lw=1.0, label="Cartesian (any rotation)")
axA.scatter([], [], marker="*", s=150, facecolors="none", edgecolors="crimson",
            label="run-best, permutation-balanced")
axA.scatter([], [], marker="s", s=55, facecolors="none", edgecolors="crimson",
            label="run-best, row-shuffle")
axA.set_xticks(range(len(PALETTE_ORDER)))
axA.set_xticklabels(PALETTE_ORDER,
                    rotation=20, ha="right", fontsize=9)
axA.axvline(2.5, color="lightgrey", lw=0.8)
axA.set_ylabel(r"run-best $\chi_{\mathrm{twist}}$")
axA.set_ylim(-0.04, 0.8)
axA.set_title("Compass character of the run-bests (palette order)", fontsize=11)
axA.legend(fontsize=7, loc="lower right", framealpha=0.9)

# (b) four_rooms ratio plane, families split
env_id = "four_rooms"
df = load_cohort(env_id)
sc = axB.scatter(df.fp_n_basins, df.fp_cycle_basin_ratio, c=df.mean_free,
                 s=4, alpha=0.25, cmap="viridis", rasterized=True,
                 label=rf"all evaluated $\sigma$ (n={len(df):,})")
prior_footprints(axB, env_id, "fp_n_basins", "fp_cycle_basin_ratio")
rb = df[df.is_run_best & (df.fitness_objective == "free_energy")]
rb = rb.drop_duplicates("run_name")[["run_name", "init_method",
                                     "fp_n_basins", "fp_cycle_basin_ratio"]]
supp = supplement_stars(env_id, set(rb.run_name))
supp["init_method"] = ["shuffle" if "shuffle" in n else "perm_balanced"
                       for n in supp.run_name]
allrb = pd.concat([rb, supp], ignore_index=True)
for init, m, s, lab in [("perm_balanced", "*", 210, "permutation-balanced run-bests"),
                        ("shuffle", "s", 70, "row-shuffle run-bests")]:
    g = allrb[allrb.init_method == init]
    axB.scatter(g.fp_n_basins, g.fp_cycle_basin_ratio, marker=m, s=s,
                facecolors="none", edgecolors="crimson" if m == "*" else "black",
                linewidths=1.5, label=f"{lab} (n={len(g)})", zorder=5)
    print(f"four_rooms {init}: n={len(g)} cbr med={g.fp_cycle_basin_ratio.median():.2f} "
          f"nb med={g.fp_n_basins.median():.2f}", flush=True)
cx, cy = cartesian_anchor(env_id)
axB.scatter([cx], [cy], marker="o", s=70, facecolors="none",
            edgecolors="black", linewidths=1.8, label="Cartesian identity", zorder=6)
axB.set_ylim(0, 1.05)
axB.set_xlabel("mean basins per label  (fp_n_basins)")
axB.set_ylabel("mean cycle/basin ratio  (fp_cycle_basin_ratio)")
axB.set_title("four_rooms 7x7: the ratio plane cannot separate them", fontsize=11)
leg = axB.legend(fontsize=7, loc="lower right", framealpha=0.9)
for h in leg.legend_handles:
    try:
        h.set_alpha(1.0)
    except AttributeError:
        pass
_uniform_colorbar(fig, sc, axB)

# (c) the palette plane: the relational statistics separate the families
pal = fourrooms_palette_cohorts()
nul = pal[pal.group == "null"]
axC.scatter(nul.union, nul.jaccard, s=14, alpha=0.35, color="lightgrey",
            label=f"row-shuffle null (n={len(nul)})")
for grp, m, s, col, lab in [
        ("perm_balanced", "*", 170, "crimson", "permutation-balanced run-bests"),
        ("shuffle", "s", 60, "black", "row-shuffle run-bests")]:
    g = pal[pal.group == grp]
    axC.scatter(g.union, g.jaccard, marker=m, s=s, facecolors="none",
                edgecolors=col, linewidths=1.4, label=f"{lab} (n={len(g)})")
    print(f"palette {grp}: n={len(g)} union med={g.union.median():.3f} "
          f"jaccard med={g.jaccard.median():.3f}", flush=True)
axC.set_xlabel("union coverage of label cycles")
axC.set_ylabel("mean pairwise Jaccard of cycle sets")
axC.set_title("four_rooms 7x7: the palette plane separates them", fontsize=11)
axC.legend(fontsize=7, framealpha=0.9)

fig.tight_layout()
out = FIG_DIR / "alignment_roundup.png"
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}")
