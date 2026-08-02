# %% [markdown]
# # 01 — Q modularity as a specialisation-versus-homing diagnostic
#
# This is the one promoted analysis from gridFour's Louvain/fingerprint work.
# It asks whether basin co-membership modularity contributes a reproducible
# structural contrast on the current gridCore + schema-11 stack.
#
# Scope is intentionally narrow:
#
# - 7x7, beta=1, determinism=0.97;
# - pure free-energy, full-goal run-best sigmas only;
# - Cartesian and fresh environment-matched uniform-random controls;
# - no FEP+M campaign, goal-subsampling, K-annealing, or scale fan;
# - Q means agreement between labels' basin partitions, not spatial adjacency.

# %%
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from gridbench.functional_graph.fingerprint import fingerprint_for_sigma
from gridbench.functional_graph.modularity import modularity_diagnostics_for_sigma
from gridbench.functional_graph.probe_env import build_goal_free_probe_env
from gridbench.store import data_root, multi_root
from gridcore.twists import sigma_hash


def _nb_dir(default: str) -> Path:
    """Jupytext/Jupyter-safe notebook directory."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        cwd = Path.cwd().resolve()
        default_path = Path(default)
        if (
            (cwd / default_path.name).exists()
            or cwd.name == default_path.parent.name
        ):
            return cwd
        return default_path.parent


NB_DIR = _nb_dir(
    "/home/karen/phd-marlyn/gridBench/notebooks/fingerprint-metrics/"
    "01-q-specialisation-screen.py"
)
FIG_DIR = NB_DIR / "figures"
ARTIFACT_DIR = NB_DIR / "artifacts"
FIG_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

ENVS = [
    "open_grid",
    "wrap_grid",
    "helical",
    "pinwheel",
    "four_rooms",
    "pillar_3",
]
DISPLAY_NAMES = {
    "open_grid": "Open",
    "wrap_grid": "Wrap",
    "helical": "Helical",
    "pinwheel": "Pinwheel",
    "four_rooms": "Four-rooms",
    "pillar_3": "Pillar-3",
}
ENV_COLOURS = dict(
    zip(
        ENVS,
        [
            "#4477aa",
            "#66ccee",
            "#228833",
            "#ccbb44",
            "#ee6677",
            "#aa3377",
        ],
    )
)

SHAPE = (7, 7)
DETERMINISM = 0.97
BETA = 1
N_RANDOM = int(os.environ.get("GRIDBENCH_Q_NULL_SAMPLES", "120"))
N_RANDOM_LOUVAIN_SEEDS = 5
N_GA_LOUVAIN_SEEDS = 10
RANDOM_BASE_SEED = 20260802
ANALYSIS_VERSION = 1

# Explicitly excludes the historical material that motivated this selective
# port.  k100/K1 are full-goal controls and are retained; k0xx are not.
EXCLUDED_RUN_NAME = re.compile(
    r"gss|k0\d\d|anneal|fepm|random|rand-", re.IGNORECASE
)
FINGERPRINT_AXES = [
    "fp_largest_basin_fraction",
    "fp_n_basins",
    "fp_cycle_basin_ratio",
]


# %% [markdown]
# ## Clean schema-11 cohort
#
# The store currently has no run.parquet manifests for these imported runs, so
# selection is made from schema-11's first-class partition path and checked
# against each run summary. A missing or non-full-goal summary is rejected.

# %%
def _summary_for_sigma_path(sigma_path: Path) -> dict:
    summaries = list(sigma_path.parent.glob("*-multi-all.summary.json"))
    if len(summaries) != 1:
        raise RuntimeError(
            f"expected one summary beside {sigma_path}, found {len(summaries)}"
        )
    return json.loads(summaries[0].read_text())


def _is_full_goal_free_energy(summary: dict) -> bool:
    config = summary.get("config", {})
    objective = summary.get("fitness_objective", config.get("fitness_objective"))
    if objective != "free_energy" or config.get("goal_mode") != "all":
        return False
    fraction = config.get("goal_subsample_fraction")
    return fraction is None or float(fraction) == 1.0


def clean_runbest_sigmas(env_id: str):
    pattern = (
        f"init_method=*/fitness_objective=free_energy/env_id={env_id}/"
        "shape=7x7/beta=1/det=0.97/run_name=*/*-multi-all.sigma.npy"
    )
    seen = set()
    selected = []
    for sigma_path in sorted(multi_root().glob(pattern)):
        run_name = sigma_path.parent.name.replace("run_name=", "")
        if EXCLUDED_RUN_NAME.search(run_name):
            continue
        summary = _summary_for_sigma_path(sigma_path)
        if not _is_full_goal_free_energy(summary):
            continue
        sigma = np.load(sigma_path).astype(int)
        digest = sigma_hash(sigma)
        if digest in seen:
            continue
        seen.add(digest)
        init_method = next(
            part.split("=", 1)[1]
            for part in sigma_path.parts
            if part.startswith("init_method=")
        )
        selected.append((sigma, digest, run_name, init_method))
    return selected


def random_sigma(env, rng: np.random.Generator) -> np.ndarray:
    return np.stack([rng.permutation(env.nA) for _ in range(env.nS)])


def metrics_for_sigma(env, sigma: np.ndarray, *, n_louvain_seeds: int) -> dict:
    fingerprint = fingerprint_for_sigma(env, sigma)
    q = modularity_diagnostics_for_sigma(
        env, sigma, n_seeds=n_louvain_seeds
    )
    return {
        **fingerprint,
        "Q_modularity": q.mean_q,
        "Q_seed_sd": q.std_q,
        "Q_partition_ari_median": q.median_pairwise_ari,
        "Q_partition_ari_min": q.min_pairwise_ari,
        "Q_n_communities_median": float(np.median(q.n_communities)),
    }


# %% [markdown]
# ## Compute observations

# %%
rows = []
for env_index, env_id in enumerate(ENVS):
    env = build_goal_free_probe_env(env_id, SHAPE, DETERMINISM)
    runbests = clean_runbest_sigmas(env_id)
    if not runbests:
        raise RuntimeError(f"no clean full-goal free-energy run-bests for {env_id}")

    for sigma, digest, run_name, init_method in runbests:
        rows.append(
            {
                "env": env_id,
                "group": "ga",
                "init_method": init_method,
                "run_name": run_name,
                "sigma_hash": digest,
                "sample_index": -1,
                "random_seed": -1,
                "n_louvain_seeds": N_GA_LOUVAIN_SEEDS,
                **metrics_for_sigma(
                    env, sigma, n_louvain_seeds=N_GA_LOUVAIN_SEEDS
                ),
            }
        )

    rng = np.random.default_rng(RANDOM_BASE_SEED + env_index)
    for sample_index in range(N_RANDOM):
        sigma = random_sigma(env, rng)
        rows.append(
            {
                "env": env_id,
                "group": "random",
                "init_method": "uniform_random",
                "run_name": "",
                "sigma_hash": sigma_hash(sigma),
                "sample_index": sample_index,
                "random_seed": RANDOM_BASE_SEED + env_index,
                "n_louvain_seeds": N_RANDOM_LOUVAIN_SEEDS,
                **metrics_for_sigma(
                    env, sigma, n_louvain_seeds=N_RANDOM_LOUVAIN_SEEDS
                ),
            }
        )

    identity = np.tile(np.arange(env.nA), (env.nS, 1))
    rows.append(
        {
            "env": env_id,
            "group": "cartesian",
            "init_method": "identity",
            "run_name": "",
            "sigma_hash": sigma_hash(identity),
            "sample_index": -1,
            "random_seed": -1,
            "n_louvain_seeds": N_GA_LOUVAIN_SEEDS,
            **metrics_for_sigma(
                env, identity, n_louvain_seeds=N_GA_LOUVAIN_SEEDS
            ),
        }
    )
    print(f"{env_id}: {len(runbests)} GA run-bests + {N_RANDOM} random")

observations = pd.DataFrame(rows)
observations["analysis_version"] = ANALYSIS_VERSION
observations["data_root"] = str(data_root())
observations["shape_h"] = SHAPE[0]
observations["shape_w"] = SHAPE[1]
observations["beta"] = BETA
observations["determinism"] = DETERMINISM


# %% [markdown]
# ## Environment-matched null standardisation and incremental Q
#
# Raw Q is strongly environment-dependent. Every scalar comparison therefore
# uses its environment's random pool as the reference. To ask whether Q merely
# renames the current fingerprint, Q is regressed on coverage, basin count and
# cycle/basin ratio in the random pool; the GA residual is then measured in
# random-residual standard deviations.

# %%
for metric in ["Q_modularity", *FINGERPRINT_AXES]:
    observations[f"z_{metric}"] = np.nan
observations["Q_incremental_residual_z"] = np.nan

summary_rows = []
for env_id in ENVS:
    env_mask = observations.env == env_id
    random_mask = env_mask & observations.group.eq("random")
    ga_mask = env_mask & observations.group.eq("ga")
    cartesian_mask = env_mask & observations.group.eq("cartesian")

    random = observations.loc[random_mask]
    ga = observations.loc[ga_mask]
    cartesian = observations.loc[cartesian_mask]

    for metric in ["Q_modularity", *FINGERPRINT_AXES]:
        mean = random[metric].mean()
        std = random[metric].std(ddof=1)
        observations.loc[env_mask, f"z_{metric}"] = (
            observations.loc[env_mask, metric] - mean
        ) / std

    random_z = observations.loc[random_mask]
    ga_z = observations.loc[ga_mask]
    x_random = np.column_stack(
        [np.ones(len(random_z))]
        + [random_z[f"z_{metric}"].to_numpy() for metric in FINGERPRINT_AXES]
    )
    y_random = random_z["z_Q_modularity"].to_numpy()
    coefficients = np.linalg.lstsq(x_random, y_random, rcond=None)[0]
    random_residual = y_random - x_random @ coefficients
    residual_sd = random_residual.std(ddof=len(coefficients))

    x_ga = np.column_stack(
        [np.ones(len(ga_z))]
        + [ga_z[f"z_{metric}"].to_numpy() for metric in FINGERPRINT_AXES]
    )
    ga_residual_z = (
        ga_z["z_Q_modularity"].to_numpy() - x_ga @ coefficients
    ) / residual_sd
    observations.loc[ga_mask, "Q_incremental_residual_z"] = ga_residual_z

    q_random = random.Q_modularity.to_numpy()
    q_ga = ga.Q_modularity.to_numpy()
    lower, upper = np.percentile(q_random, [2.5, 97.5])
    separation = (q_ga.mean() - q_random.mean()) / q_random.std(ddof=1)
    outside = np.mean((q_ga < lower) | (q_ga > upper))
    p_value = mannwhitneyu(q_ga, q_random, alternative="two-sided").pvalue

    joined = observations.loc[env_mask & observations.group.isin(["ga", "random"])]
    summary_rows.append(
        {
            "env": env_id,
            "n_ga": len(ga),
            "n_random": len(random),
            "Q_ga_median": np.median(q_ga),
            "Q_random_median": np.median(q_random),
            "Q_cartesian": cartesian.Q_modularity.iloc[0],
            "Q_ga_vs_random_sd": separation,
            "Q_ga_outside_random_95": outside,
            "Q_mannwhitney_p": p_value,
            "Q_seed_sd_ga_median": ga.Q_seed_sd.median(),
            "Q_partition_ari_ga_median": ga.Q_partition_ari_median.median(),
            "rho_Q_coverage": spearmanr(
                joined.Q_modularity, joined.fp_largest_basin_fraction
            ).statistic,
            "rho_Q_n_basins": spearmanr(
                joined.Q_modularity, joined.fp_n_basins
            ).statistic,
            "rho_Q_cycle_basin_ratio": spearmanr(
                joined.Q_modularity, joined.fp_cycle_basin_ratio
            ).statistic,
            "Q_incremental_residual_ga_median": float(np.median(ga_residual_z)),
        }
    )

summary = pd.DataFrame(summary_rows)
observations.to_parquet(ARTIFACT_DIR / "q-specialisation-observations.parquet", index=False)
summary.to_csv(ARTIFACT_DIR / "q-specialisation-summary.csv", index=False)
print(summary.round(3).to_string(index=False))


# %% [markdown]
# ## Consolidated diagnostic figure

# %%
fig, axes = plt.subplots(1, 3, figsize=(18.4, 5.2), dpi=150)
rng = np.random.default_rng(17)

# (a) Raw Q against the environment-matched random null.
for env_index, env_id in enumerate(ENVS):
    random = observations[(observations.env == env_id) & observations.group.eq("random")]
    ga = observations[(observations.env == env_id) & observations.group.eq("ga")]
    cartesian = observations[(observations.env == env_id) & observations.group.eq("cartesian")]
    violin = axes[0].violinplot(
        [random.Q_modularity], positions=[env_index], widths=0.7, showextrema=False
    )
    for body in violin["bodies"]:
        body.set_facecolor("#c9c9c9")
        body.set_edgecolor("none")
        body.set_alpha(0.9)
    axes[0].scatter(
        env_index + rng.uniform(-0.11, 0.11, len(ga)),
        ga.Q_modularity,
        s=34,
        color=ENV_COLOURS[env_id],
        edgecolor="black",
        linewidth=0.4,
        zorder=4,
    )
    axes[0].scatter(
        env_index,
        cartesian.Q_modularity.iloc[0],
        marker="D",
        s=60,
        color="black",
        facecolor="white",
        linewidth=1.1,
        zorder=5,
    )
axes[0].set_xticks(range(len(ENVS)), [DISPLAY_NAMES[e] for e in ENVS], rotation=25, ha="right")
axes[0].set_ylabel("Q modularity")
axes[0].set_title("(a) GA run-bests lie below matched random Q")
axes[0].grid(axis="y", linestyle=":", alpha=0.35)

# (b) Both axes in each environment's random-null standard deviations.
random = observations[observations.group.eq("random")]
axes[1].scatter(
    random.z_fp_largest_basin_fraction,
    random.z_Q_modularity,
    s=9,
    color="grey",
    alpha=0.13,
    rasterized=True,
)
for env_id in ENVS:
    ga = observations[(observations.env == env_id) & observations.group.eq("ga")]
    axes[1].scatter(
        ga.z_fp_largest_basin_fraction,
        ga.z_Q_modularity,
        marker="*",
        s=95,
        color=ENV_COLOURS[env_id],
        edgecolor="black",
        linewidth=0.5,
        label=DISPLAY_NAMES[env_id],
    )
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].set_xlabel("coverage relative to random null (z)")
axes[1].set_ylabel("Q relative to random null (z)")
axes[1].set_title("(b) Homing and cross-label specialisation separate")
axes[1].legend(fontsize=7, ncol=2, loc="lower left")
axes[1].grid(linestyle=":", alpha=0.25)

# (c) Q remaining after the current three fingerprint axes.
for env_index, env_id in enumerate(ENVS):
    ga = observations[(observations.env == env_id) & observations.group.eq("ga")]
    axes[2].scatter(
        env_index + rng.uniform(-0.1, 0.1, len(ga)),
        ga.Q_incremental_residual_z,
        s=35,
        color=ENV_COLOURS[env_id],
        edgecolor="black",
        linewidth=0.4,
    )
    axes[2].plot(
        [env_index - 0.2, env_index + 0.2],
        [ga.Q_incremental_residual_z.median()] * 2,
        color="black",
        linewidth=2,
    )
axes[2].axhline(0, color="black", linewidth=0.8)
axes[2].set_xticks(range(len(ENVS)), [DISPLAY_NAMES[e] for e in ENVS], rotation=25, ha="right")
axes[2].set_ylabel("incremental Q residual (random-null SD)")
axes[2].set_title("(c) Independent signal is concentrated by geometry")
axes[2].grid(axis="y", linestyle=":", alpha=0.35)

fig.suptitle(
    "Basin co-membership modularity: a secondary specialisation diagnostic",
    fontsize=13,
)
fig.tight_layout()
for suffix in ("png", "pdf"):
    fig.savefig(FIG_DIR / f"F-q-specialisation-screen.{suffix}", bbox_inches="tight")
plt.close(fig)
print(f"saved {FIG_DIR / 'F-q-specialisation-screen.png'}")
