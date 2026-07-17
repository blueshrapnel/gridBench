# %% [markdown]
# # Uniform null and generation-zero initialiser footprints
#
# Question (2026-07-16, Karen): the fingerprint figures currently introduce
# 2,000 fresh uniform-random valid twists as a no-selection null.  Is that a
# useful third reference distribution, or would the generation-zero
# distributions of the paper's two actual initialisers carry the comparison
# more directly?
#
# We compare:
#
# 1. **fresh uniform** — every non-wall state-row is an independent uniform
#    draw from all 4! permutations;
# 2. **row-shuffle (IP-00)** — the paper's baseline initialiser;
# 3. **permutation-balanced (IP-05)** — the production initialiser, sampled as
#    complete 96-member populations so its epsilon schedule is preserved.
#
# The frozen gridFour notebook
# `notebooks/action-alignment/action_ordering_epsilon_profiles.ipynb` records
# some development history, but it is not used as the specification here.
# This diagnostic calls the current implementations of the two strategies
# retained by the paper.

# %%
from __future__ import annotations

import json
import random
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

import gridbench
import gridcore
import evolution_core
from evolution_core.initial_population import (
    build_individual_genes,
    build_population_perm_balanced,
)
from gridbench.functional_graph.fingerprint import fingerprint_for_sigma
from gridbench.functional_graph.probe_env import build_goal_free_probe_env


def repository_commit(package) -> tuple[str, str]:
    """Return the editable-install repository name and short commit."""
    root = Path(package.__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return root.name, commit


for package in (gridcore, gridbench, evolution_core):
    name, commit = repository_commit(package)
    print(f"{name} @ {commit}")

GRIDBENCH_ROOT = Path(gridbench.__file__).resolve().parents[2]
HERE = GRIDBENCH_ROOT / "notebooks" / "twist-generation"
FIG_DIR = HERE / "figures"
ARTIFACT_DIR = HERE / "artifacts"
CACHE_DIR = HERE / "_cache"
for directory in (FIG_DIR, ARTIFACT_DIR, CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Twenty-one complete populations gives 2,016 samples, close to the paper's
# 2,000-draw null without breaking the production initialiser's 96-member
# epsilon schedule.
POPULATION_SIZE = 96
N_POPULATIONS = 21
N_SAMPLES = POPULATION_SIZE * N_POPULATIONS
RATIO_ENVIRONMENTS = ("wrap_grid", "helical", "pillar_3")
COVERAGE_ENVIRONMENTS = ("open_grid", "pinwheel", "four_rooms")
ALL_ENVIRONMENTS = RATIO_ENVIRONMENTS + COVERAGE_ENVIRONMENTS
SHAPE = (7, 7)
DETERMINISM = 0.97
SEED = 20260716

GENERATOR_LABELS = {
    "uniform": "fresh uniform",
    "row_shuffle": "row-shuffle",
    "perm_balanced": "permutation-balanced",
}
GENERATOR_COLOURS = {
    "uniform": "#CC6677",
    "row_shuffle": "#4477AA",
    "perm_balanced": "#228833",
}


# %% [markdown]
# ## Generate twists with the current production operators
#
# Wall rows are filled with identity only to make a full `(nS, nA)` sigma;
# fingerprinting excludes them.  Consequently fresh-uniform and row-shuffle
# have exactly the same probability law over every fingerprint-relevant row.
# Their independent clouds provide a finite-sample check of the KDE procedure.

# %%
def nonwall_states(env) -> list[int]:
    walls = {int(s) for s in getattr(env, "walls_flat", [])}
    return [s for s in range(int(env.nS)) if s not in walls]


def genes_to_sigma(genes, env, state_order: list[int]) -> np.ndarray:
    sigma = np.tile(np.arange(int(env.nA)), (int(env.nS), 1))
    sigma[state_order] = np.asarray(genes, dtype=int).reshape(
        len(state_order), int(env.nA)
    )
    return sigma


def generate_uniform(env, state_order: list[int], seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(N_SAMPLES):
        sigma = np.tile(np.arange(int(env.nA)), (int(env.nS), 1))
        sigma[state_order] = np.stack(
            [rng.permutation(int(env.nA)) for _ in state_order]
        )
        out.append(sigma)
    return out


def generate_row_shuffle(
    env, state_order: list[int], seed: int
) -> list[np.ndarray]:
    rng = random.Random(seed)
    out = []
    for _ in range(N_SAMPLES):
        genes = build_individual_genes(
            state_order=state_order,
            n_actions=int(env.nA),
            init_mode="shuffle",
            rng=rng,
        )
        out.append(genes_to_sigma(genes, env, state_order))
    return out


def generate_perm_balanced(
    env, state_order: list[int], seed: int
) -> list[np.ndarray]:
    rng = random.Random(seed)
    out = []
    for _ in range(N_POPULATIONS):
        population, _ = build_population_perm_balanced(
            state_order=state_order,
            n_actions=int(env.nA),
            population_size=POPULATION_SIZE,
            base_init_mode="hybrid_schedule",
            init_schedule="uniform",
            init_derangement_prob=0.5,
            init_derangement_power=1.0,
            dedupe=True,
            rng=rng,
        )
        out.extend(genes_to_sigma(genes, env, state_order) for genes in population)
    assert len(out) == N_SAMPLES
    return out


GENERATORS = {
    "uniform": generate_uniform,
    "row_shuffle": generate_row_shuffle,
    "perm_balanced": generate_perm_balanced,
}


def sample_fingerprints() -> pd.DataFrame:
    """Generate or load all three distributions on the Figure 7 plane."""
    cache = CACHE_DIR / (
        f"initialiser-fingerprints-v2-{SHAPE[0]}x{SHAPE[1]}-"
        f"n{N_SAMPLES}-seed{SEED}.csv.gz"
    )
    if cache.exists():
        print(f"loading {cache}")
        return pd.read_csv(cache)

    records: list[dict[str, object]] = []
    for env_index, env_id in enumerate(ALL_ENVIRONMENTS):
        env = build_goal_free_probe_env(env_id, SHAPE, DETERMINISM)
        state_order = nonwall_states(env)
        for generator_index, (generator, make_sigmas) in enumerate(GENERATORS.items()):
            draw_seed = SEED + 10_000 * env_index + 1_000 * generator_index
            print(f"fingerprinting {env_id}: {generator}", flush=True)
            for sample, sigma in enumerate(make_sigmas(env, state_order, draw_seed)):
                fp = fingerprint_for_sigma(env, sigma)
                records.append(
                    {
                        "env_id": env_id,
                        "generator": generator,
                        "sample": sample,
                        "fp_n_basins": fp["fp_n_basins"],
                        "fp_cycle_basin_ratio": fp["fp_cycle_basin_ratio"],
                        "fp_largest_basin_fraction": fp[
                            "fp_largest_basin_fraction"
                        ],
                    }
                )
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(cache, index=False, compression="gzip")
    print(f"saved {cache}")
    return frame


fingerprints = sample_fingerprints()
fingerprints.groupby(["env_id", "generator"]).size()


# %% [markdown]
# ## What the 95% and 99% lines mean
#
# For each scatter cloud, fit a two-dimensional Gaussian KDE.  Evaluate that
# KDE at every sampled point.  The thick and thin contour levels are the 5th
# and 1st percentiles of those point densities: approximately 95% and 99% of
# the sampled points lie on the higher-density side.  These are smoothed
# empirical footprint boundaries, not 95%/99% confidence intervals and not
# lines at "95% density".  A multimodal footprint may have several closed
# components at the same level.

# %%
def fit_footprint(x: np.ndarray, y: np.ndarray):
    kde = gaussian_kde(np.vstack([x, y]))
    point_density = kde(np.vstack([x, y]))
    levels = {
        99: float(np.percentile(point_density, 1)),
        95: float(np.percentile(point_density, 5)),
    }
    return kde, levels


def common_grid(
    groups: list[pd.DataFrame],
    y_col: str = "fp_cycle_basin_ratio",
):
    x = np.concatenate([group.fp_n_basins.to_numpy() for group in groups])
    y = np.concatenate([group[y_col].to_numpy() for group in groups])
    x_pad = max(0.35, 0.05 * float(np.ptp(x)))
    y_pad = max(0.035, 0.06 * float(np.ptp(y)))
    xg = np.linspace(float(x.min() - x_pad), float(x.max() + x_pad), 180)
    yg = np.linspace(max(0.0, float(y.min() - y_pad)), min(1.02, float(y.max() + y_pad)), 180)
    return np.meshgrid(xg, yg)


def draw_footprint(
    ax,
    frame: pd.DataFrame,
    xx,
    yy,
    colour: str,
    y_col: str = "fp_cycle_basin_ratio",
):
    x = frame.fp_n_basins.to_numpy()
    y = frame[y_col].to_numpy()
    kde, levels = fit_footprint(x, y)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contour(
        xx,
        yy,
        zz,
        levels=[levels[99], levels[95]],
        colors=[colour, colour],
        linewidths=[1.0, 2.0],
        zorder=4,
    )
    return kde, levels


# %% [markdown]
# ## Separate clouds
#
# Rows distinguish the three generators; columns reproduce the three
# environments of the paper's ratio-plane figure.  This view makes clear that
# each pair of contour lines is derived from the scatter immediately beneath
# it.

# %%
all_groups = [
    fingerprints[
        (fingerprints.env_id == env_id)
        & (fingerprints.generator == generator)
    ]
    for env_id in RATIO_ENVIRONMENTS
    for generator in GENERATORS
]
shared_xx, shared_yy = common_grid(all_groups)
shared_xlim = (float(shared_xx.min()), float(shared_xx.max()))
shared_ylim = (float(shared_yy.min()), float(shared_yy.max()))

fig, axes = plt.subplots(
    3,
    3,
    figsize=(13.5, 11.0),
    dpi=150,
    sharex=True,
    sharey=True,
)
for column, env_id in enumerate(RATIO_ENVIRONMENTS):
    groups = [
        fingerprints[(fingerprints.env_id == env_id) & (fingerprints.generator == generator)]
        for generator in GENERATORS
    ]
    for row, (generator, frame) in enumerate(zip(GENERATORS, groups)):
        ax = axes[row, column]
        colour = GENERATOR_COLOURS[generator]
        ax.scatter(
            frame.fp_n_basins,
            frame.fp_cycle_basin_ratio,
            s=5,
            alpha=0.12,
            color=colour,
            rasterized=True,
        )
        draw_footprint(ax, frame, shared_xx, shared_yy, colour)
        ax.set_xlim(*shared_xlim)
        ax.set_ylim(*shared_ylim)
        if row == 0:
            ax.set_title(f"{env_id}  7x7")
        if column == 0:
            ax.set_ylabel(
                f"{GENERATOR_LABELS[generator]}\nmean cycle/basin ratio"
            )
        if row == 2:
            ax.set_xlabel("mean basins per label")
fig.suptitle(
    "Generation-zero twist priors on the fingerprint plane\n"
    "shared axes; thick contour ≈95% of samples; thin contour ≈99%",
    y=1.01,
)
fig.tight_layout()
cloud_path = FIG_DIR / "F-initialiser-fingerprint-clouds.png"
fig.savefig(cloud_path, bbox_inches="tight")
print(f"saved {cloud_path}")
plt.show()


# %% [markdown]
# ## Overlaid footprints
#
# This is the decision view: if the fresh-uniform null is redundant with an
# initialiser, its contours should coincide with that initialiser's contours.

# %%
fig, axes = plt.subplots(
    1,
    3,
    figsize=(14.5, 4.4),
    dpi=150,
    sharex=True,
    sharey=True,
)
for ax, env_id in zip(axes, RATIO_ENVIRONMENTS):
    groups = {
        generator: fingerprints[
            (fingerprints.env_id == env_id)
            & (fingerprints.generator == generator)
        ]
        for generator in GENERATORS
    }
    for generator, frame in groups.items():
        draw_footprint(
            ax,
            frame,
            shared_xx,
            shared_yy,
            GENERATOR_COLOURS[generator],
        )
    ax.set_title(f"{env_id}  7x7")
    ax.set_xlabel("mean basins per label")
    ax.set_xlim(*shared_xlim)
    ax.set_ylim(*shared_ylim)
axes[0].set_ylabel("mean cycle/basin ratio")

generator_handles = [
    Line2D([0], [0], color=GENERATOR_COLOURS[key], lw=2, label=label)
    for key, label in GENERATOR_LABELS.items()
]
level_handles = [
    Line2D([0], [0], color="0.35", lw=2.0, label="≈95% footprint"),
    Line2D([0], [0], color="0.35", lw=1.0, label="≈99% footprint"),
]
fig.legend(
    handles=generator_handles + level_handles,
    loc="upper center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, 1.07),
)
fig.tight_layout()
overlay_path = FIG_DIR / "F-initialiser-footprint-overlay.png"
fig.savefig(overlay_path, bbox_inches="tight")
print(f"saved {overlay_path}")
plt.show()


# %% [markdown]
# ## Figure 7 preview: both initialisation priors
#
# This deliberately bold version of the paper figure overlays the two priors
# actually used in the paper.  The row-shuffle contours retain the role of a
# fully shuffled no-selection reference.  The permutation-balanced contours
# show the structured production starting population.  They should not be
# read as two estimates of the same null.

# %%
from matplotlib import ticker as mticker


EVALUATED_CACHE_ROOT = Path(
    "/media/merlin/grid-twist/data-schema-10/reports/_cache/functional_graph"
)
COHORT_OBJECTIVES = {"decision_information", "free_energy"}


def full_goal_runs(run_names: pd.Series) -> pd.Series:
    """Select K=1 runs, including untagged runs that predate subsampling."""
    return ~(
        run_names.str.contains("gss")
        | run_names.str.contains(r"k0\d\d", regex=True)
    )


def load_evaluated_cohort(env_id: str) -> pd.DataFrame:
    path = (
        EVALUATED_CACHE_ROOT
        / f"env_id={env_id}/shape={SHAPE[0]}x{SHAPE[1]}"
        / f"det={DETERMINISM}/beta=1/fingerprints.parquet"
    )
    frame = pd.read_parquet(
        path,
        columns=[
            "fp_n_basins",
            "fp_cycle_basin_ratio",
            "mean_free",
            "is_run_best",
            "run_type",
            "fitness_objective",
            "run_name",
        ],
    )
    keep = (
        (frame.run_type == "multi")
        & (
            frame.fitness_objective.isin(COHORT_OBJECTIVES)
            | frame.fitness_objective.isna()
        )
        & full_goal_runs(frame.run_name)
    )
    return frame[keep]


def free_energy_limits(frames: list[pd.DataFrame]):
    """Equal-width, topology-centred colour limits used by Figure 7."""
    spans = [
        float(frame.mean_free.quantile(0.995) - frame.mean_free.quantile(0.005))
        for frame in frames
    ]
    width = max(spans)
    return [
        (
            float(frame.mean_free.median()) - width / 2,
            float(frame.mean_free.median()) + width / 2,
        )
        for frame in frames
    ]


def cartesian_anchor(env_id: str) -> tuple[float, float]:
    env = build_goal_free_probe_env(env_id, SHAPE, DETERMINISM)
    identity = np.tile(np.arange(int(env.nA)), (int(env.nS), 1))
    fp = fingerprint_for_sigma(env, identity)
    return fp["fp_n_basins"], fp["fp_cycle_basin_ratio"]


evaluated_frames = [
    load_evaluated_cohort(env_id) for env_id in RATIO_ENVIRONMENTS
]
colour_limits = free_energy_limits(evaluated_frames)

fig, axes = plt.subplots(1, 3, figsize=(18.6, 5.2), dpi=150)
for ax, env_id, evaluated, (vmin, vmax) in zip(
    axes, RATIO_ENVIRONMENTS, evaluated_frames, colour_limits
):
    scatter = ax.scatter(
        evaluated.fp_n_basins,
        evaluated.fp_cycle_basin_ratio,
        c=evaluated.mean_free,
        s=4,
        alpha=0.25,
        cmap="viridis",
        rasterized=True,
        vmin=vmin,
        vmax=vmax,
        label=rf"all evaluated $\sigma$ (n={len(evaluated):,})",
    )

    prior_frames = {
        generator: fingerprints[
            (fingerprints.env_id == env_id)
            & (fingerprints.generator == generator)
        ]
        for generator in ("row_shuffle", "perm_balanced")
    }
    prior_xx, prior_yy = common_grid(list(prior_frames.values()))
    for generator, prior in prior_frames.items():
        colour = GENERATOR_COLOURS[generator]
        draw_footprint(ax, prior, prior_xx, prior_yy, colour)
        ax.plot(
            [],
            [],
            color=colour,
            lw=1.7,
            label=(
                f"{GENERATOR_LABELS[generator]} prior "
                f"(n={len(prior):,}; 95/99%)"
            ),
        )

    run_bests = evaluated[evaluated.is_run_best]
    ax.scatter(
        run_bests.fp_n_basins,
        run_bests.fp_cycle_basin_ratio,
        marker="*",
        s=210,
        facecolors="none",
        edgecolors="crimson",
        linewidths=1.5,
        label=f"GA run-bests (n={len(run_bests)})",
        zorder=5,
    )
    anchor_x, anchor_y = cartesian_anchor(env_id)
    ax.scatter(
        [anchor_x],
        [anchor_y],
        marker="o",
        s=125,
        facecolors="none",
        edgecolors="black",
        linewidths=1.7,
        label="Cartesian identity",
        zorder=6,
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("mean basins per label  (fp_n_basins)")
    ax.set_ylabel("mean cycle/basin ratio  (fp_cycle_basin_ratio)")
    ax.set_title(f"{env_id}  {SHAPE[0]}x{SHAPE[1]}  $\\beta=1$", fontsize=12)
    legend = ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    for handle in legend.legend_handles:
        try:
            handle.set_alpha(1.0)
        except AttributeError:
            pass
    colourbar = fig.colorbar(
        scatter,
        ax=ax,
        label="mean free energy",
        shrink=0.85,
    )
    colourbar.locator = mticker.MaxNLocator(nbins=5)
    colourbar.update_ticks()

fig.tight_layout()
figure7_path = FIG_DIR / "F-figure7-two-initialiser-footprints.png"
fig.savefig(figure7_path, bbox_inches="tight")
print(f"saved {figure7_path}")
plt.show()


# %% [markdown]
# ## Figure 8 preview: matching priors on the coverage plane
#
# Figure 8 carries the same two-prior grammar onto the bounded, handed, and
# partitioned environments.  Only the vertical fingerprint changes, from
# cycle/basin ratio to largest single-label basin coverage.

# %%
def load_evaluated_coverage(env_id: str) -> pd.DataFrame:
    path = (
        EVALUATED_CACHE_ROOT
        / f"env_id={env_id}/shape={SHAPE[0]}x{SHAPE[1]}"
        / f"det={DETERMINISM}/beta=1/fingerprints.parquet"
    )
    frame = pd.read_parquet(
        path,
        columns=[
            "fp_n_basins",
            "fp_largest_basin_fraction",
            "mean_free",
            "is_run_best",
            "run_type",
            "fitness_objective",
            "run_name",
        ],
    )
    keep = (
        (frame.run_type == "multi")
        & (
            frame.fitness_objective.isin(COHORT_OBJECTIVES)
            | frame.fitness_objective.isna()
        )
        & full_goal_runs(frame.run_name)
    )
    return frame[keep]


def cartesian_coverage_anchor(env_id: str) -> tuple[float, float]:
    env = build_goal_free_probe_env(env_id, SHAPE, DETERMINISM)
    identity = np.tile(np.arange(int(env.nA)), (int(env.nS), 1))
    fp = fingerprint_for_sigma(env, identity)
    return fp["fp_n_basins"], fp["fp_largest_basin_fraction"]


coverage_frames = [
    load_evaluated_coverage(env_id) for env_id in COVERAGE_ENVIRONMENTS
]
coverage_colour_limits = free_energy_limits(coverage_frames)
coverage_statistics: list[dict[str, object]] = []

fig, axes = plt.subplots(1, 3, figsize=(18.6, 5.2), dpi=150)
for ax, env_id, evaluated, (vmin, vmax) in zip(
    axes,
    COVERAGE_ENVIRONMENTS,
    coverage_frames,
    coverage_colour_limits,
):
    scatter = ax.scatter(
        evaluated.fp_n_basins,
        evaluated.fp_largest_basin_fraction,
        c=evaluated.mean_free,
        s=4,
        alpha=0.25,
        cmap="viridis",
        rasterized=True,
        vmin=vmin,
        vmax=vmax,
        label=rf"all evaluated $\sigma$ (n={len(evaluated):,})",
    )

    prior_frames = {
        generator: fingerprints[
            (fingerprints.env_id == env_id)
            & (fingerprints.generator == generator)
        ]
        for generator in ("row_shuffle", "perm_balanced")
    }
    prior_xx, prior_yy = common_grid(
        list(prior_frames.values()),
        y_col="fp_largest_basin_fraction",
    )
    prior_fits = {}
    for generator, prior in prior_frames.items():
        colour = GENERATOR_COLOURS[generator]
        prior_fits[generator] = draw_footprint(
            ax,
            prior,
            prior_xx,
            prior_yy,
            colour,
            y_col="fp_largest_basin_fraction",
        )
        ax.plot(
            [],
            [],
            color=colour,
            lw=1.7,
            label=(
                f"{GENERATOR_LABELS[generator]} prior "
                f"(n={len(prior):,}; 95/99%)"
            ),
        )

    run_bests = evaluated[evaluated.is_run_best]
    run_best_median = float(run_bests.fp_largest_basin_fraction.median())
    run_best_positions = np.vstack(
        [
            run_bests.fp_n_basins.to_numpy(),
            run_bests.fp_largest_basin_fraction.to_numpy(),
        ]
    )
    for generator, prior in prior_frames.items():
        coverage = prior.fp_largest_basin_fraction.to_numpy()
        prior_kde, prior_levels = prior_fits[generator]
        run_best_density = prior_kde(run_best_positions)
        coverage_statistics.append(
            {
                "env_id": env_id,
                "generator": generator,
                "n": len(prior),
                "run_best_median": run_best_median,
                "prior_coverage_median": float(np.median(coverage)),
                "prior_coverage_max": float(np.max(coverage)),
                "n_at_or_above_run_best_median": int(
                    np.sum(coverage >= run_best_median)
                ),
                "run_bests_inside_95": int(
                    np.sum(run_best_density >= prior_levels[95])
                ),
                "run_bests_inside_99": int(
                    np.sum(run_best_density >= prior_levels[99])
                ),
            }
        )

    ax.scatter(
        run_bests.fp_n_basins,
        run_bests.fp_largest_basin_fraction,
        marker="*",
        s=210,
        facecolors="none",
        edgecolors="crimson",
        linewidths=1.5,
        label=f"GA run-bests (n={len(run_bests)})",
        zorder=5,
    )
    anchor_x, anchor_y = cartesian_coverage_anchor(env_id)
    ax.scatter(
        [anchor_x],
        [anchor_y],
        marker="o",
        s=125,
        facecolors="none",
        edgecolors="black",
        linewidths=1.7,
        label="Cartesian identity",
        zorder=6,
    )
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("mean basins per label  (fp_n_basins)")
    ax.set_ylabel("coverage  (largest single-label basin fraction)")
    ax.set_title(f"{env_id}  {SHAPE[0]}x{SHAPE[1]}  $\\beta=1$", fontsize=12)
    legend = ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    for handle in legend.legend_handles:
        try:
            handle.set_alpha(1.0)
        except AttributeError:
            pass
    colourbar = fig.colorbar(
        scatter,
        ax=ax,
        label="mean free energy",
        shrink=0.85,
    )
    colourbar.locator = mticker.MaxNLocator(nbins=5)
    colourbar.update_ticks()

fig.tight_layout()
figure8_path = FIG_DIR / "F-figure8-two-initialiser-footprints.png"
fig.savefig(figure8_path, bbox_inches="tight")
print(f"saved {figure8_path}")
plt.show()

coverage_statistics = pd.DataFrame(coverage_statistics)
coverage_statistics_path = ARTIFACT_DIR / "initialiser-coverage-comparison.csv"
coverage_statistics.to_csv(coverage_statistics_path, index=False)
print(coverage_statistics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
print(f"saved {coverage_statistics_path}")


# %% [markdown]
# ## Why do the open-grid run-bests form bands?
#
# This is a run-level audit of the stars in the left panel of Figure 8.  It
# distinguishes the configured generation budget from the first generation at
# which the final run-best fitness appeared.  Runs whose archived objective is
# blank predate the free-energy fitness implementation and are therefore
# classified as decision-information runs.  The seed panel tests exact seed
# reuse; numerical proximity between seed integers has no stochastic meaning.
#
# Coverage is an integer basin size divided by 49, so its horizontal lattice
# has spacing 1/49.  Mean basins averages four integer label counts, giving the
# horizontal coordinate spacing 1/4.  The diagnostic asks what distributes
# runs across this lattice; it does not interpret the lattice itself as an
# evolutionary trajectory.

# %%
RUN_DATA_ROOT = Path("/media/merlin/grid-twist/data-schema-10/multi")


def first_generation_with_final_fitness(
    log_path: Path, final_fitness: float
) -> tuple[int, int]:
    """Return the first matching generation and the largest logged generation."""
    rows: list[tuple[int, float]] = []
    for line in log_path.read_text(errors="replace").splitlines():
        match = re.match(
            r"^\s*(\d+)\s+\d+\s+([+\-.\deE]+)\s+",
            line,
        )
        if match:
            rows.append((int(match.group(1)), float(match.group(2))))
    if not rows:
        raise ValueError(f"no generation rows found in {log_path}")
    matching = [
        generation
        for generation, fitness in rows
        if np.isclose(fitness, final_fitness, rtol=2e-6, atol=6e-6)
    ]
    if not matching:
        raise ValueError(
            f"final fitness {final_fitness} not found in {log_path}"
        )
    return min(matching), max(generation for generation, _ in rows)


def open_grid_run_metadata(run_bests: pd.DataFrame) -> pd.DataFrame:
    run_directories = {
        directory.name.removeprefix("run_name="): directory
        for directory in RUN_DATA_ROOT.glob(
            "init_method=*/env_id=open_grid/shape=7x7/beta=1/"
            "det=0.97/run_name=*"
        )
        if directory.is_dir()
    }
    rows: list[dict[str, object]] = []
    for run in run_bests.itertuples(index=False):
        run_directory = run_directories[run.run_name]
        summary_path = next(run_directory.glob("*summary.json"))
        summary = json.loads(summary_path.read_text())
        config = summary["config"]
        log_path = next(run_directory.glob("*.log"))
        winning_generation, logged_generation_max = (
            first_generation_with_final_fitness(
                log_path,
                float(summary["best_fitness"]),
            )
        )
        objective = run.fitness_objective
        if pd.isna(objective):
            objective = "decision_information"
        rows.append(
            {
                "run_name": run.run_name,
                "fp_n_basins": run.fp_n_basins,
                "coverage": run.fp_largest_basin_fraction,
                "seed": int(config["seed"]),
                "generation_budget": int(config["generations"]),
                "winning_generation": winning_generation,
                "logged_generation_max": logged_generation_max,
                "fitness_objective": objective,
                "init_mode": config["init_mode"],
                "source_run_id": config.get("source_run_id"),
            }
        )
    frame = pd.DataFrame(rows)
    frame["seed_reused_in_cohort"] = frame.seed.duplicated(keep=False)
    frame["appears_to_be_extension"] = (
        frame.source_run_id.notna()
        & frame.source_run_id.ne(frame.run_name)
    )
    return frame.sort_values(["coverage", "fp_n_basins", "seed"])


open_evaluated = coverage_frames[0]
open_run_bests = open_evaluated[open_evaluated.is_run_best].copy()
assert open_run_bests.run_name.is_unique
open_metadata = open_grid_run_metadata(open_run_bests)
open_metadata_path = ARTIFACT_DIR / "open-grid-run-best-metadata.csv"
open_metadata.to_csv(open_metadata_path, index=False)

print("open_grid run-best objective counts")
print(open_metadata.fitness_objective.value_counts().to_string())
print("\nopen_grid run-best generation budgets")
print(open_metadata.generation_budget.value_counts().sort_index().to_string())
print("\ncoverage by generation budget")
print(
    open_metadata.groupby("generation_budget").coverage.agg(
        ["count", "min", "median", "max"]
    ).to_string(float_format=lambda value: f"{value:.3f}")
)
print("\ncoverage by fitness objective")
print(
    open_metadata.groupby("fitness_objective").coverage.agg(
        ["count", "min", "median", "max"]
    ).to_string(float_format=lambda value: f"{value:.3f}")
)
print("\ncoverage by initialisation strategy")
print(
    open_metadata.groupby("init_mode").coverage.agg(
        ["count", "min", "median", "max"]
    ).to_string(float_format=lambda value: f"{value:.3f}")
)
print(
    "\nSpearman(coverage, generation budget) = "
    f"{open_metadata[['coverage', 'generation_budget']].corr(method='spearman').iloc[0, 1]:.3f}"
)
print(
    "Spearman(coverage, winning generation) = "
    f"{open_metadata[['coverage', 'winning_generation']].corr(method='spearman').iloc[0, 1]:.3f}"
)
print(
    "exact seeds reused = "
    f"{open_metadata.loc[open_metadata.seed_reused_in_cohort, 'seed'].nunique()} pairs"
)
print(
    "apparent continuation runs = "
    f"{int(open_metadata.appears_to_be_extension.sum())}"
)
print(f"saved {open_metadata_path}")


# %%
def setup_open_diagnostic_axis(ax) -> None:
    ax.scatter(
        open_evaluated.fp_n_basins,
        open_evaluated.fp_largest_basin_fraction,
        s=3,
        color="#b8b8b8",
        alpha=0.11,
        rasterized=True,
        zorder=0,
    )
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax.set_xlim(7.25, 9.05)
    ax.set_ylim(0.35, 1.01)
    ax.set_xlabel("mean basins per label")
    ax.set_ylabel("coverage")


fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.0), dpi=150, sharex=True, sharey=True)
for ax in axes.flat:
    setup_open_diagnostic_axis(ax)

budget_colours = {
    40: "#88CCEE",
    100: "#44AA99",
    200: "#DDCC77",
    500: "#CC6677",
}
for budget, group in open_metadata.groupby("generation_budget"):
    axes[0, 0].scatter(
        group.fp_n_basins,
        group.coverage,
        marker="*",
        s=190,
        color=budget_colours[budget],
        edgecolors="#333333",
        linewidths=0.65,
        label=f"{budget} generations (n={len(group)})",
        zorder=4,
    )
axes[0, 0].set_title("A  Recorded generation budget")
axes[0, 0].legend(fontsize=8, loc="lower left")

winning_scatter = axes[0, 1].scatter(
    open_metadata.fp_n_basins,
    open_metadata.coverage,
    marker="*",
    s=190,
    c=open_metadata.winning_generation,
    cmap="plasma",
    vmin=0,
    vmax=500,
    edgecolors="#333333",
    linewidths=0.65,
    zorder=4,
)
axes[0, 1].set_title("B  First generation containing the final winner")
fig.colorbar(
    winning_scatter,
    ax=axes[0, 1],
    label="generation",
    shrink=0.84,
)

objective_colours = {
    "decision_information": "#4477AA",
    "free_energy": "#EE7733",
}
objective_labels = {
    "decision_information": "decision information",
    "free_energy": "free energy",
}
for objective, group in open_metadata.groupby("fitness_objective"):
    axes[1, 0].scatter(
        group.fp_n_basins,
        group.coverage,
        marker="*",
        s=190,
        color=objective_colours[objective],
        edgecolors="#333333",
        linewidths=0.65,
        label=f"{objective_labels[objective]} (n={len(group)})",
        zorder=4,
    )
shuffle_runs = open_metadata[open_metadata.init_mode == "shuffle"]
axes[1, 0].scatter(
    shuffle_runs.fp_n_basins,
    shuffle_runs.coverage,
    marker="o",
    s=255,
    facecolors="none",
    edgecolors="black",
    linewidths=1.15,
    label=f"row-shuffle initialisation (n={len(shuffle_runs)})",
    zorder=5,
)
axes[1, 0].set_title("C  Fitness objective; rings mark row-shuffle")
axes[1, 0].legend(fontsize=8, loc="lower left")

seed_ax = axes[1, 1]
singletons = open_metadata[~open_metadata.seed_reused_in_cohort]
seed_ax.scatter(
    singletons.fp_n_basins,
    singletons.coverage,
    marker="*",
    s=160,
    facecolors="white",
    edgecolors="#777777",
    linewidths=0.8,
    label=f"unique seed (n={len(singletons)})",
    zorder=3,
)
paired = open_metadata[open_metadata.seed_reused_in_cohort]
seed_colours = plt.get_cmap("tab10")(
    np.linspace(0, 1, paired.seed.nunique())
)
for colour, (seed, group) in zip(seed_colours, paired.groupby("seed")):
    seed_ax.plot(
        group.fp_n_basins,
        group.coverage,
        color=colour,
        alpha=0.72,
        lw=1.2,
        zorder=2,
    )
    seed_ax.scatter(
        group.fp_n_basins,
        group.coverage,
        marker="*",
        s=190,
        color=colour,
        edgecolors="#333333",
        linewidths=0.65,
        zorder=4,
    )
seed_ax.plot(
    [],
    [],
    color="#555555",
    lw=1.2,
    marker="*",
    markersize=10,
    label=f"exact reused seed ({paired.seed.nunique()} pairs)",
)
seed_ax.set_title("D  Exact seed reuse; paired outcomes joined")
seed_ax.legend(fontsize=8, loc="lower left")

fig.suptitle(
    "open_grid run-bests: the bands are the 7x7 fingerprint lattice",
    fontsize=14,
)
fig.tight_layout()
diagnostic_path = FIG_DIR / "F-open-grid-run-best-stratification-diagnostic.png"
fig.savefig(diagnostic_path, bbox_inches="tight")
print(f"saved {diagnostic_path}")
plt.show()


# %% [markdown]
# ## Numerical overlap with the fresh-uniform footprint
#
# The final table evaluates every other cloud under the fresh-uniform KDE.
# `inside_uniform_95` and `inside_uniform_99` are the fractions lying on the
# high-density side of the corresponding uniform contour.  If two generators
# have the same law, these should be near 0.95 and 0.99 up to Monte Carlo and
# KDE error.

# %%
summary = []
for env_id in RATIO_ENVIRONMENTS:
    uniform = fingerprints[
        (fingerprints.env_id == env_id) & (fingerprints.generator == "uniform")
    ]
    uniform_kde, uniform_levels = fit_footprint(
        uniform.fp_n_basins.to_numpy(),
        uniform.fp_cycle_basin_ratio.to_numpy(),
    )
    for generator in GENERATORS:
        frame = fingerprints[
            (fingerprints.env_id == env_id)
            & (fingerprints.generator == generator)
        ]
        positions = np.vstack(
            [frame.fp_n_basins.to_numpy(), frame.fp_cycle_basin_ratio.to_numpy()]
        )
        density_under_uniform = uniform_kde(positions)
        summary.append(
            {
                "env_id": env_id,
                "generator": generator,
                "n": len(frame),
                "median_n_basins": frame.fp_n_basins.median(),
                "median_cycle_basin_ratio": frame.fp_cycle_basin_ratio.median(),
                "inside_uniform_95": np.mean(
                    density_under_uniform >= uniform_levels[95]
                ),
                "inside_uniform_99": np.mean(
                    density_under_uniform >= uniform_levels[99]
                ),
            }
        )

summary = pd.DataFrame(summary)
summary_path = ARTIFACT_DIR / "initialiser-footprint-overlap.csv"
summary.to_csv(summary_path, index=False)
print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
print(f"saved {summary_path}")
