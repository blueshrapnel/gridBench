# %% [markdown]
# # Four-rooms single-label reach exemplar
#
# Rebuild the twists-home-vectors paper's per-state single-label reach
# figure (Figure 8 at the time of migration).  This is the current gridBench
# replacement for the legacy gridFour notebook
# `notebooks/attractor-fingerprint-probe/43-journey-and-reach.py`.
#
# The two formerly implicit choices are made explicit here:
#
# 1. the evolved exemplar is the maximum-`fp_mean_rho` run-best in the same
#    pooled, full-goal DI/free-energy cohort used by the reach-by-class plot;
# 2. a home label maximises largest-basin coverage, then maximum per-state
#    reach.  This tie-break selects the random twist's longer right-hand path
#    instead of the first tied label in numerical order.

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gridbench
from gridbench.functional_graph.decomposition import (
    decompose,
    deterministic_successor,
)
from gridbench.functional_graph.fingerprint import fingerprint_for_sigma
from gridbench.functional_graph.probe_env import build_goal_free_probe_env


GRIDBENCH_ROOT = Path(gridbench.__file__).resolve().parents[2]
HERE = GRIDBENCH_ROOT / "notebooks" / "single-label-reach"
FIGURE_DIR = HERE / "figures"
ARTIFACT_DIR = HERE / "artifacts"
for directory in (FIGURE_DIR, ARTIFACT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path("/media/merlin/grid-twist/data-schema-10/multi")
FINGERPRINT_PATH = Path(
    "/media/merlin/grid-twist/data-schema-10/reports/_cache/functional_graph/"
    "env_id=four_rooms/shape=7x7/det=0.97/beta=1/fingerprints.parquet"
)

ENV_ID = "four_rooms"
SHAPE = (7, 7)
DETERMINISM = 0.97
RANDOM_SEED = 11
COHORT_OBJECTIVES = {"decision_information", "free_energy"}
ACTION_LABELS = ("N", "E", "S", "W")
COLOURS = {
    "Cartesian": "#4477AA",
    "Random twist": "#EE6677",
    "Highest-reach run-best": "#228833",
}


# %% [markdown]
# ## Select the evolved exemplar from the current run-best cohort
#
# Untagged objective values are legacy decision-information runs: they predate
# free energy being available as a fitness function.  Goal-subsampled runs are
# excluded.  The figure is about reach, so the exemplar is selected by maximum
# mean single-label reach, with diameter and coverage as deterministic
# tie-breaks.  This is not a claim that it has the lowest free energy.

# %%
def full_goal_runs(run_names: pd.Series) -> pd.Series:
    return ~(
        run_names.str.contains("gss", na=False)
        | run_names.str.contains(r"k0\d\d", regex=True, na=False)
    )


def load_run_best_cohort() -> pd.DataFrame:
    columns = [
        "sigma_hash",
        "run_name",
        "is_run_best",
        "run_type",
        "fitness_objective",
        "fp_largest_basin_fraction",
        "fp_mean_rho",
        "fp_diameter",
        "fp_n_basins",
        "fp_cycle_basin_ratio",
        "mean_free",
        "mean_info",
    ]
    frame = pd.read_parquet(FINGERPRINT_PATH, columns=columns)
    keep = (
        frame.is_run_best
        & frame.run_type.eq("multi")
        & (
            frame.fitness_objective.isin(COHORT_OBJECTIVES)
            | frame.fitness_objective.isna()
        )
        & full_goal_runs(frame.run_name)
    )
    cohort = frame.loc[keep].copy()
    cohort["fitness_objective_recorded"] = cohort.fitness_objective
    cohort["fitness_objective"] = cohort.fitness_objective.fillna(
        "decision_information_legacy"
    )
    cohort = cohort.sort_values(
        ["fp_mean_rho", "fp_diameter", "fp_largest_basin_fraction"],
        ascending=False,
    ).reset_index(drop=True)
    cohort.insert(0, "reach_rank", np.arange(1, len(cohort) + 1))
    return cohort


def run_directory(run_name: str) -> Path:
    candidates = list(
        DATA_ROOT.glob(
            "init_method=*/env_id=four_rooms/shape=7x7/beta=1/"
            f"det=0.97/run_name={run_name}"
        )
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one directory for {run_name!r}, found {len(candidates)}"
        )
    return candidates[0]


cohort = load_run_best_cohort()
cohort_path = ARTIFACT_DIR / "four-rooms-full-goal-run-best-reach-audit.csv"
cohort.to_csv(cohort_path, index=False)
selected_run = cohort.iloc[0]
selected_directory = run_directory(str(selected_run.run_name))
selected_sigma_path = next(selected_directory.glob("*multi-all.sigma.npy"))
evolved_sigma = np.load(selected_sigma_path)

print(f"eligible full-goal run-bests: {len(cohort)}")
print("selected highest-reach run-best")
print(
    selected_run[
        [
            "run_name",
            "fitness_objective",
            "fp_mean_rho",
            "fp_diameter",
            "fp_largest_basin_fraction",
            "mean_free",
        ]
    ].to_string()
)
print(f"sigma: {selected_sigma_path}")
print(f"saved {cohort_path}")


# %% [markdown]
# ## Decompose the three twists label by label

# %%
@dataclass(frozen=True)
class LabelReach:
    label: int
    coverage: float
    mean_rho: float
    max_rho: int
    successor: np.ndarray
    graph: object


env = build_goal_free_probe_env(ENV_ID, SHAPE, DETERMINISM)
n_states = int(env.nS)
n_actions = int(env.nA)
width = SHAPE[1]
walls = {int(state) for state in (getattr(env, "walls_flat", []) or [])}
nonwall_states = [state for state in range(n_states) if state not in walls]
physical_successors = np.stack(
    [deterministic_successor(env, action) for action in range(n_actions)]
)


def cartesian_sigma() -> np.ndarray:
    return np.tile(np.arange(n_actions, dtype=int), (n_states, 1))


def random_sigma(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(n_actions) for _ in range(n_states)])


def per_label_reach(sigma: np.ndarray) -> list[LabelReach]:
    inverse = np.argsort(sigma, axis=1)
    state_index = np.arange(n_states)
    results = []
    for label in range(n_actions):
        successor = physical_successors[inverse[:, label], state_index]
        graph = decompose(successor, walls=sorted(walls))
        basin_sizes = np.asarray(graph.basin_sizes, dtype=float)
        coverage = (
            float(basin_sizes.max() / basin_sizes.sum())
            if basin_sizes.size and basin_sizes.sum() > 0
            else 0.0
        )
        rho = np.asarray(graph.rho, dtype=int)
        results.append(
            LabelReach(
                label=label,
                coverage=coverage,
                mean_rho=float(rho[nonwall_states].mean()),
                max_rho=int(rho[nonwall_states].max()),
                successor=successor,
                graph=graph,
            )
        )
    return results


def home_label(sigma: np.ndarray) -> LabelReach:
    """Choose coverage first, then the longest and broadest reach."""
    return max(
        per_label_reach(sigma),
        key=lambda result: (
            result.coverage,
            result.max_rho,
            result.mean_rho,
            -result.label,
        ),
    )


def farthest_start(result: LabelReach) -> int:
    """Choose a max-rho state; prefer the rightmost, then uppermost tie."""
    rho = np.asarray(result.graph.rho, dtype=int)
    return max(
        nonwall_states,
        key=lambda state: (
            int(rho[state]),
            state % width,
            -(state // width),
        ),
    )


def walk(successor: np.ndarray, start: int) -> list[int]:
    path = [int(start)]
    seen = {int(start)}
    state = int(start)
    for _ in range(n_states + 2):
        state = int(successor[state])
        path.append(state)
        if state in seen:
            break
        seen.add(state)
    return path


cases = [
    ("Cartesian", cartesian_sigma()),
    ("Random twist", random_sigma(RANDOM_SEED)),
    ("Highest-reach run-best", evolved_sigma),
]

case_rows = []
decomposed_cases = []
for name, sigma in cases:
    fingerprint = fingerprint_for_sigma(env, sigma)
    result = home_label(sigma)
    start = farthest_start(result)
    path = walk(result.successor, start)
    decomposed_cases.append((name, result, path))
    case_rows.append(
        {
            "case": name,
            "label": ACTION_LABELS[result.label],
            "label_index": result.label,
            "largest_basin_coverage": result.coverage,
            "home_label_mean_rho": result.mean_rho,
            "home_label_max_rho": result.max_rho,
            "walk_moves_to_first_repeat": len(path) - 1,
            "start_state": start,
            "start_row": start // width,
            "start_col": start % width,
            "fp_mean_rho": fingerprint["fp_mean_rho"],
            "fp_diameter": fingerprint["fp_diameter"],
        }
    )

case_summary = pd.DataFrame(case_rows)
summary_path = ARTIFACT_DIR / "four-rooms-reach-exemplar-summary.csv"
case_summary.to_csv(summary_path, index=False)
print("\ncase summary")
print(case_summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
print(f"saved {summary_path}")

random_label_table = pd.DataFrame(
    [
        {
            "label": ACTION_LABELS[result.label],
            "label_index": result.label,
            "largest_basin_coverage": result.coverage,
            "mean_rho": result.mean_rho,
            "max_rho": result.max_rho,
        }
        for result in per_label_reach(cases[1][1])
    ]
)
print("\nrandom-twist label tie audit")
print(random_label_table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


# %% [markdown]
# ## Render the paper figure

# %%
fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0), dpi=150)
for ax, (name, result, path) in zip(axes, decomposed_cases):
    rho = np.asarray(result.graph.rho, dtype=float)
    grid = np.full(SHAPE, np.nan)
    for state in nonwall_states:
        grid[state // width, state % width] = rho[state]

    image = ax.imshow(
        grid,
        cmap="YlOrRd",
        origin="upper",
        vmin=0,
        vmax=max(float(np.nanmax(grid)), 1.0),
    )
    for state in walls:
        ax.add_patch(
            plt.Rectangle(
                (state % width - 0.5, state // width - 0.5),
                1,
                1,
                color="0.35",
            )
        )

    ys = [state // width for state in path]
    xs = [state % width for state in path]
    ax.plot(xs, ys, "-", color="black", lw=1.8, zorder=5)
    ax.scatter([xs[0]], [ys[0]], marker="o", s=70, color="black", zorder=6)
    ax.scatter([xs[-1]], [ys[-1]], marker="*", s=140, color="black", zorder=6)

    title_name = name
    if name == "Highest-reach run-best":
        title_name += f" (coverage {result.coverage:.3f})"
    ax.set_title(
        f"{title_name}\nhome label {ACTION_LABELS[result.label]}: "
        f"mean reach {result.mean_rho:.2f}, longest walk {result.max_rho}",
        fontsize=9.5,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)

fig.suptitle(
    "Four-rooms 7×7: per-state single-label reach and a longest label-walk.  "
    "● start  ★ first repeated cycle state",
    fontsize=11,
    y=1.02,
)
fig.tight_layout()
figure_path = FIGURE_DIR / "F-reach-trajectories.png"
fig.savefig(figure_path, bbox_inches="tight")
print(f"saved {figure_path}")
plt.show()

