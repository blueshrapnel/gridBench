# %% [markdown]
# # Dominant-basin atlas across the environment palette
#
# Rebuild the twists-home-vectors dominant-basin atlas (Figure 12 at the
# time of migration) with the current gridCore/gridBench implementation.
# The atlas asks a deliberately narrow question: for the label with the
# broadest basin in a twist, which states drain to the same terminal cycle?
#
# It compares Cartesian labelling, one paired row-shuffle control, and one
# evolved run-best exemplar in six 7x7 environments.  The random control is
# paired: seed 3011 generates the same 49x4 relabelling in every environment,
# so changes across its row come from topology rather than a new random draw.
# The wrap_grid, pinwheel, and four_rooms evolved exemplars are exactly the
# twists opened into all four labels by the paper's following anatomy figure.

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

import gridbench
from gridbench.functional_graph.decomposition import (
    decompose,
    deterministic_successor,
)
from gridbench.functional_graph.probe_env import build_goal_free_probe_env


GRIDBENCH_ROOT = Path(gridbench.__file__).resolve().parents[2]
HERE = GRIDBENCH_ROOT / "notebooks" / "basin-typology"
FIGURE_DIR = HERE / "figures"
ARTIFACT_DIR = HERE / "artifacts"
for directory in (FIGURE_DIR, ARTIFACT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path("/media/merlin/grid-twist/data-schema-10/multi")
FINGERPRINT_ROOT = Path(
    "/media/merlin/grid-twist/data-schema-10/reports/_cache/functional_graph"
)

SHAPE = (7, 7)
DETERMINISM = 0.97
RANDOM_SEED = 3011
ACTION_LABELS = ("N", "E", "S", "W")
ENVIRONMENT_ORDER = (
    "wrap_grid",
    "helical",
    "open_grid",
    "pinwheel",
    "pillar_3",
    "four_rooms",
)
CLASS_ORDER = ("Cartesian", "Paired random", "Evolved exemplar")

# Run-best exemplars are structural illustrations, not maxima selected on
# coverage.  The three marked ``anatomy match`` are the exact twists used by
# the next paper figure, allowing the dominant-label projection here to be
# opened into its full four-label switchboard there.
EVOLVED_RUNS = {
    "wrap_grid": (
        "g200-pop-96-perm-bal-17-06-b1-free-ga-wrap-grid-fediv-sp3011-08",
        "anatomy match",
    ),
    "helical": (
        "g200-pop-96-perm-bal-24-05-b1-free-ga-corkscrew-7x7-k100-sp3011-12",
        "total-basin exemplar",
    ),
    "open_grid": (
        "g200-pop-96-perm-bal-24-03-b1-free-ga-open-grid-sp3011-10",
        "near-total exemplar",
    ),
    "pinwheel": (
        "g200-pop-96-perm-bal-06-06-b1-free-ga-pinwheel-7x7-k100-sp3011-16",
        "anatomy match",
    ),
    "pillar_3": (
        "g200-pop-96-perm-bal-20-03-b1-free-ga-pillar-3-7x7-sp3011-08",
        "lowest-free-energy full-goal run-best in the cached cohort",
    ),
    "four_rooms": (
        "g500-pop-96-perm-bal-07-06-b1-free-ga-four-rooms-7x7-k100-sp3011-06-survey",
        "anatomy match",
    ),
}


# %% [markdown]
# ## Load and audit the evolved exemplars
#
# Every exemplar must be a full-goal, free-energy run-best.  The audit table
# keeps the plotted sigma tied to the same fingerprint cache used by the
# population figures in the paper.

# %%
def run_directory(env_id: str, run_name: str) -> Path:
    candidates = list(
        DATA_ROOT.glob(
            f"init_method=*/env_id={env_id}/shape=7x7/beta=1/det=0.97/"
            f"run_name={run_name}"
        )
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one directory for {env_id}/{run_name}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def fingerprint_row(env_id: str, run_name: str) -> pd.Series:
    path = (
        FINGERPRINT_ROOT
        / f"env_id={env_id}/shape=7x7/det=0.97/beta=1/fingerprints.parquet"
    )
    frame = pd.read_parquet(path)
    rows = frame.loc[
        frame.run_name.eq(run_name)
        & frame.is_run_best
        & frame.run_type.eq("multi")
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected one cached run-best row for {env_id}/{run_name}, "
            f"found {len(rows)}"
        )
    row = rows.iloc[0]
    if row.fitness_objective != "free_energy":
        raise ValueError(f"{run_name} is not a free-energy run-best")
    if "gss" in run_name or any(f"k0{i:02d}" in run_name for i in range(100)):
        raise ValueError(f"{run_name} is not a full-goal run")
    return row


evolved_sigmas: dict[str, np.ndarray] = {}
audit_rows: list[dict[str, object]] = []
for environment, (run_name, selection_note) in EVOLVED_RUNS.items():
    directory = run_directory(environment, run_name)
    sigma_path = next(directory.glob("*multi-all.sigma.npy"))
    sigma = np.asarray(np.load(sigma_path), dtype=int)
    if sigma.shape != (SHAPE[0] * SHAPE[1], len(ACTION_LABELS)):
        raise ValueError(f"unexpected sigma shape at {sigma_path}: {sigma.shape}")
    evolved_sigmas[environment] = sigma

    cached = fingerprint_row(environment, run_name)
    audit_rows.append(
        {
            "environment": environment,
            "run_name": run_name,
            "selection_note": selection_note,
            "sigma_hash": cached.sigma_hash,
            "mean_free": cached.mean_free,
            "fingerprint_coverage": cached.fp_largest_basin_fraction,
            "fingerprint_mean_basins": cached.fp_n_basins,
            "fingerprint_cycle_basin_ratio": cached.fp_cycle_basin_ratio,
            "sigma_path": str(sigma_path),
        }
    )

audit = pd.DataFrame(audit_rows)
audit_path = ARTIFACT_DIR / "dominant-basin-atlas-evolved-exemplars.csv"
audit.to_csv(audit_path, index=False)
print(audit.to_string(index=False))
print(f"saved {audit_path}")


# %% [markdown]
# ## Decompose the dominant-basin label

# %%
@dataclass(frozen=True)
class LabelBasin:
    label: int
    graph: object
    dominant_basin: int
    coverage: float
    cycle: tuple[int, ...]


def walls_of(env) -> set[int]:
    values = getattr(env, "walls_flat", None)
    if values is None:
        return set()
    return {int(state) for state in np.asarray(values).ravel()}


def cartesian_sigma(n_states: int, n_actions: int) -> np.ndarray:
    return np.tile(np.arange(n_actions, dtype=int), (n_states, 1))


def paired_random_sigma(n_states: int, n_actions: int) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    return np.stack([rng.permutation(n_actions) for _ in range(n_states)])


def label_basins(env, sigma: np.ndarray) -> list[LabelBasin]:
    n_states, n_actions = int(env.nS), int(env.nA)
    walls = walls_of(env)
    physical_successors = np.stack(
        [deterministic_successor(env, action) for action in range(n_actions)]
    )
    inverse = np.argsort(sigma, axis=1)
    state_index = np.arange(n_states)
    results = []
    for label in range(n_actions):
        successor = physical_successors[inverse[:, label], state_index]
        graph = decompose(successor, walls=walls)
        sizes = np.asarray(graph.basin_sizes, dtype=int)
        dominant = int(sizes.argmax())
        cycle = tuple(int(state) for state in graph.cycles[dominant])
        results.append(
            LabelBasin(
                label=label,
                graph=graph,
                dominant_basin=dominant,
                coverage=float(sizes[dominant] / sizes.sum()),
                cycle=cycle,
            )
        )
    return results


def dominant_label(env, sigma: np.ndarray) -> LabelBasin:
    # Coverage is primary.  When labels tie, prefer the more compact terminal
    # cycle, then fewer basins, then the familiar label order.
    return max(
        label_basins(env, sigma),
        key=lambda result: (
            result.coverage,
            -len(result.cycle),
            -result.graph.n_basins,
            -result.label,
        ),
    )


environment_data: dict[str, dict[str, object]] = {}
reference_random = None
for environment in ENVIRONMENT_ORDER:
    env = build_goal_free_probe_env(environment, SHAPE, DETERMINISM)
    random = paired_random_sigma(int(env.nS), int(env.nA))
    if reference_random is None:
        reference_random = random
    else:
        assert np.array_equal(random, reference_random)
    environment_data[environment] = {
        "env": env,
        "walls": walls_of(env),
        "sigmas": {
            "Cartesian": cartesian_sigma(int(env.nS), int(env.nA)),
            "Paired random": random,
            "Evolved exemplar": evolved_sigmas[environment],
        },
    }


# %% [markdown]
# ## Six-environment atlas
#
# Blue is the selected label's largest basin.  Other colours are its remaining
# basins, dark cells are walls, and yellow outlines the terminal cycle of the
# blue basin.  The annotations report label, coverage, basin count, and the
# selected cycle length; the paper caption supplies their interpretation.

# %%
OTHER_BASIN_COLOURS = [
    np.asarray(to_rgba(colour))
    for colour in (
        list(plt.colormaps["Pastel1"].colors)
        + list(plt.colormaps["Pastel2"].colors)
    )
]
DOMINANT_COLOUR = np.array([0.10, 0.45, 0.78, 0.88])
WALL_COLOUR = np.array([0.12, 0.12, 0.12, 1.00])
EMPTY_COLOUR = np.array([0.94, 0.94, 0.94, 1.00])
CYCLE_COLOUR = "#FFCD00"


def render_panel(ax, environment: str, class_name: str) -> dict[str, object]:
    entry = environment_data[environment]
    env = entry["env"]
    walls = entry["walls"]
    sigma = entry["sigmas"][class_name]
    selected = dominant_label(env, sigma)
    graph = selected.graph
    height, width = SHAPE

    image = np.tile(EMPTY_COLOUR, (height, width, 1))
    for state in range(int(env.nS)):
        row, column = divmod(state, width)
        if state in walls:
            image[row, column] = WALL_COLOUR
            continue
        basin = int(graph.basin_id[state])
        if basin == selected.dominant_basin:
            image[row, column] = DOMINANT_COLOUR
        else:
            image[row, column] = OTHER_BASIN_COLOURS[
                basin % len(OTHER_BASIN_COLOURS)
            ]

    ax.imshow(image, origin="upper", interpolation="nearest")
    for state in selected.cycle:
        row, column = divmod(state, width)
        ax.add_patch(
            plt.Rectangle(
                (column - 0.5, row - 0.5),
                1,
                1,
                fill=False,
                edgecolor=CYCLE_COLOUR,
                linewidth=1.65,
                zorder=4,
            )
        )
    for coordinate in range(width + 1):
        ax.axvline(coordinate - 0.5, color="white", linewidth=0.32)
    for coordinate in range(height + 1):
        ax.axhline(coordinate - 0.5, color="white", linewidth=0.32)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(
        f"{ACTION_LABELS[selected.label]}: cov {selected.coverage:.2f}\n"
        f"{graph.n_basins} basins; cycle {len(selected.cycle)}",
        fontsize=5.6,
        labelpad=1.5,
    )
    return {
        "environment": environment,
        "class": class_name,
        "label": ACTION_LABELS[selected.label],
        "coverage": selected.coverage,
        "n_basins": graph.n_basins,
        "cycle_length": len(selected.cycle),
        "random_seed": RANDOM_SEED if class_name == "Paired random" else None,
        "run_name": EVOLVED_RUNS[environment][0]
        if class_name == "Evolved exemplar"
        else None,
    }


figure, axes = plt.subplots(
    len(CLASS_ORDER),
    len(ENVIRONMENT_ORDER),
    figsize=(7.15, 4.45),
    dpi=220,
)
panel_rows = []
for row, class_name in enumerate(CLASS_ORDER):
    for column, environment in enumerate(ENVIRONMENT_ORDER):
        axis = axes[row, column]
        panel_rows.append(render_panel(axis, environment, class_name))
        if row == 0:
            axis.set_title(environment, fontsize=7.2, pad=3.5)
        if column == 0:
            axis.annotate(
                class_name,
                xy=(-0.22, 0.5),
                xycoords="axes fraction",
                rotation=90,
                ha="center",
                va="center",
                fontsize=7.1,
                fontweight="semibold",
            )

figure.subplots_adjust(
    left=0.075,
    right=0.998,
    top=0.955,
    bottom=0.035,
    wspace=0.105,
    hspace=0.36,
)

panel_data = pd.DataFrame(panel_rows)
panel_path = ARTIFACT_DIR / "dominant-basin-atlas-panels.csv"
panel_data.to_csv(panel_path, index=False)

png_path = FIGURE_DIR / "F-dominant-basin-atlas.png"
pdf_path = FIGURE_DIR / "F-dominant-basin-atlas.pdf"
figure.savefig(png_path, dpi=300, bbox_inches="tight")
figure.savefig(pdf_path, bbox_inches="tight")
print(panel_data.to_string(index=False))
print(f"saved {panel_path}")
print(f"saved {png_path}")
print(f"saved {pdf_path}")
