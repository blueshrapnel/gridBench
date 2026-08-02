# %% [markdown]
# # Per-label basin anatomy for the paper's small and larger worlds
#
# Rebuilds Figures 12 and 13 of *Twists, home vectors*.  These were formerly
# one-off exports under gridFour's `attractor-fingerprint-probe/figs` with no
# surviving authoritative generator.  The run and sigma references below are
# the provenance encoded in those filenames/titles and in the paper text.

# %%
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from gridbench.functional_graph.label_graphs import label_graphs, walls_and_nonwalls
from gridbench.functional_graph.probe_env import build_goal_free_probe_env
from gridbench.papers.home_vectors import WALL_COLOUR


def _nb_dir(default: str) -> Path:
    """Resolve correctly as a script and from a jupytext/Jupyter kernel."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        cwd = Path.cwd().resolve()
        default_path = Path(default)
        if (cwd / default_path.name).exists() or cwd.name == default_path.parent.name:
            return cwd
        return default_path.parent


HERE = _nb_dir(
    "/home/karen/phd-marlyn/gridBench/notebooks/basin-typology/"
    "03-per-label-basin-anatomy.py"
)
FIGURE_DIR = HERE / "figures"
ARTIFACT_DIR = HERE / "artifacts"
PAPER_FIGURES = Path(
    "/home/karen/Dropbox/phd/writing/twists-home-vectors/figures"
)
for directory in (FIGURE_DIR, ARTIFACT_DIR, PAPER_FIGURES):
    directory.mkdir(parents=True, exist_ok=True)

SCHEMA10_MULTI = Path("/media/merlin/grid-twist/data-schema-10/multi")
GRIDTWIST_OUTPUTS = Path("/media/merlin/grid-twist/gridtwist-outputs")
ACTION_NAMES = "NESW"


@dataclass(frozen=True)
class Exemplar:
    output_name: str
    environment: str
    shape: tuple[int, int]
    run_name: str
    display_id: str
    sigma_hash: str | None = None


EXEMPLARS = (
    Exemplar(
        "F-basins-wrap7.png",
        "wrap_grid",
        (7, 7),
        "g200-pop-96-perm-bal-17-06-b1-free-ga-wrap-grid-fediv-sp3011-08",
        "1856c82c4a05",
    ),
    Exemplar(
        "F-basins-fourrooms7.png",
        "four_rooms",
        (7, 7),
        "g500-pop-96-perm-bal-07-06-b1-free-ga-four-rooms-7x7-k100-sp3011-06-survey",
        "425f30e3d5cd",
    ),
    Exemplar(
        "F-basins-pinwheel7.png",
        "pinwheel",
        (7, 7),
        "g200-pop-96-perm-bal-06-06-b1-free-ga-pinwheel-7x7-k100-sp3011-16",
        "996cf8089efe",
    ),
    Exemplar(
        "F-basins-wrap13.png",
        "wrap_grid",
        (13, 13),
        "g1000-pop-96-perm-bal-04-07-b1-free-ga-wrap-grid-13x13-s316-warm-sp3011-07",
        "s316 g1000",
    ),
    Exemplar(
        "F-basins-fourrooms9-g1000.png",
        "four_rooms",
        (9, 9),
        "g1000-pop-96-perm-bal-20-06-b1-free-ga-four-rooms-9x9-k100-seedmatch-sp3011-02",
        "f5d5c81bfd2c g1000",
        "f5d5c81bfd2c7fc895e4d3834a0292aa",
    ),
    Exemplar(
        "F-basins-fourrooms13.png",
        "four_rooms",
        (13, 13),
        "g1000-pop-96-perm-bal-04-07-b1-free-ga-four-rooms-13x13-s141-warm-sp3011-03",
        "s141 g1000",
    ),
)


def sigma_content_hash(sigma: np.ndarray) -> str:
    """gridFour-compatible BLAKE2b sigma identifier."""
    array = np.asarray(sigma, dtype=np.uint8)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(int(array.shape[0]).to_bytes(4, "little"))
    digest.update(int(array.shape[1]).to_bytes(4, "little"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _schema_run_directory(exemplar: Exemplar) -> Path | None:
    height, width = exemplar.shape
    candidates = list(
        SCHEMA10_MULTI.glob(
            f"init_method=*/env_id={exemplar.environment}/shape={height}x{width}/"
            f"beta=1/det=0.97/run_name={exemplar.run_name}"
        )
    )
    if len(candidates) > 1:
        raise RuntimeError(f"ambiguous run reference for {exemplar.run_name}")
    return candidates[0] if candidates else None


def load_sigma(exemplar: Exemplar) -> tuple[np.ndarray, Path]:
    """Load the pinned run-best, falling back to its raw hive reference."""
    run_directory = _schema_run_directory(exemplar)
    if run_directory is not None:
        candidates = list(run_directory.glob("*multi-all.sigma.npy"))
        if len(candidates) == 1:
            return np.asarray(np.load(candidates[0]), dtype=int), candidates[0]

    raw_run = next(GRIDTWIST_OUTPUTS.glob(f"*/{exemplar.run_name}"), None)
    if raw_run is None:
        raise FileNotFoundError(f"cannot locate {exemplar.run_name}")
    if exemplar.sigma_hash:
        candidates = list(
            raw_run.glob(
                f"hive_sigma/**/sigma_id={exemplar.sigma_hash}/sigma.npy"
            )
        )
    else:
        candidates = list(raw_run.glob("*multi-all.sigma.npy"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one sigma for {exemplar.run_name}, found {len(candidates)}"
        )
    return np.asarray(np.load(candidates[0]), dtype=int), candidates[0]


# Keep the original paper mapping: imshow normalises each panel's basin IDs
# across Matplotlib's 20 categorical colours.  Extending this palette changes
# every normalised lookup, even in panels with fewer than 20 basins.
BASIN_CMAP = ListedColormap(plt.colormaps["tab20"].colors)


def cycle_lengths_text(lengths: list[int]) -> str:
    counts = {length: lengths.count(length) for length in sorted(set(lengths))}
    if len(counts) == 1 and next(iter(counts.values())) > 3:
        length, count = next(iter(counts.items()))
        return f"{{{count}×{length}}}"
    return "[" + ", ".join(str(length) for length in lengths) + "]"


def render_label_panel(ax, env, graph_record) -> dict[str, object]:
    graph = graph_record.fg
    walls, nonwalls = walls_and_nonwalls(env)
    height, width = env.shape
    image = np.full((height, width), np.nan)
    for state in nonwalls:
        row, column = divmod(state, width)
        image[row, column] = int(graph.basin_id[state])
    masked = np.ma.masked_invalid(image)
    cmap = BASIN_CMAP.copy()
    cmap.set_bad(WALL_COLOUR)
    ax.imshow(masked, cmap=cmap, origin="upper", interpolation="nearest")

    for cycle in graph.cycles:
        for state in cycle:
            row, column = divmod(int(state), width)
            ax.plot(
                column,
                row,
                "o",
                markersize=4.2,
                markerfacecolor="black",
                markeredgecolor="white",
                markeredgewidth=0.85,
                zorder=4,
            )

    for basin, size in enumerate(graph.basin_sizes):
        states = np.flatnonzero(graph.basin_id == basin)
        if states.size == 0:
            continue
        rows = states // width
        columns = states % width
        centre = np.argmin(
            (rows - float(rows.mean())) ** 2 + (columns - float(columns.mean())) ** 2
        )
        label = f"{basin + 1}" + (f" ({size})" if size > 1 else "")
        ax.text(
            int(columns[centre]),
            int(rows[centre]),
            label,
            ha="center",
            va="center",
            fontsize=7.4 if max(env.shape) <= 9 else 6.4,
            fontweight="semibold",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            },
            zorder=5,
        )

    lengths = [len(cycle) for cycle in graph.cycles]
    ax.set_title(
        f"label {ACTION_NAMES[graph_record.label]}  ·  {graph.n_basins} "
        f"basin{'s' if graph.n_basins != 1 else ''}\n"
        f"cycle lengths {cycle_lengths_text(lengths)}",
        fontsize=8.6,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    return {
        "label": ACTION_NAMES[graph_record.label],
        "n_basins": graph.n_basins,
        "largest_basin": max(graph.basin_sizes),
        "cycle_lengths": ";".join(str(value) for value in lengths),
        "n_walls": len(walls),
    }


def render(exemplar: Exemplar) -> list[dict[str, object]]:
    sigma, sigma_path = load_sigma(exemplar)
    expected_shape = (exemplar.shape[0] * exemplar.shape[1], 4)
    if sigma.shape != expected_shape:
        raise ValueError(f"{sigma_path}: expected {expected_shape}, got {sigma.shape}")
    content_hash = sigma_content_hash(sigma)
    if exemplar.sigma_hash and content_hash != exemplar.sigma_hash:
        raise ValueError(
            f"{exemplar.run_name}: expected {exemplar.sigma_hash}, got {content_hash}"
        )
    if exemplar.display_id[0].isalnum() and len(exemplar.display_id.split()[0]) == 12:
        if not content_hash.startswith(exemplar.display_id.split()[0]):
            raise ValueError(
                f"{exemplar.run_name}: title hash does not match {content_hash}"
            )

    env = build_goal_free_probe_env(exemplar.environment, exemplar.shape, 0.97)
    graphs = label_graphs(env, sigma)
    # Preserve the paper panels' established 1500×1590 canvas and typography.
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.0, 10.6),
        dpi=150,
    )
    rows = []
    for axis, graph_record in zip(axes.flat, graphs, strict=True):
        row = render_label_panel(axis, env, graph_record)
        row.update(
            {
                "output": exemplar.output_name,
                "environment": exemplar.environment,
                "shape": f"{exemplar.shape[0]}x{exemplar.shape[1]}",
                "run_name": exemplar.run_name,
                "sigma_hash": content_hash,
                "sigma_path": str(sigma_path),
            }
        )
        rows.append(row)
    figure.suptitle(
        f"σ {exemplar.display_id}  ·  {exemplar.environment} "
        f"{exemplar.shape[0]}x{exemplar.shape[1]} β=1",
        fontsize=11,
    )
    figure.subplots_adjust(
        left=0.032,
        right=0.968,
        top=0.894,
        bottom=0.0145,
        wspace=0.11,
        hspace=0.098,
    )
    local_path = FIGURE_DIR / exemplar.output_name
    paper_path = PAPER_FIGURES / exemplar.output_name
    figure.savefig(local_path)
    figure.savefig(paper_path)
    plt.close(figure)
    print(f"saved {local_path}")
    print(f"saved {paper_path}")
    return rows


all_rows = []
for exemplar in EXEMPLARS:
    all_rows.extend(render(exemplar))

audit_path = ARTIFACT_DIR / "per-label-basin-anatomy.csv"
with audit_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
    writer.writeheader()
    writer.writerows(all_rows)
print(f"saved {audit_path}")
