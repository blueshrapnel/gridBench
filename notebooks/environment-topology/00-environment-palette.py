#!/usr/bin/env python3
"""Render the paper's topology-only environment palette from gridCore."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

from gridcore.bridge import build_env_by_id
from gridbench.papers.home_vectors import WALL_COLOUR
from gridvis.env import plot_environment


DISPLAY_ROWS = (
    ("wrap_grid", "open_grid", "helical"),
    ("pinwheel", "four_rooms", "pillar_3"),
)
EXPECTED = {
    "wrap_grid": (49, 0, True, (0, 0)),
    "open_grid": (49, 0, False, (0, 0)),
    "helical": (49, 0, True, (1, 0)),
    "pinwheel": (32, 17, False, (0, 0)),
    "four_rooms": (40, 9, False, (0, 0)),
    "x_wall": (40, 9, False, (0, 0)),
    "pillar_3": (40, 9, False, (0, 0)),
}
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
    "/home/karen/phd-marlyn/gridBench/notebooks/environment-topology/"
    "00-environment-palette.py"
)
PAPER_FIGURES = Path(
    "/home/karen/Dropbox/phd/writing/twists-home-vectors/figures"
)


def build(env_id: str):
    """Build an environment and assert the paper's topology contract."""
    env = build_env_by_id(
        env_id=env_id,
        shape=(7, 7),
        goal=0,
        determinism=0.97,
        manhattan=True,
    )
    observed = (
        len(env.available_states),
        len(env.walls_flat),
        bool(env.wrap),
        tuple(env.seam_shift),
    )
    assert observed == EXPECTED[env_id], (env_id, observed, EXPECTED[env_id])

    if env_id == "pillar_3":
        centre = {row * 7 + col for row in range(2, 5) for col in range(2, 5)}
        assert set(env.walls_flat) == centre

    return env


def add_periodic_markers(ax, env_id: str) -> None:
    """Mark edge identifications that are invisible in a wall-only view."""
    blue = "#2b6cb0"
    arrow = dict(
        arrowstyle="<->",
        color=blue,
        linewidth=1.4,
        shrinkA=0,
        shrinkB=0,
        clip_on=False,
    )

    # All wrapped environments identify their left and right edges.
    ax.annotate("", xy=(1.0, -0.09), xytext=(0.0, -0.09),
                xycoords="axes fraction", arrowprops=arrow)

    if env_id == "wrap_grid":
        ax.annotate("", xy=(-0.09, 1.0), xytext=(-0.09, 0.0),
                    xycoords="axes fraction", arrowprops=arrow)
    else:
        # Keep the shifted-seam marker outside the cells so that it cannot be
        # mistaken for a trajectory or policy.  Its horizontal displacement
        # is one cell-width: bottom column j is identified with top j + 1.
        seam = FancyArrowPatch(
            (-0.16, 0.0),
            (-0.02, 1.0),
            arrowstyle="-|>",
            mutation_scale=9,
            color=blue,
            linewidth=1.4,
            clip_on=False,
            transform=ax.transAxes,
        )
        ax.add_patch(seam)


def draw(ax, env_id: str) -> None:
    env = build(env_id)
    topology = SimpleNamespace(shape=env.shape, walls_flat=env.walls_flat, goals=[])
    plot_environment(
        topology,
        ax=ax,
        wall_color=WALL_COLOUR,
        empty_color="#fbfbfa",
        grid_color="#c8cbd0",
    )

    # Use vector rectangles for walls.  The image layer used by gridVis can be
    # resampled by PDF viewers when it is scaled, softening wall edges; these
    # patches remain sharp at every zoom level.
    rows, cols = env.shape
    for state in env.walls_flat:
        row, column = divmod(int(state), cols)
        ax.add_patch(
            Rectangle(
                (column - 0.5, row - 0.5),
                1,
                1,
                facecolor=WALL_COLOUR,
                edgecolor="none",
                zorder=1.5,
            )
        )

    # gridVis normally derives its grid from Matplotlib's minor ticks.  Draw
    # every cell boundary explicitly here: minor ticks that overlap a major
    # locator can otherwise be suppressed by Matplotlib, dropping a line.
    x_edges = np.arange(-0.5, cols + 0.5, 1.0)
    y_edges = np.arange(-0.5, rows + 0.5, 1.0)
    ax.grid(False, which="both")
    ax.vlines(
        x_edges,
        -0.5,
        rows - 0.5,
        colors="#c8cbd0",
        linewidth=0.8,
        zorder=2,
    )
    ax.hlines(
        y_edges,
        -0.5,
        cols - 0.5,
        colors="#c8cbd0",
        linewidth=0.8,
        zorder=2,
    )
    ax.set_title(env_id, family="monospace", fontsize=9, pad=5)

    wrapped = env_id in {"wrap_grid", "helical"}
    for spine in ax.spines.values():
        spine.set_color("#2b6cb0" if wrapped else "#50555a")
        spine.set_linewidth(1.4 if wrapped else 1.0)
        spine.set_linestyle((0, (3, 2)) if wrapped else "solid")
    if wrapped:
        add_periodic_markers(ax, env_id)


def render(output: Path, preview: Path | None) -> None:
    # Validate every named environment, including x_wall, even though the
    # compact paper palette displays only six of them.
    for env_id in EXPECTED:
        build(env_id)

    fig, axes = plt.subplots(2, 3, figsize=(5.35, 3.72))
    fig.patch.set_facecolor("white")

    for row_axes, row_ids in zip(axes, DISPLAY_ROWS, strict=True):
        for ax, env_id in zip(row_axes, row_ids, strict=True):
            draw(ax, env_id)

    fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.055,
                        wspace=0.37, hspace=0.43)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    if preview is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            preview,
            dpi=220,
            bbox_inches="tight",
            pad_inches=0.06,
            facecolor="white",
        )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PAPER_FIGURES / "F-environment-palette.pdf",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=PAPER_FIGURES / "F-environment-palette.png",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render(args.output, args.preview)
