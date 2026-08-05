# %% [markdown]
# # cross-world 06 — The ten-world four-rooms stack
#
# Select the five additional masks exactly, rather than relying on a manually
# assembled panel.  The fixed part of the ensemble is the template's rotation
# orbit plus the symmetric centre cross.  Among the remaining catalogue
# members, choose five that maximise the minimum pairwise Hamming distance over
# all ten masks.  Ties maximise the number of contested cells, then the total
# pairwise distance, then use lexical order.  The figure uses the same wall
# colour as the environment palette in the homing and cross-world papers.

# %%
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from gridbench.papers.home_vectors import WALL_COLOUR


def _nb_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


NB_DIR = _nb_dir()
CATALOGUE = NB_DIR / "data" / "fourrooms_stack_catalogue.json"
DEFAULT_OUTPUT = NB_DIR / "figs" / "fourroom-stack-ten.png"

TEMPLATE_ORBIT = [f"fr-R4Cu4Cd3-rot{angle}" for angle in (0, 90, 180, 270)]
SYMMETRIC_CENTRE = "fr-R3Cu3Cd3-rot0"


def wall_mask(member: dict) -> frozenset[tuple[int, int]]:
    return frozenset(tuple(cell) for cell in member["wall_cells"])


def hamming(a: frozenset, b: frozenset) -> int:
    return len(a.symmetric_difference(b))


def select_ensemble(members: list[dict]) -> tuple[list[dict], dict]:
    """Return the reproducible ten-mask ensemble and its audit statistics."""
    by_label = {member["label"]: member for member in members}
    if len(by_label) != 84:
        raise ValueError(f"expected 84 unique catalogue members, got {len(by_label)}")
    if not all(
        member["valid"]
        and member["connected"]
        and member["walls"] == 9
        and member["n_doors"] == 4
        and len(member["rooms"]) == 4
        for member in members
    ):
        raise ValueError("catalogue contains an invalid four-room member")

    fixed_labels = TEMPLATE_ORBIT + [SYMMETRIC_CENTRE]
    missing = [label for label in fixed_labels if label not in by_label]
    if missing:
        raise ValueError(f"catalogue is missing fixed members: {missing}")

    masks = {label: wall_mask(member) for label, member in by_label.items()}
    candidates = [label for label in sorted(by_label) if label not in fixed_labels]
    distances = sorted(
        {
            hamming(masks[a], masks[b])
            for a, b in itertools.combinations(sorted(by_label), 2)
        },
        reverse=True,
    )

    selected_extra: tuple[str, ...] | None = None
    selected_threshold: int | None = None
    best_secondary: tuple[int, int] | None = None

    for threshold in distances:
        eligible = [
            label
            for label in candidates
            if all(hamming(masks[label], masks[fixed]) >= threshold for fixed in fixed_labels)
        ]
        graph = nx.Graph()
        graph.add_nodes_from(eligible)
        graph.add_edges_from(
            (a, b)
            for a, b in itertools.combinations(eligible, 2)
            if hamming(masks[a], masks[b]) >= threshold
        )

        seen: set[tuple[str, ...]] = set()
        for clique in nx.find_cliques(graph):
            if len(clique) < 5:
                continue
            for extra in itertools.combinations(sorted(clique), 5):
                if extra in seen:
                    continue
                seen.add(extra)
                labels = fixed_labels + list(extra)
                contested = len(set().union(*(masks[label] for label in labels)))
                total_distance = sum(
                    hamming(masks[a], masks[b])
                    for a, b in itertools.combinations(labels, 2)
                )
                secondary = (contested, total_distance)
                if (
                    selected_extra is None
                    or secondary > best_secondary
                    or (secondary == best_secondary and extra < selected_extra)
                ):
                    selected_extra = extra
                    selected_threshold = threshold
                    best_secondary = secondary
        if selected_extra is not None:
            break

    if selected_extra is None or selected_threshold is None:
        raise RuntimeError("could not select five additional catalogue members")

    labels = fixed_labels + list(selected_extra)
    selected_masks = [masks[label] for label in labels]
    always_wall = set.intersection(*(set(mask) for mask in selected_masks))
    ever_wall = set.union(*(set(mask) for mask in selected_masks))
    min_distance = min(
        hamming(masks[a], masks[b]) for a, b in itertools.combinations(labels, 2)
    )
    stats = {
        "labels": labels,
        "minimum_pairwise_hamming": min_distance,
        "contested_cells": len(ever_wall - always_wall),
        "always_wall_cells": len(always_wall),
        "always_walkable_cells": 49 - len(ever_wall),
        "total_pairwise_hamming": sum(
            hamming(masks[a], masks[b]) for a, b in itertools.combinations(labels, 2)
        ),
    }
    if min_distance != selected_threshold:
        raise AssertionError("selection threshold and realised minimum disagree")
    if stats["always_wall_cells"] != 0:
        raise AssertionError("every genome row should be walkable in at least one world")

    return [by_label[label] for label in labels], stats


def plot_ensemble(selected: list[dict], output: Path, *, suptitle: bool = True) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(15.5, 7.5), dpi=180)
    for index, (ax, member) in enumerate(zip(axes.flat, selected)):
        walls = wall_mask(member)
        for row in range(7):
            for col in range(7):
                ax.add_patch(
                    plt.Rectangle(
                        (col, 6 - row),
                        1,
                        1,
                        facecolor=WALL_COLOUR if (row, col) in walls else "white",
                        edgecolor="#d9d9d9",
                        linewidth=0.7,
                    )
                )
        for spine in ax.spines.values():
            spine.set_color("#d4a017" if index == 0 else "#808080")
            spine.set_linewidth(2.4 if index == 0 else 0.9)
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 7)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        title_colour = "#8a6500" if index == 0 else "#222222"
        ax.set_title(
            f"{member['label']}\nrooms {member['rooms']}",
            fontsize=8.3,
            color=title_colour,
            pad=5,
        )
    if suptitle:
        fig.suptitle(
            "Ten-world four-rooms ensemble (template in gold)",
            fontsize=12,
            y=0.995,
        )
    fig.subplots_adjust(
        left=0.025,
        right=0.985,
        bottom=0.035,
        top=0.88 if suptitle else 0.94,
        wspace=0.06,
        hspace=0.34,
    )
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-suptitle", action="store_true",
                        help="omit the figure suptitle (journal-clean export)")
    args = parser.parse_args()

    catalogue = json.loads(args.catalogue.read_text())
    if catalogue.get("n") != 84 or catalogue.get("shape") != [7, 7]:
        raise ValueError("unexpected four-rooms catalogue metadata")
    selected, stats = select_ensemble(catalogue["members"])
    plot_ensemble(selected, args.output, suptitle=not args.no_suptitle)
    print(json.dumps(stats, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
