"""Check the gridBench ownership manifest for *Twists, home vectors*."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
GRIDBENCH = HERE.parents[1]
NOTEBOOKS = GRIDBENCH / "notebooks"
PAPER = Path("/home/karen/Dropbox/phd/writing/twists-home-vectors")


@dataclass(frozen=True)
class Figure:
    number: str
    generator: str
    assets: tuple[str, ...]
    shared_wall_palette: bool = False


FIGURES = (
    Figure("1", "environment-topology/00-environment-palette.py", ("F-environment-palette.png",), True),
    Figure("2", "relabelling-actions/figure-1-simplified.ipynb", ("F-alife-four-rooms-twist.pdf",)),
    Figure("3", "relabelling-actions/figure-3-tradeoff.py", ("F-alife-tradeoff.pdf",)),
    Figure("4", "basin-typology/01-habit-cycle-intro.py", ("F0-habit-cycle-intro.png",), True),
    Figure("5", "single-label-reach/02-random-twist-diffusion.py", ("F1-trajectory-streaks.png",)),
    Figure("6", "single-label-reach/02-random-twist-diffusion.py", ("F2-diffusion-summary.png",)),
    Figure("7", "single-label-reach/01-reach-by-class-two-shapes.py", ("F-reach-by-class.png",)),
    Figure("8", "single-label-reach/00-four-rooms-reach-exemplar.py", ("F-reach-trajectories.png",), True),
    Figure("9", "paper-homing/21-fingerprint-planes.py", ("F-fingerprint-open-interiors.png",)),
    Figure("10", "paper-homing/21-fingerprint-planes.py", ("F-fingerprint-walled-interiors.png",)),
    Figure("11", "basin-typology/00-dominant-basin-atlas.py", ("F-dominant-basin-atlas.pdf",), True),
    Figure(
        "12",
        "basin-typology/03-per-label-basin-anatomy.py",
        ("F-basins-wrap7.png", "F-basins-fourrooms7.png", "F-basins-pinwheel7.png"),
        True,
    ),
    Figure(
        "13",
        "basin-typology/03-per-label-basin-anatomy.py",
        ("F-basins-wrap13.png", "F-basins-fourrooms9-g1000.png", "F-basins-fourrooms13.png"),
        True,
    ),
    Figure("14", "goal-geometry/14-seed-strips-four-label.py", ("F-seed-strips-four-label.png",)),
    Figure("15", "basin-typology/02-cycle-escape-policy.py", ("F-cycle-escape-policy-view-v2.png",), True),
    Figure("16", "paper-homing/21-fingerprint-planes.py", ("F-alignment-roundup.png",)),
)


def main() -> None:
    tex = (PAPER / "main.tex").read_text()
    failures = []
    for figure in FIGURES:
        generator = NOTEBOOKS / figure.generator
        if not generator.is_file():
            failures.append(f"Figure {figure.number}: missing generator {generator}")
            continue
        if figure.shared_wall_palette:
            source = generator.read_text(errors="replace")
            if "gridbench.papers.home_vectors import WALL_COLOUR" not in source:
                failures.append(
                    f"Figure {figure.number}: generator bypasses shared WALL_COLOUR"
                )
        for asset in figure.assets:
            path = PAPER / "figures" / asset
            if not path.is_file():
                failures.append(f"Figure {figure.number}: missing paper asset {path}")
            if asset not in tex:
                failures.append(
                    f"Figure {figure.number}: {asset} is not included by main.tex"
                )
        print(
            f"Figure {figure.number:>2}: {figure.generator} -> "
            + ", ".join(figure.assets)
        )
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"\nOK: {len(FIGURES)} numbered figures have gridBench owners and paper assets")


if __name__ == "__main__":
    main()
