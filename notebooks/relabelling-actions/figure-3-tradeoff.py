"""Render the ALife action-alignment trade-off used as paper Figure 3.

The expensive all-goal sweep is a committed, paper-specific data asset.  This
renderer deliberately reads that asset directly: rebuilding a figure must not
import the frozen gridFour report stack or repeat the numerical sweep.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    "/home/karen/phd-marlyn/gridBench/notebooks/relabelling-actions/"
    "figure-3-tradeoff.py"
)
SOURCE = (
    HERE
    / "four_rooms"
    / "assets"
    / "ga_evolution"
    / "four_rooms_g200_tradeoff_curve_all_goals_wide_beta.csv"
)
LOCAL_BASE = HERE / "figures" / "figure3_tradeoff_curve"
PAPER_BASE = Path(
    "/home/karen/Dropbox/phd/writing/twists-home-vectors/figures/"
    "F-alife-tradeoff"
)


def _finite_sorted(frame: pd.DataFrame, prefix: str) -> dict[str, np.ndarray]:
    fields = {
        name: np.asarray(frame[f"{name}_{prefix}"], dtype=float)
        for name in (
            "mean_info",
            "mean_value",
            "value_q25",
            "value_q75",
            "value_min",
            "value_max",
        )
    }
    valid = np.logical_and.reduce([np.isfinite(values) for values in fields.values()])
    order = np.argsort(fields["mean_info"][valid])
    return {name: values[valid][order] for name, values in fields.items()}


def _draw_family(
    ax,
    values: dict[str, np.ndarray],
    *,
    colour: str,
    mean_label: str,
    range_label: str,
    iqr_label: str,
    marker: str,
    linestyle: str,
) -> None:
    x = values["mean_info"]
    ax.plot(
        x,
        values["mean_value"],
        linestyle=linestyle,
        linewidth=1.2,
        color=colour,
        marker=marker,
        markersize=9 if marker == "*" else 6.6,
        markerfacecolor="none",
        markeredgewidth=0.7,
        label=mean_label,
        zorder=3,
    )
    ax.fill_between(
        x,
        values["value_min"],
        values["value_max"],
        color=colour,
        alpha=0.18,
        linewidth=0,
        label=range_label,
        zorder=1,
    )
    ax.plot(
        x,
        values["value_q25"],
        linestyle=(0, (1.2, 1.8)),
        linewidth=1,
        color=colour,
        alpha=0.9,
        label=iqr_label,
        zorder=2,
    )
    ax.plot(
        x,
        values["value_q75"],
        linestyle=(0, (1.2, 1.8)),
        linewidth=1,
        color=colour,
        alpha=0.9,
        zorder=2,
    )


def render() -> None:
    curve = pd.read_csv(SOURCE)
    baseline = _finite_sorted(curve, "baseline")
    twist = _finite_sorted(curve, "twist")

    figure, axis = plt.subplots(figsize=(7.2, 5.4))
    _draw_family(
        axis,
        baseline,
        colour="#4d4d4d",
        mean_label=r"$\chi=0$ mean",
        range_label=r"$\chi=0$ goal min-max",
        iqr_label=r"$\chi=0$ IQR (q25/q75)",
        marker="o",
        linestyle="--",
    )
    _draw_family(
        axis,
        twist,
        colour="#d55e00",
        mean_label=r"Best $\sigma$ mean",
        range_label=r"Best $\sigma$ goal min-max",
        iqr_label=r"Best $\sigma$ IQR (q25/q75)",
        marker="*",
        linestyle="-",
    )

    all_x = np.concatenate([baseline["mean_info"], twist["mean_info"]])
    all_y = np.concatenate(
        [
            baseline[name]
            for name in ("mean_value", "value_q25", "value_q75", "value_min", "value_max")
        ]
        + [
            twist[name]
            for name in ("mean_value", "value_q25", "value_q75", "value_min", "value_max")
        ]
    )
    xpad = 0.05 * (float(all_x.max()) - float(all_x.min()))
    ypad = 0.06 * (float(all_y.max()) - float(all_y.min()))
    axis.set_xlim(float(all_x.min()) - xpad, float(all_x.max()) + xpad)
    axis.set_ylim(float(all_y.min()) - ypad, min(0.0, float(all_y.max()) + ypad))
    axis.set_xlabel(r"$E_{s,g}\!\left[\,\mathcal{I}_{D}^{\pi}(s)\,\right]$")
    axis.set_ylabel(r"$E_{s,g}\!\left[\,V^{\pi}(s)\,\right]$")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=8)

    outputs = (
        LOCAL_BASE.with_suffix(".png"),
        LOCAL_BASE.with_suffix(".pdf"),
        PAPER_BASE.with_suffix(".pdf"),
    )
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.02)
        print(f"saved {output}")
    plt.close(figure)


if __name__ == "__main__":
    render()
