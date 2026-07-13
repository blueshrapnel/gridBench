from __future__ import annotations

import argparse
import itertools
import math
import pickle
import shutil
import textwrap
import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from gridbench import store
import gridvis.display as di
from gridcore import twists
from gridbench.metrics import matrix_stats
from gridvis.geometry_plots import get_MDS_plot
from gridvis.geometry_plots import calculate_symmetric_adjacency_matrix, get_matrix_embedding
from gridvis.twist_plot_helpers import render_twist_figure

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None


METRIC_KEYS = ("frees", "infos", "values", "env_responses")


def _fmt_float(val, precision: int = 4) -> str:
    if val is None:
        return "NA"
    try:
        v = float(val)
    except Exception:
        return str(val)
    if np.isnan(v):
        return "nan"
    return f"{v:.{precision}g}"


def _fmt_timestamp_compact(val) -> str:
    if val is None:
        return "NA"
    text = str(val).strip()
    if not text:
        return "NA"
    # Accept already-compact values (yy-mm-dd-hh-mm).
    try:
        if len(text) == 14 and text.count("-") == 4:
            return text
    except Exception:
        pass
    # Parse common ISO-like timestamps and render compact.
    try:
        norm = text.replace("Z", "+00:00")
        dt_obj = datetime.fromisoformat(norm)
        return dt_obj.strftime("%y-%m-%d-%H-%M")
    except Exception:
        return text


def _compute_eps_actual(data: dict, env) -> float | None:
    sigma = data.get("sigma")
    if sigma is None:
        return None
    states = getattr(env, "available_states", list(range(env.nS)))
    n_actions = getattr(env, "nA", getattr(getattr(env, "actions", None), "nA", None))
    try:
        return float(twists.compute_epsilon_actual(sigma, states, n_actions))
    except Exception:
        return None


def _compute_metric_means(data: dict) -> dict[str, float | None]:
    out: dict[str, float | None] = {"mean_free": None, "mean_info": None, "mean_value": None}
    if "frees" in data:
        try:
            out["mean_free"] = float(matrix_stats(np.asarray(data["frees"], dtype=float)).mean_free)
        except Exception:
            pass
    if "infos" in data:
        try:
            out["mean_info"] = float(matrix_stats(np.asarray(data["infos"], dtype=float)).mean_free)
        except Exception:
            pass
    if "values" in data:
        try:
            out["mean_value"] = float(matrix_stats(np.asarray(data["values"], dtype=float)).mean_free)
        except Exception:
            pass
    return out


def _assignment_alignment_metrics(
    sigma: np.ndarray,
    available_states: list[int],
) -> tuple[int, float, float, tuple[int, ...], str]:
    arr = np.asarray(sigma, dtype=int)
    if arr.ndim != 2:
        raise ValueError("sigma must be 2D")
    n_actions = int(arr.shape[1])
    states = [int(s) for s in available_states if 0 <= int(s) < arr.shape[0]]
    if n_actions <= 0 or not states:
        return n_actions, 0.0, 1.0, tuple(), "lexicographic_first"

    counts = np.zeros((n_actions, n_actions), dtype=int)
    for s in states:
        row = arr[s, :n_actions]
        for b in range(n_actions):
            a = int(row[b])
            if 0 <= a < n_actions:
                counts[a, b] += 1

    best_sum = -1
    best_perm: tuple[int, ...] = tuple(range(n_actions))
    for perm in itertools.permutations(range(n_actions)):
        score = sum(int(counts[a, perm[a]]) for a in range(n_actions))
        if score > best_sum:
            best_sum = score
            best_perm = tuple(int(x) for x in perm)
        # deterministic tie-break: first lexicographic max permutation wins

    denom = float(len(states) * n_actions)
    chi_align = float(best_sum / denom) if denom > 0 else 0.0
    chi_twist = float(1.0 - chi_align)
    return n_actions, chi_align, chi_twist, best_perm, "lexicographic_first"


def _resolve_orientation_metadata(data: dict, env, provenance: dict) -> dict[str, str]:
    """Return orientation metadata, preferring provenance and falling back to on-the-fly computation."""
    out: dict[str, str] = {
        "n_actions": str(provenance.get("n_actions", "NA")),
        "chi_align": _fmt_float(provenance.get("chi_align")),
        "chi_twist": _fmt_float(provenance.get("chi_twist")),
        "predominant_ordering": str(provenance.get("predominant_ordering", "NA")),
        "predominant_ordering_tie_rule": str(provenance.get("predominant_ordering_tie_rule", "NA")),
    }
    sigma = data.get("sigma")
    if sigma is None:
        return out
    states = [int(s) for s in getattr(env, "available_states", list(range(int(getattr(env, "nS", 0)))))]
    try:
        n_actions, chi_align, chi_twist, ordering, tie_rule = _assignment_alignment_metrics(
            np.asarray(sigma, dtype=int),
            states,
        )
    except Exception:
        return out

    if out["n_actions"] in {"NA", "None", ""}:
        out["n_actions"] = str(int(n_actions))
    if out["chi_align"] in {"NA", "nan", "None", ""}:
        out["chi_align"] = _fmt_float(chi_align)
    if out["chi_twist"] in {"NA", "nan", "None", ""}:
        out["chi_twist"] = _fmt_float(chi_twist)
    if out["predominant_ordering"] in {"NA", "None", ""}:
        out["predominant_ordering"] = str(tuple(int(x) for x in ordering))
    if out["predominant_ordering_tie_rule"] in {"NA", "None", ""}:
        out["predominant_ordering_tie_rule"] = str(tie_rule)
    return out

def _load_collated(path: Path) -> dict:
    with path.open("rb") as fh:
        with warnings.catch_warnings():
            # Legacy pickles can emit NumPy's numpy.core.numeric deprecation warning
            # while unpickling. Suppress this specific warning only.
            warnings.filterwarnings(
                "ignore",
                message=r".*numpy\.core\.numeric is deprecated.*",
                category=DeprecationWarning,
            )
            warnings.filterwarnings(
                "ignore",
                module=r"numpy(\.|$)",
                category=DeprecationWarning,
            )
            return pickle.load(fh)


def _parse_tag(path: Path, key: str) -> str | None:
    needle = f"{key}="
    for part in path.parts:
        if part.startswith(needle):
            return part[len(needle):]
    return None


def _iter_collated(root: Path) -> Iterable[Path]:
    yield from root.rglob("collated-*.pickle")


def _filename_env_tag(env_id: str | None) -> str:
    if not env_id:
        return "env-unknown"
    return f"env-{env_id.replace('/', '-').replace(' ', '_')}"


def _build_undirected_edges(env) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for state in range(env.nS):
        for action in env.D[state]:
            for _, successor, _, _ in env.D[state][action]:
                if successor == state:
                    continue
                edge = tuple(sorted((state, successor)))
                edges.add(edge)
    return edges


def _active_states(env, n_states: int) -> list[int]:
    walls = set(getattr(env, "walls_flat", []))
    return [s for s in range(n_states) if s not in walls]


def _node_colours_hex(env, env_id: str | None = None) -> list[str]:
    return [di.mcolors.to_hex(c) for c in di.get_state_node_colours(env, env_id=env_id)]


def _state_colour_legend_traces(env, env_id: str | None = None):
    raw_colours = _node_colours_hex(env, env_id=env_id)
    group_to_states: dict[str, list[int]] = {}
    for state, colour in enumerate(raw_colours):
        group_to_states.setdefault(str(colour), []).append(state)

    traces = []
    for idx, colour in enumerate(sorted(group_to_states)):
        states = group_to_states[colour]
        state_text = ", ".join(map(str, states[:10]))
        if len(states) > 10:
            state_text += ", ..."
        traces.append(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(
                    size=8,
                    color=colour,
                    line=dict(color="black", width=1),
                    symbol="circle",
                ),
                name=f"group {idx}: [{state_text}]",
                showlegend=True,
                hoverinfo="skip",
            )
        )
    return traces


def _grid_legend_coords(env) -> tuple[np.ndarray, np.ndarray]:
    xs = np.zeros(env.nS, dtype=float)
    ys = np.zeros(env.nS, dtype=float)
    width = env.shape[1]
    for state in range(env.nS):
        row, col = divmod(state, width)
        xs[state] = col
        ys[state] = row
    return xs, ys


def _make_sphere_trace(
    x: float,
    y: float,
    z: float,
    radius: float,
    fill: str,
    *,
    opacity: float = 1.0,
    n_theta: int = 14,
    n_phi: int = 10,
):
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    ii: list[int] = []
    jj: list[int] = []
    kk: list[int] = []

    for phi_idx in range(n_phi + 1):
        phi = math.pi * phi_idx / n_phi
        for theta_idx in range(n_theta):
            theta = (2.0 * math.pi * theta_idx) / n_theta
            xs.append(x + radius * math.sin(phi) * math.cos(theta))
            ys.append(y + radius * math.sin(phi) * math.sin(theta))
            zs.append(z + radius * math.cos(phi))

    def vertex_index(phi_idx: int, theta_idx: int) -> int:
        return phi_idx * n_theta + (theta_idx % n_theta)

    for phi_idx in range(n_phi):
        for theta_idx in range(n_theta):
            a = vertex_index(phi_idx, theta_idx)
            b = vertex_index(phi_idx, theta_idx + 1)
            c = vertex_index(phi_idx + 1, theta_idx)
            d = vertex_index(phi_idx + 1, theta_idx + 1)
            ii.extend([a, b])
            jj.extend([c, c])
            kk.extend([b, d])

    return go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=ii,
        j=jj,
        k=kk,
        color=fill,
        opacity=opacity,
        flatshading=False,
        lighting=dict(ambient=1.0, diffuse=0.15, specular=0.03, roughness=0.95, fresnel=0.0),
        lightposition=dict(x=100, y=80, z=120),
        hoverinfo="skip",
        showlegend=False,
    )


def _build_compact_state_map_traces(env, node_opacity: float = 1.0, env_id: str | None = None):
    gx, gy = _grid_legend_coords(env)
    spacing = 1.15
    gx = [float(x * spacing) for x in gx]
    gy = [float(y * spacing) for y in gy]
    colours = _node_colours_hex(env, env_id=env_id)

    edge_x, edge_y = [], []
    for i, j in sorted(_build_undirected_edges(env)):
        edge_x.extend([gx[i], gx[j], None])
        edge_y.extend([gy[i], gy[j], None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(color="rgba(90,90,90,0.42)", width=0.9),
        hoverinfo="skip",
        showlegend=False,
        xaxis="x2",
        yaxis="y2",
    )

    node_size = max(20, int(165 / max(env.shape)))
    node_trace = go.Scatter(
        x=gx,
        y=gy,
        mode="markers",
        marker=dict(
            size=node_size,
            color=colours,
            line=dict(color="black", width=1.0),
            symbol="circle",
            opacity=node_opacity,
        ),
        hovertemplate="state=%{text}<extra></extra>",
        text=[str(s) for s in range(env.nS)],
        showlegend=False,
        xaxis="x2",
        yaxis="y2",
    )
    text_trace = go.Scatter(
        x=gx,
        y=gy,
        mode="text",
        text=[str(s) for s in range(env.nS)],
        textposition="middle center",
        textfont=dict(color="white", size=max(8, int(86 / max(env.shape))), family="Arial Black, DejaVu Sans, sans-serif"),
        hoverinfo="skip",
        showlegend=False,
        xaxis="x2",
        yaxis="y2",
    )
    return edge_trace, node_trace, text_trace


def _build_metadata_text(env, metadata: dict[str, str] | None) -> str:
    lines = [
        "<b>Experiment</b>",
        f"shape: {metadata.get('shape', 'NA') if metadata else env.shape}",
        f"states: {env.nS}",
    ]
    if metadata:
        detail_keys = (
            "env_id",
            "neighbourhood",
            "state_dist",
            "det",
            "beta",
            "metric",
            "run_type",
            "chi_twist",
            "mean_info",
            "mean_value",
            "mean_free",
        )
        for key in detail_keys:
            val = metadata.get(key)
            if val not in {None, "", "NA"}:
                lines.append(f"{key}: {val}")
        source_run_id = metadata.get("source_run_id")
        if source_run_id not in {None, "", "NA"}:
            wrapped = "<br>".join(textwrap.wrap(str(source_run_id), width=40))
            lines.append(f"run: {wrapped}")
    return "<br>".join(lines)


def _write_twist_panel_png(env, out_path: Path, *, title: str = "", layout: str = "focus_manhattan") -> None:
    """Write a pure-sigma twist panel image for quick visual inspection."""
    fig, _ = render_twist_figure(
        env,
        policy=None,
        title=title,
        show_full=True,
        layout=layout,
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_twist_panel_exports(
    env,
    out_dir: Path,
    stem: str,
    *,
    title: str = "",
    primary_layout: str = "focus_manhattan",
) -> None:
    outputs = [
        (out_dir / f"{stem}-twist-panel.png", primary_layout),
        (out_dir / f"{stem}-twist-panel-focus-manhattan.png", "focus_manhattan"),
        (out_dir / f"{stem}-twist-panel-dual-landscape.png", "dual_landscape"),
    ]
    rendered_by_layout: dict[str, Path] = {}
    for out_path, layout in outputs:
        existing = rendered_by_layout.get(layout)
        if existing is None:
            _write_twist_panel_png(env, out_path, title=title, layout=layout)
            rendered_by_layout[layout] = out_path
            continue
        if existing == out_path:
            continue
        shutil.copy2(existing, out_path)


def _write_interactive_html(
    matrix: np.ndarray,
    env,
    title: str,
    out_html: Path,
    *,
    node_opacity: float = 1.0,
    include_state_legend: bool = True,
    include_grid_legend_panel: bool = True,
    metadata: dict[str, str] | None = None,
) -> bool:
    if go is None:
        return False
    sym = calculate_symmetric_adjacency_matrix(matrix)
    active_states = _active_states(env, sym.shape[0])
    sym = sym[np.ix_(active_states, active_states)]
    coords = get_matrix_embedding(sym, state=1, components=3)
    active_set = set(active_states)
    state_to_idx = {s: i for i, s in enumerate(active_states)}
    edges = {
        (i, j)
        for s1, s2 in _build_undirected_edges(env)
        if s1 in active_set and s2 in active_set
        for i, j in [(state_to_idx[s1], state_to_idx[s2])]
    }
    env_id = metadata.get("env_id") if metadata else None
    all_node_colours = _node_colours_hex(env, env_id=env_id)
    node_colours = [all_node_colours[s] for s in active_states]

    edge_x, edge_y, edge_z = [], [], []
    for i, j in edges:
        edge_x.extend([coords[i, 0], coords[j, 0], None])
        edge_y.extend([coords[i, 1], coords[j, 1], None])
        edge_z.extend([coords[i, 2], coords[j, 2], None])

    xs = coords[:, 0].astype(float).tolist()
    ys = coords[:, 1].astype(float).tolist()
    zs = coords[:, 2].astype(float).tolist()
    if len(node_colours) != len(xs):
        node_colours = ["#f28c26"] * len(xs)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = max(float((maxs - mins).max()), 1e-6)
    fill_radius = 0.028 * span
    highlight_radius = 0.009 * span
    centre = coords.mean(axis=0)

    traces: list = []
    label_x: list[float] = []
    label_y: list[float] = []
    label_z: list[float] = []
    label_text: list[str] = []

    for idx, state in enumerate(active_states):
        x = float(coords[idx, 0])
        y = float(coords[idx, 1])
        z = float(coords[idx, 2])
        fill = node_colours[idx]
        traces.append(_make_sphere_trace(x, y, z, fill_radius, fill=fill, opacity=node_opacity))
        traces.append(
            _make_sphere_trace(
                x - (0.010 * span),
                y + (0.010 * span),
                z + (0.010 * span),
                highlight_radius,
                fill="#ffffff",
                opacity=0.95 * node_opacity,
            )
        )
        direction = np.asarray([x, y, z], dtype=float) - centre
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            direction = np.asarray([0.0, 0.0, 1.0], dtype=float)
            norm = 1.0
        direction = direction / norm
        label_offset = 1.45 * fill_radius
        label_x.append(x + (label_offset * float(direction[0])))
        label_y.append(y + (label_offset * float(direction[1])))
        label_z.append(z + (label_offset * float(direction[2])))
        label_text.append(str(state))

    traces.append(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(color="rgba(100,100,100,0.82)", width=7),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    scene_annotations = [
        dict(
            x=label_x[i],
            y=label_y[i],
            z=label_z[i],
            text=label_text[i],
            showarrow=False,
            font=dict(color="white", size=13, family="Arial Black, DejaVu Sans, sans-serif"),
            xanchor="center",
            yanchor="middle",
            bgcolor="rgba(0,0,0,0.35)",
            borderpad=1,
        )
        for i in range(len(label_text))
    ]

    if include_grid_legend_panel:
        inset_edge_trace, inset_node_trace, inset_text_trace = _build_compact_state_map_traces(
            env, node_opacity=node_opacity, env_id=env_id
        )
        fig = go.Figure(data=[*traces, inset_edge_trace, inset_node_trace, inset_text_trace])
        inset_w = max(0.14, min(0.24, 0.18 * (env.shape[1] / max(1, env.shape[0]))))
        fig.update_layout(
            xaxis2=dict(
                visible=False,
                domain=[0.02, 0.02 + inset_w],
                range=[-0.6, ((env.shape[1] - 1) * 1.15) + 0.6],
                constrain="domain",
                anchor="y2",
            ),
            yaxis2=dict(
                visible=False,
                domain=[0.73, 0.95],
                range=[((env.shape[0] - 1) * 1.15) + 0.6, -0.6],
                scaleanchor="x2",
                scaleratio=1,
                constrain="domain",
                anchor="x2",
            ),
        )
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text="<b>State Colour Map</b>",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(size=14),
        )
        fig.add_annotation(
            x=0.02,
            y=0.70,
            xref="paper",
            yref="paper",
            text=_build_metadata_text(env, metadata),
            showarrow=False,
            xanchor="left",
            yanchor="top",
            align="left",
            font=dict(size=13),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        )
    else:
        standalone_traces = list(traces)
        if include_state_legend:
            standalone_traces.extend(_state_colour_legend_traces(env, env_id=env_id))
        fig = go.Figure(data=standalone_traces)

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(projection=dict(type="orthographic")),
            annotations=scene_annotations,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=10),
        ),
    )
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-generate 2D/3D MDS plots from collated all-goal pickles."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(store.data_root()),
        help="Root data directory containing shape=/env_id=/.../collated-*.pickle runs.",
    )
    parser.add_argument(
        "--env-id",
        action="append",
        default=[],
        help="Filter by env_id (repeatable), e.g. --env-id open_grid --env-id wrap_grid",
    )
    parser.add_argument(
        "--eps",
        action="append",
        default=[],
        help="Filter by epsilon directory label (repeatable), e.g. --eps 0.0 --eps 0.25",
    )
    parser.add_argument(
        "--seed",
        action="append",
        default=[],
        help="Filter by seed directory label (repeatable), e.g. --seed 42 --seed 31415",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["frees"],
        choices=METRIC_KEYS,
        help="Metric matrices to embed.",
    )
    parser.add_argument(
        "--components",
        choices=["2", "3", "both"],
        default="both",
        help="Which MDS dimensionality to render (default: both).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of collated files to process (0 = all).",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate node indices on plots.",
    )
    parser.add_argument(
        "--interactive-html",
        action="store_true",
        help="Also export interactive 3D Plotly HTML for each metric.",
    )
    parser.add_argument(
        "--twist-layout",
        choices=["classic", "focus_manhattan", "dual_columns", "dual_landscape"],
        default="focus_manhattan",
        help="Layout for the exported pure-sigma twist panel PNG.",
    )
    parser.add_argument(
        "--node-opacity",
        type=float,
        default=1.0,
        help="Node opacity for interactive HTML markers (default: 1.0).",
    )
    parser.add_argument(
        "--no-html-state-legend",
        action="store_true",
        help="Disable HTML legend for GridRoom state-colour groups.",
    )
    parser.add_argument(
        "--no-html-grid-legend-panel",
        action="store_true",
        help="Disable 2D spatial state-colour panel in interactive HTML.",
    )
    parser.add_argument(
        "--interactive-mpl",
        action="store_true",
        help="Show Matplotlib windows for manual 3D camera rotation during generation.",
    )
    parser.add_argument(
        "--angles",
        nargs="*",
        type=float,
        default=[],
        help="Optional azimuth angles (degrees) to save additional rotated 3D PNGs.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    env_filter = set(args.env_id)
    eps_filter = set(args.eps)
    seed_filter = set(args.seed)

    processed = 0
    skipped = 0
    for collated in sorted(_iter_collated(root)):
        env_id = _parse_tag(collated, "env_id")
        eps = _parse_tag(collated, "eps")
        seed = _parse_tag(collated, "seed")

        if env_filter and env_id not in env_filter:
            skipped += 1
            continue
        if eps_filter and eps not in eps_filter:
            skipped += 1
            continue
        if seed_filter and seed not in seed_filter:
            skipped += 1
            continue

        data = _load_collated(collated)
        env = data.get("env") or data["state_dist"].env
        provenance = data.get("provenance", {}) if isinstance(data, dict) else {}
        orientation_meta = _resolve_orientation_metadata(data, env, provenance)
        eps_param = data.get("epsilon", eps)
        eps_actual = _compute_eps_actual(data, env)
        means = _compute_metric_means(data)
        out_dir = collated.parent / "sigma-plots"
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = collated.stem
        try:
            _write_twist_panel_exports(
                env,
                out_dir,
                stem,
                title=f"Twist Panel | {env_id or 'unknown'} | seed={seed or 'NA'}",
                primary_layout=args.twist_layout,
            )
        except Exception as e:
            print(f"[WARN] twist panel export failed for {collated}: {e}")
        for metric in args.metrics:
            if metric not in data:
                continue
            matrix = np.asarray(data[metric], dtype=float)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                continue
            if not np.isfinite(matrix).any():
                print(f"[SKIP] {collated} metric={metric} has no finite values")
                continue

            title_prefix = (
                f"{env_id or 'unknown'} | beta={_parse_tag(collated, 'beta') or 'NA'} | "
                f"eps_actual={_fmt_float(eps_actual)} | "
                f"seed={seed or 'NA'} | {metric}"
            )
            env_tag = _filename_env_tag(env_id)
            if args.components in {"2", "both"}:
                ax2d = get_MDS_plot(
                    matrix, env, annotate=args.annotate, components=2, title=f"2D MDS | {title_prefix}", env_id=env_id
                )
                fig2d = ax2d.get_figure()
                fig2d.savefig(out_dir / f"{stem}-{env_tag}-{metric}-2d-mds.png", dpi=180, bbox_inches="tight")
                if args.interactive_mpl:
                    plt.show(block=True)
                plt.close(fig2d)

            if args.components in {"3", "both"}:
                ax3d = get_MDS_plot(
                    matrix, env, annotate=args.annotate, components=3, title=f"3D MDS | {title_prefix}", env_id=env_id
                )
                fig3d = ax3d.get_figure()
                fig3d.savefig(out_dir / f"{stem}-{env_tag}-{metric}-3d-mds.png", dpi=180, bbox_inches="tight")
                if args.angles and getattr(ax3d, "name", "") == "3d":
                    for angle in args.angles:
                        ax3d.view_init(elev=30, azim=angle)
                        fig3d.savefig(
                            out_dir / f"{stem}-{env_tag}-{metric}-3d-mds-azim-{int(angle):03d}.png",
                            dpi=180,
                            bbox_inches="tight",
                        )
                if args.interactive_mpl:
                    plt.show(block=True)
                plt.close(fig3d)

            if args.interactive_html:
                ok = _write_interactive_html(
                    matrix,
                    env,
                    title=title_prefix,
                    out_html=out_dir / f"{stem}-{env_tag}-{metric}-3d-mds.html",
                    node_opacity=max(0.0, min(1.0, args.node_opacity)),
                    include_state_legend=not args.no_html_state_legend,
                    include_grid_legend_panel=not args.no_html_grid_legend_panel,
                    metadata={
                        "shape": _parse_tag(collated, "shape") or "NA",
                        "env_id": env_id or "unknown",
                        "neighbourhood": _parse_tag(collated, "neighbourhood") or "NA",
                        "state_dist": _parse_tag(collated, "state_dist") or "NA",
                        "det": _parse_tag(collated, "det") or "NA",
                        "beta": _parse_tag(collated, "beta") or "NA",
                        "eps_param": _fmt_float(eps_param),
                        "eps_actual": _fmt_float(eps_actual),
                        "seed": seed or "NA",
                        "metric": metric,
                        "run_id": _parse_tag(collated, "run_id") or "NA",
                        "run_type": str(provenance.get("run_type", "NA")),
                        "pipeline_family": str(provenance.get("pipeline_family", "NA")),
                        "source_run_id": str(provenance.get("source_run_id", "NA")),
                        "sigma_generated_at_utc": _fmt_timestamp_compact(provenance.get("sigma_generated_at_utc")),
                        "sigma_hash": str(provenance.get("sigma_hash", "NA")),
                        "n_actions": orientation_meta.get("n_actions", "NA"),
                        "chi_align": orientation_meta.get("chi_align", "NA"),
                        "chi_twist": orientation_meta.get("chi_twist", "NA"),
                        "predominant_ordering": orientation_meta.get("predominant_ordering", "NA"),
                        "mean_info": _fmt_float(means.get("mean_info")),
                        "mean_value": _fmt_float(means.get("mean_value")),
                        "mean_free": _fmt_float(means.get("mean_free")),
                    },
                )
                if not ok:
                    print("[WARN] plotly not installed; skipped interactive HTML export.")

        processed += 1
        print(f"[OK] {collated}")
        if args.limit > 0 and processed >= args.limit:
            break

    print(f"done: processed={processed} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
