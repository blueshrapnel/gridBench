from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_SCRIPT_CANDIDATES = (
    Path("/home/karen/phd-marlyn/gridTwist/scripts/run_commands/generate_shallow_run_report.py"),
    Path("/home/karen/Dropbox/phd/gridTwist/scripts/run_commands/generate_shallow_run_report.py"),
)
DEFAULT_FRONTIER_BETAS: tuple[float, ...] = (
    100.0,
    55.5047307785,
    30.8077513879,
    17.0997594668,
    9.4911754558,
    5.26805138445,
    2.92401773821,
    1.62296817351,
    0.900824115327,
    0.5,
)
DEFAULT_FRONTIER_MAX_GOALS = 24
DEFAULT_HEX_GRIDSIZE = 24

_REQUIRED_EXPORT_SYMBOLS = (
    "_write_action_alignment_frontier",
    "_write_free_chi_di_scatter",
    "_write_free_chi_2d_hexbin_density",
)

PAPER_FREE_CHI_XLIM = (9.6, 13.9)
PAPER_FREE_CHI_YLIM = (-0.04, 0.74)
PAPER_FREE_CHI_XTICKS = np.arange(10.0, 13.5 + 1e-9, 0.5)
PAPER_FREE_CHI_YTICKS = np.arange(0.0, 0.7 + 1e-9, 0.1)
REFERENCE_MARKER_COLOR = "#111111"
REFERENCE_MARKER_LINEWIDTH = 0.8
REFERENCE_MARKER_DOT_SIZE = 6.0
REFERENCE_MARKER_DOT_ALPHA = 1.0
REFERENCE_MARKER_LINE_ALPHA = 1.0


def _load_module_from_path(script_path: Path) -> ModuleType:
    module_name = f"alife_shallow_report_{abs(hash(script_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_shallow_report_module(script_candidates: Iterable[Path] = DEFAULT_SCRIPT_CANDIDATES) -> tuple[ModuleType, Path]:
    errors: list[str] = []
    for script_path in script_candidates:
        path = Path(script_path).expanduser().resolve()
        if not path.exists():
            errors.append(f"missing:{path}")
            continue
        try:
            module = _load_module_from_path(path)
        except Exception as exc:  # pragma: no cover - surfaced to notebook
            errors.append(f"import_failed:{path}:{exc}")
            continue
        if all(hasattr(module, name) for name in _REQUIRED_EXPORT_SYMBOLS):
            return module, path
        missing = [name for name in _REQUIRED_EXPORT_SYMBOLS if not hasattr(module, name)]
        errors.append(f"missing_symbols:{path}:{','.join(missing)}")
    raise RuntimeError("No compatible shallow report generator found. " + " | ".join(errors))


def discover_run_summary_path(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    matches = sorted(run_dir.glob("*.summary.json"))
    if not matches:
        raise FileNotFoundError(f"No *.summary.json found in {run_dir}")
    return matches[-1]


def load_run_summary(run_dir: str | Path) -> dict:
    summary_path = discover_run_summary_path(run_dir)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_sigma_aggregates(run_dir: str | Path) -> pd.DataFrame:
    run_dir = Path(run_dir).expanduser().resolve()
    csv_path = run_dir / "sigma_aggregates.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing sigma_aggregates.csv under {run_dir}")
    df = pd.read_csv(csv_path)
    for col in ("mean_free", "chi_twist", "mean_info", "epsilon_actual"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_pooled_sigma_aggregates(run_dirs: Sequence[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir_raw in run_dirs:
        run_dir = Path(run_dir_raw).expanduser().resolve()
        frame = load_sigma_aggregates(run_dir).copy()
        frame["source_run_dir"] = str(run_dir)
        frame["source_run_name"] = _run_name_for_dir(run_dir)
        frames.append(frame)
    if not frames:
        raise ValueError("run_dirs must contain at least one run directory")
    return pd.concat(frames, ignore_index=True)


def _finite_frontier_rows(df: pd.DataFrame, *, require_mean_info: bool) -> pd.DataFrame:
    required = ["mean_free", "chi_twist"]
    if require_mean_info:
        required.append("mean_info")
    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"sigma_aggregates missing required column: {col}")
    mask = np.isfinite(df["mean_free"]) & np.isfinite(df["chi_twist"])
    if require_mean_info:
        mask &= np.isfinite(df["mean_info"])
    cleaned = df[mask].copy()
    if cleaned.empty:
        raise RuntimeError("No finite frontier rows available")
    return cleaned


def sample_sigma_cloud(
    df: pd.DataFrame,
    *,
    n_rows: int,
    seed: int = 0,
    dedupe_by: str = "sigma_hash",
) -> pd.DataFrame:
    sampled = df.copy()
    if dedupe_by and dedupe_by in sampled.columns:
        sampled = sampled.drop_duplicates(subset=[dedupe_by]).reset_index(drop=True)
    if len(sampled) < int(n_rows):
        raise ValueError(f"Requested {n_rows} rows but only {len(sampled)} are available after filtering")
    sampled = sampled.sample(n=int(n_rows), replace=False, random_state=int(seed)).reset_index(drop=True)
    return sampled


def _run_name_for_dir(run_dir: Path) -> str:
    return run_dir.name.replace("run_name=", "")


def cartesian_baseline_cache_path(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    return run_dir / "frames" / "analysis" / "cartesian_label_baseline_reference.json"


def load_or_compute_cartesian_baseline_reference(
    *,
    run_dir: str | Path,
    gridfour_src: str | Path,
    script_candidates: Iterable[Path] = DEFAULT_SCRIPT_CANDIDATES,
) -> dict[str, object]:
    run_dir = Path(run_dir).expanduser().resolve()
    cache_path = cartesian_baseline_cache_path(run_dir)
    summary = load_run_summary(run_dir)
    goals_raw = summary.get("goals", [])
    goals = [int(g) for g in goals_raw] if isinstance(goals_raw, list) else []
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        required = {"expected_free", "expected_di", "expected_value", "sigma_hash", "goal_scope"}
        if required.issubset(cached.keys()) and str(cached.get("goal_scope", "")) == "all":
            return cached

    module, _ = load_shallow_report_module(script_candidates)
    shape_raw = summary.get("config", {}).get("shape", [7, 7]) if isinstance(summary.get("config"), dict) else [7, 7]
    n_states = int(shape_raw[0]) * int(shape_raw[1])
    identity_sigma = np.tile(np.arange(4, dtype=int), (n_states, 1))
    cfg_eval = module._build_eval_cfg_from_summary(summary)
    if goals:
        infos: list[float] = []
        frees: list[float] = []
        values: list[float] = []
        sigma_hash: str | None = None
        for goal in goals:
            result = module.evaluate_sigma(
                identity_sigma,
                replace(cfg_eval, goal=int(goal)),
                gridfour_src=str(Path(gridfour_src).expanduser().resolve()),
            )
            if not bool(result.get("ok", False)):
                continue
            di = float(result.get("expected_di", np.nan))
            free = float(result.get("expected_free", np.nan))
            value = float(result.get("expected_value", np.nan))
            if not (np.isfinite(di) and np.isfinite(free) and np.isfinite(value)):
                continue
            infos.append(di)
            frees.append(free)
            values.append(value)
            if sigma_hash is None:
                sigma_hash = str(result.get("sigma_hash", ""))
        payload = {
            "kind": "cartesian_label_baseline",
            "sigma_family": "identity",
            "goal_scope": "all",
            "expected_free": float(np.mean(frees)) if frees else float("nan"),
            "expected_di": float(np.mean(infos)) if infos else float("nan"),
            "expected_value": float(np.mean(values)) if values else float("nan"),
            "sigma_hash": str(sigma_hash or ""),
            "run_name": _run_name_for_dir(run_dir),
            "beta": float(summary.get("config", {}).get("beta", 1.0)),
            "determinism": float(summary.get("config", {}).get("determinism", 1.0)),
            "state_dist": str(summary.get("config", {}).get("state_dist", "uniform")),
            "env_id": str(summary.get("config", {}).get("env_id", "")),
            "shape": [int(shape_raw[0]), int(shape_raw[1])],
            "goal_count": int(len(goals)),
        }
    else:
        result = module.evaluate_sigma(
            identity_sigma,
            cfg_eval,
            gridfour_src=str(Path(gridfour_src).expanduser().resolve()),
        )
        payload = {
            "kind": "cartesian_label_baseline",
            "sigma_family": "identity",
            "goal_scope": "single_fallback",
            "expected_free": float(result["expected_free"]),
            "expected_di": float(result["expected_di"]),
            "expected_value": float(result["expected_value"]),
            "sigma_hash": str(result["sigma_hash"]),
            "run_name": _run_name_for_dir(run_dir),
            "beta": float(summary.get("config", {}).get("beta", 1.0)),
            "determinism": float(summary.get("config", {}).get("determinism", 1.0)),
            "state_dist": str(summary.get("config", {}).get("state_dist", "uniform")),
            "env_id": str(summary.get("config", {}).get("env_id", "")),
            "shape": [int(shape_raw[0]), int(shape_raw[1])],
            "goal_count": int(len(summary.get("goals", []))),
        }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _paper_axis_spec() -> tuple[tuple[float, float], tuple[float, float], np.ndarray, np.ndarray]:
    return PAPER_FREE_CHI_XLIM, PAPER_FREE_CHI_YLIM, PAPER_FREE_CHI_XTICKS, PAPER_FREE_CHI_YTICKS


def _style_colorbar_for_export(cbar) -> None:
    solids = getattr(cbar, "solids", None)
    if solids is not None:
        try:
            solids.set_edgecolor("face")
            solids.set_linewidth(0.0)
            solids.set_rasterized(True)
        except Exception:
            pass
    outline = getattr(cbar, "outline", None)
    if outline is not None:
        try:
            outline.set_linewidth(0.6)
        except Exception:
            pass


def _normalize_free_chi_axes(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, tuple[float, float], tuple[float, float], np.ndarray, np.ndarray, float, float]:
    xlim, ylim, xticks, yticks = _paper_axis_spec()
    x_raw = np.asarray(df["mean_free"], dtype=float)
    y_raw = np.asarray(df["chi_twist"], dtype=float)
    x_span = float(xlim[1] - xlim[0])
    y_span = float(ylim[1] - ylim[0])
    x = (x_raw - float(xlim[0])) / x_span
    y = (y_raw - float(ylim[0])) / y_span
    return x, y, xlim, ylim, xticks, yticks, x_span, y_span


def _draw_random_density_hexbin(ax, x: np.ndarray, y: np.ndarray, *, gridsize: int = DEFAULT_HEX_GRIDSIZE, cmap: str = "Oranges", alpha: float = 0.55, mincnt: int = 1, zorder: int = 1):
    hb = ax.hexbin(
        x,
        y,
        gridsize=int(max(6, gridsize)),
        mincnt=int(mincnt),
        extent=[0.0, 1.0, 0.0, 1.0],
        cmap=cmap,
        linewidths=0.0,
        alpha=float(alpha),
        zorder=int(zorder),
    )
    counts = np.asarray(hb.get_array(), dtype=float)
    if counts.size:
        log_counts = np.log1p(np.maximum(counts, 0.0))
        hb.set_array(log_counts)
        vmin = float(np.nanmin(log_counts))
        vmax = float(np.nanmax(log_counts))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1e-9
        hb.set_clim(vmin=vmin, vmax=vmax)
        try:
            hb.set_rasterized(True)
        except Exception:
            pass
    return hb


def _normalize_xy_point(x: float, y: float, *, xlim: tuple[float, float], ylim: tuple[float, float]) -> tuple[float, float]:
    x_span = float(xlim[1] - xlim[0])
    y_span = float(ylim[1] - ylim[0])
    return (float(x) - float(xlim[0])) / x_span, (float(y) - float(ylim[0])) / y_span


def _draw_reference_markers(
    ax,
    *,
    best_xy: tuple[float, float],
    baseline_xy: tuple[float, float] | None = None,
    chi0_xy: tuple[float, float] | None = None,
    random_xy: tuple[float, float] | None = None,
    line_width: float = REFERENCE_MARKER_LINEWIDTH,
    star_size: float = 200.0,
    ref_size: float = 92.0,
    baseline_size: float = 110.0,
    baseline_dot_size: float = REFERENCE_MARKER_DOT_SIZE,
    zorder_base: int = 5,
    clip_on: bool = False,
) -> None:
    best_x, best_y = best_xy
    ax.scatter(
        [best_x],
        [best_y],
        marker="*",
        s=star_size,
        facecolors="none",
        edgecolors=REFERENCE_MARKER_COLOR,
        alpha=REFERENCE_MARKER_LINE_ALPHA,
        linewidths=line_width,
        zorder=zorder_base,
        clip_on=clip_on,
    )
    ax.scatter(
        [best_x],
        [best_y],
        marker="o",
        s=baseline_dot_size,
        color=REFERENCE_MARKER_COLOR,
        alpha=REFERENCE_MARKER_DOT_ALPHA,
        linewidths=0.0,
        zorder=zorder_base + 1,
        clip_on=clip_on,
    )
    if chi0_xy is not None:
        cx, cy = chi0_xy
        ax.scatter(
            [cx],
            [cy],
            marker="o",
            s=ref_size,
            facecolors="none",
            edgecolors=REFERENCE_MARKER_COLOR,
            alpha=REFERENCE_MARKER_LINE_ALPHA,
            linewidths=line_width * 1.4,
            zorder=zorder_base + 2,
            clip_on=clip_on,
        )
        ax.scatter(
            [cx],
            [cy],
            marker="o",
            s=baseline_dot_size,
            color=REFERENCE_MARKER_COLOR,
            alpha=REFERENCE_MARKER_DOT_ALPHA,
            linewidths=0.0,
            zorder=zorder_base + 3,
            clip_on=clip_on,
        )
    if random_xy is not None:
        rx, ry = random_xy
        ax.scatter(
            [rx],
            [ry],
            marker="D",
            s=ref_size,
            facecolors="none",
            edgecolors=REFERENCE_MARKER_COLOR,
            alpha=REFERENCE_MARKER_LINE_ALPHA,
            linewidths=line_width * 1.4,
            zorder=zorder_base + 2,
            clip_on=clip_on,
        )
        ax.scatter(
            [rx],
            [ry],
            marker="o",
            s=baseline_dot_size,
            color=REFERENCE_MARKER_COLOR,
            alpha=REFERENCE_MARKER_DOT_ALPHA,
            linewidths=0.0,
            zorder=zorder_base + 3,
            clip_on=clip_on,
        )
    if baseline_xy is not None:
        base_x, base_y = baseline_xy
        ax.scatter(
            [base_x],
            [base_y],
            marker="s",
            s=baseline_size,
            facecolors="none",
            edgecolors=REFERENCE_MARKER_COLOR,
            alpha=REFERENCE_MARKER_LINE_ALPHA,
            linewidths=line_width,
            zorder=zorder_base + 4,
            clip_on=clip_on,
        )
        ax.scatter(
            [base_x],
            [base_y],
            marker="o",
            s=baseline_dot_size,
            color=REFERENCE_MARKER_COLOR,
            alpha=REFERENCE_MARKER_DOT_ALPHA,
            linewidths=0.0,
            zorder=zorder_base + 5,
            clip_on=clip_on,
        )


def _reference_marker_legend_handle(
    marker: str,
    *,
    markersize: float = 9.0,
    dot_markersize: float = 2.2,
    line_width: float = REFERENCE_MARKER_LINEWIDTH,
):
    from matplotlib.lines import Line2D

    outer = Line2D(
        [],
        [],
        linestyle="None",
        marker=marker,
        markersize=markersize,
        markerfacecolor="none",
        markeredgecolor=REFERENCE_MARKER_COLOR,
        markeredgewidth=line_width,
        color=REFERENCE_MARKER_COLOR,
        alpha=REFERENCE_MARKER_LINE_ALPHA,
    )
    dot = Line2D(
        [],
        [],
        linestyle="None",
        marker="o",
        markersize=dot_markersize,
        markerfacecolor=REFERENCE_MARKER_COLOR,
        markeredgecolor=REFERENCE_MARKER_COLOR,
        color=REFERENCE_MARKER_COLOR,
        alpha=REFERENCE_MARKER_DOT_ALPHA,
    )
    return (outer, dot)


def _format_action_order_for_sigma(run_dir: str | Path, sigma_hash: str) -> str:
    run_dir = Path(run_dir).expanduser().resolve()
    matches = list(run_dir.glob(f"hive_sigma/**/sigma_id={sigma_hash}/sigma.npy"))
    if not matches:
        return "[?, ?, ?, ?]"
    sigma = np.load(matches[0])
    if sigma.ndim != 2 or sigma.shape[1] < 4:
        return "[?, ?, ?, ?]"
    nonterminal_rows = sigma[np.all(sigma[:, :4] < 4, axis=1), :4]
    if len(nonterminal_rows) == 0:
        return "[?, ?, ?, ?]"
    rows, counts = np.unique(nonterminal_rows, axis=0, return_counts=True)
    dominant = rows[int(np.argmax(counts))]
    labels = np.array(["N", "E", "S", "W"], dtype=object)
    try:
        mapped = [str(labels[int(v)]) for v in dominant]
    except Exception:
        return "[?, ?, ?, ?]"
    return "[" + ", ".join(mapped) + "]"


def _resolve_di_limits(module: ModuleType, summary: dict, df: pd.DataFrame, search_root: Path) -> tuple[float, float, str]:
    cfg = summary.get("config", {}) if isinstance(summary.get("config"), dict) else {}
    env_id = str(cfg.get("env_id", "")).strip()
    beta_value = cfg.get("beta")
    combo_lims = None
    if env_id and hasattr(module, "_collect_di_limits_for_env_beta"):
        combo_lims = module._collect_di_limits_for_env_beta(search_root, env_id, beta_value)
    if combo_lims is None:
        vmin = float(np.nanmin(df["mean_info"]))
        vmax = float(np.nanmax(df["mean_info"]))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1e-9
        return vmin, vmax, "run_local"
    vmin, vmax = combo_lims
    return float(vmin), float(vmax), "env_beta_combo"


def export_action_alignment_frontier(
    *,
    run_dir: str | Path,
    out_base: str | Path,
    csv_path: str | Path | None = None,
    search_root: str | Path,
    gridfour_src: str | Path,
    betas: Sequence[float] = DEFAULT_FRONTIER_BETAS,
    max_goals: int = DEFAULT_FRONTIER_MAX_GOALS,
    script_candidates: Iterable[Path] = DEFAULT_SCRIPT_CANDIDATES,
) -> dict[str, Path]:
    module, script_path = load_shallow_report_module(script_candidates)
    run_dir = Path(run_dir).expanduser().resolve()
    out_base = Path(out_base).expanduser().resolve()
    summary = load_run_summary(run_dir)
    run_name = _run_name_for_dir(run_dir)
    csv_path = Path(csv_path).expanduser().resolve() if csv_path is not None else out_base.with_suffix(".csv")

    saved: dict[str, Path] = {"csv": csv_path, "script": script_path}
    for fmt in ("png", "pdf"):
        out_path = out_base.with_suffix(f".{fmt}")
        ok = module._write_action_alignment_frontier(
            run_dir,
            out_path,
            csv_path,
            summary=summary,
            run_name=run_name,
            search_root=Path(search_root).expanduser().resolve(),
            gridfour_src=Path(gridfour_src).expanduser().resolve(),
            betas=tuple(float(b) for b in betas),
            max_goals=int(max_goals),
        )
        if not ok:
            raise RuntimeError(f"Failed to export action frontier to {out_path}")
        saved[fmt] = out_path
    return saved


def export_paper_action_alignment_frontier(
    *,
    run_dir: str | Path,
    out_base: str | Path,
    csv_path: str | Path | None = None,
    search_root: str | Path,
    gridfour_src: str | Path,
    betas: Sequence[float] = DEFAULT_FRONTIER_BETAS,
    max_goals: int = DEFAULT_FRONTIER_MAX_GOALS,
    script_candidates: Iterable[Path] = DEFAULT_SCRIPT_CANDIDATES,
) -> dict[str, Path]:
    # First ensure the expensive frontier data exists in the shared cache and local CSV.
    saved = export_action_alignment_frontier(
        run_dir=run_dir,
        out_base=out_base,
        csv_path=csv_path,
        search_root=search_root,
        gridfour_src=gridfour_src,
        betas=betas,
        max_goals=max_goals,
        script_candidates=script_candidates,
    )

    module, script_path = load_shallow_report_module(script_candidates)
    csv_path = Path(saved["csv"]).expanduser().resolve()
    out_base = Path(out_base).expanduser().resolve()
    curve = pd.read_csv(csv_path)
    if curve.empty:
        raise RuntimeError(f"No tradeoff rows found in {csv_path}")

    import matplotlib

    matplotlib.use("Agg")
    if hasattr(module, "_apply_daily_plot_style"):
        module._apply_daily_plot_style(matplotlib)
    import matplotlib.pyplot as plt

    def _draw_curve_ribbon(
        ax,
        *,
        xvals: np.ndarray,
        y_lo: np.ndarray,
        y_hi: np.ndarray,
        color: str,
        alpha: float,
        label: str,
    ) -> None:
        valid = np.isfinite(xvals) & np.isfinite(y_lo) & np.isfinite(y_hi)
        if not bool(np.any(valid)):
            return
        xv = np.asarray(xvals[valid], dtype=float)
        lo = np.asarray(y_lo[valid], dtype=float)
        hi = np.asarray(y_hi[valid], dtype=float)
        order = np.argsort(xv)
        xv = xv[order]
        lo = lo[order]
        hi = hi[order]
        if xv.size == 1:
            dx = 0.04
            ax.fill_between(
                [xv[0] - dx, xv[0] + dx],
                [lo[0], lo[0]],
                [hi[0], hi[0]],
                color=color,
                alpha=alpha,
                linewidth=0.0,
                label=label,
                zorder=1,
            )
            return
        ax.fill_between(
            xv,
            lo,
            hi,
            color=color,
            alpha=alpha,
            linewidth=0.0,
            label=label,
            zorder=1,
        )

    xb = np.asarray(curve["mean_info_baseline"], dtype=float)
    yb = np.asarray(curve["mean_value_baseline"], dtype=float)
    bq25 = np.asarray(curve["value_q25_baseline"], dtype=float)
    bq75 = np.asarray(curve["value_q75_baseline"], dtype=float)
    bmin = np.asarray(curve["value_min_baseline"], dtype=float)
    bmax = np.asarray(curve["value_max_baseline"], dtype=float)
    xt = np.asarray(curve["mean_info_twist"], dtype=float)
    yt = np.asarray(curve["mean_value_twist"], dtype=float)
    tq25 = np.asarray(curve["value_q25_twist"], dtype=float)
    tq75 = np.asarray(curve["value_q75_twist"], dtype=float)
    tmin = np.asarray(curve["value_min_twist"], dtype=float)
    tmax = np.asarray(curve["value_max_twist"], dtype=float)

    if xb.size:
        order = np.argsort(xb)
        xb, yb, bq25, bq75, bmin, bmax = xb[order], yb[order], bq25[order], bq75[order], bmin[order], bmax[order]
    if xt.size:
        order = np.argsort(xt)
        xt, yt, tq25, tq75, tmin, tmax = xt[order], yt[order], tq25[order], tq75[order], tmin[order], tmax[order]

    for fmt in ("png", "pdf"):
        out_path = out_base.with_suffix(f".{fmt}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7.2, 5.4))

        if xb.size:
            ax.plot(
                xb,
                yb,
                linestyle="--",
                linewidth=1.1,
                color="#4d4d4d",
                marker="o",
                markersize=6.6,
                markerfacecolor="none",
                markeredgecolor="#4d4d4d",
                markeredgewidth=0.7,
                label=r"$\chi=0$ mean",
            )
            _draw_curve_ribbon(
                ax,
                xvals=xb,
                y_lo=bmin,
                y_hi=bmax,
                color="#7f7f7f",
                alpha=0.14,
                label=r"$\chi=0$ goal min-max",
            )
            ax.plot(
                xb,
                bq25,
                linestyle=(0, (1.2, 1.8)),
                linewidth=1.0,
                color="#666666",
                alpha=0.95,
                label=r"$\chi=0$ IQR (q25/q75)",
                zorder=2,
            )
            ax.plot(
                xb,
                bq75,
                linestyle=(0, (1.2, 1.8)),
                linewidth=1.0,
                color="#666666",
                alpha=0.95,
                zorder=2,
            )
        if xt.size:
            ax.plot(
                xt,
                yt,
                linestyle="-",
                linewidth=1.2,
                color="#d55e00",
                marker="*",
                markersize=9.0,
                markerfacecolor="none",
                markeredgecolor="#d55e00",
                markeredgewidth=0.7,
                label=r"Best $\sigma$ mean",
            )
            _draw_curve_ribbon(
                ax,
                xvals=xt,
                y_lo=tmin,
                y_hi=tmax,
                color="#d55e00",
                alpha=0.22,
                label=r"Best $\sigma$ goal min-max",
            )
            ax.plot(
                xt,
                tq25,
                linestyle=(0, (1.2, 1.8)),
                linewidth=1.0,
                color="#b64a00",
                alpha=0.95,
                label=r"Best $\sigma$ IQR (q25/q75)",
                zorder=2,
            )
            ax.plot(
                xt,
                tq75,
                linestyle=(0, (1.2, 1.8)),
                linewidth=1.0,
                color="#b64a00",
                alpha=0.95,
                zorder=2,
            )

        all_x = np.concatenate([xb, xt]) if xb.size or xt.size else np.array([], dtype=float)
        all_y = (
            np.concatenate([yb, yt, bmin, bmax, bq25, bq75, tmin, tmax, tq25, tq75])
            if xb.size or xt.size
            else np.array([], dtype=float)
        )
        if all_x.size and all_y.size:
            finite_x = all_x[np.isfinite(all_x)]
            finite_y = all_y[np.isfinite(all_y)]
            if finite_x.size and finite_y.size:
                xmin, xmax = float(np.nanmin(finite_x)), float(np.nanmax(finite_x))
                ymin, ymax = float(np.nanmin(finite_y)), float(np.nanmax(finite_y))
                xpad = 0.05 * max(xmax - xmin, 1e-9)
                ypad = 0.06 * max(ymax - ymin, 1e-9)
                ax.set_xlim(xmin - xpad, xmax + xpad)
                ax.set_ylim(ymin - ypad, min(0.0, ymax + ypad))

        ax.set_xlabel(module._LABEL_DI_EXPECTED)
        ax.set_ylabel(module._LABEL_VALUE_EXPECTED)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

        fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        saved[fmt] = out_path

    saved["script"] = script_path
    return saved


def export_free_chi_di_scatter(
    *,
    run_dir: str | Path,
    out_base: str | Path,
    search_root: str | Path,
    script_candidates: Iterable[Path] = DEFAULT_SCRIPT_CANDIDATES,
) -> dict[str, Path]:
    module, script_path = load_shallow_report_module(script_candidates)
    run_dir = Path(run_dir).expanduser().resolve()
    out_base = Path(out_base).expanduser().resolve()
    summary = load_run_summary(run_dir)
    run_name = _run_name_for_dir(run_dir)

    saved: dict[str, Path] = {"script": script_path}
    for fmt in ("png", "pdf"):
        out_path = out_base.with_suffix(f".{fmt}")
        ok = module._write_free_chi_di_scatter(
            run_dir,
            out_path,
            summary=summary,
            run_name=run_name,
            search_root=Path(search_root).expanduser().resolve(),
        )
        if not ok:
            raise RuntimeError(f"Failed to export free/chi/info scatter to {out_path}")
        saved[fmt] = out_path
    return saved


def export_free_chi_2d_hexbin_density(
    *,
    run_dir: str | Path,
    out_base: str | Path,
    csv_path: str | Path | None = None,
    hex_gridsize: int = DEFAULT_HEX_GRIDSIZE,
    script_candidates: Iterable[Path] = DEFAULT_SCRIPT_CANDIDATES,
) -> dict[str, Path]:
    module, script_path = load_shallow_report_module(script_candidates)
    run_dir = Path(run_dir).expanduser().resolve()
    out_base = Path(out_base).expanduser().resolve()
    summary = load_run_summary(run_dir)
    run_name = _run_name_for_dir(run_dir)
    csv_path = Path(csv_path).expanduser().resolve() if csv_path is not None else out_base.with_suffix(".csv")

    saved: dict[str, Path] = {"csv": csv_path, "script": script_path}
    for fmt in ("png", "pdf"):
        out_path = out_base.with_suffix(f".{fmt}")
        ok = module._write_free_chi_2d_hexbin_density(
            run_dir,
            out_path,
            csv_path,
            summary=summary,
            run_name=run_name,
            hex_gridsize=int(hex_gridsize),
        )
        if not ok:
            raise RuntimeError(f"Failed to export free/chi 2d hexbin to {out_path}")
        saved[fmt] = out_path
    return saved


def export_paper_free_chi_di_scatter(
    *,
    run_dir: str | Path,
    out_base: str | Path,
    search_root: str | Path,
    gridfour_src: str | Path,
    random_reference_csv: str | Path | None = None,
    script_candidates: Iterable[Path] = DEFAULT_SCRIPT_CANDIDATES,
) -> dict[str, object]:
    module, script_path = load_shallow_report_module(script_candidates)
    run_dir = Path(run_dir).expanduser().resolve()
    out_base = Path(out_base).expanduser().resolve()
    search_root = Path(search_root).expanduser().resolve()
    random_reference_csv = Path(random_reference_csv).expanduser().resolve() if random_reference_csv is not None else None
    summary = load_run_summary(run_dir)

    df = load_sigma_aggregates(run_dir)
    required = ("mean_free", "chi_twist", "mean_info")
    if any(c not in df.columns for c in required):
        raise RuntimeError(f"sigma_aggregates missing required columns: {required}")
    df = df[np.isfinite(df["mean_free"]) & np.isfinite(df["chi_twist"]) & np.isfinite(df["mean_info"])].copy()
    if df.empty:
        raise RuntimeError("No finite free/chi/info rows available")

    best_idx = int(df["mean_free"].idxmin())
    best_row = df.loc[best_idx]
    best_x = float(best_row["mean_free"])
    best_y = float(best_row["chi_twist"])
    best_sigma = str(best_row.get("sigma_hash", ""))

    identity_result = load_or_compute_cartesian_baseline_reference(
        run_dir=run_dir,
        gridfour_src=gridfour_src,
        script_candidates=script_candidates,
    )
    identity_x = float(identity_result["expected_free"])
    identity_y = 0.0

    chi0 = df[np.isclose(df["chi_twist"], 0.0)].sort_values("mean_free").head(1)
    chi0_x = None
    chi0_sigma = ""
    if not chi0.empty:
        chi0_x = float(chi0.iloc[0]["mean_free"])
        chi0_sigma = str(chi0.iloc[0].get("sigma_hash", ""))
    chi0_order_label = _format_action_order_for_sigma(run_dir, chi0_sigma) if chi0_sigma else "[?, ?, ?, ?]"

    random_ref_x = None
    random_ref_y = None
    random_ref_sigma = ""
    if random_reference_csv is not None and random_reference_csv.exists():
        random_ref_df = pd.read_csv(random_reference_csv)
        if {"mean_free", "chi_twist"}.issubset(random_ref_df.columns):
            random_ref_df["mean_free"] = pd.to_numeric(random_ref_df["mean_free"], errors="coerce")
            random_ref_df["chi_twist"] = pd.to_numeric(random_ref_df["chi_twist"], errors="coerce")
            random_ref_df = random_ref_df[np.isfinite(random_ref_df["mean_free"]) & np.isfinite(random_ref_df["chi_twist"])].copy()
            if not random_ref_df.empty:
                random_ref_row = random_ref_df.iloc[0]
                random_ref_x = float(random_ref_row["mean_free"])
                random_ref_y = float(random_ref_row["chi_twist"])
                random_ref_sigma = str(random_ref_row.get("sigma_hash", ""))

    vmin, vmax, lim_source = _resolve_di_limits(module, summary, df, search_root)

    import matplotlib

    matplotlib.use("Agg")
    if hasattr(module, "_apply_daily_plot_style"):
        module._apply_daily_plot_style(matplotlib)
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.patches import Patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    xlim, ylim, xticks, yticks = _paper_axis_spec()
    saved: dict[str, object] = {
        "script": script_path,
        "best_sigma": best_sigma,
        "baseline_mean_free": identity_x,
        "baseline_cache": cartesian_baseline_cache_path(run_dir),
    }
    if chi0_x is not None:
        saved["chi0_mean_free"] = chi0_x
        saved["chi0_sigma"] = chi0_sigma
    if random_ref_x is not None:
        saved["random_reference_sigma"] = random_ref_sigma
        saved["random_reference_mean_free"] = random_ref_x
        saved["random_reference_chi_twist"] = random_ref_y
    for fmt in ("png", "pdf"):
        out_path = out_base.with_suffix(f".{fmt}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7.0, 5.8))
        fig.subplots_adjust(left=0.12, right=0.88, bottom=0.10, top=0.98)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.8%", pad=0.05)

        norm = Normalize(vmin=float(vmin), vmax=float(vmax))
        cmap = module.gridflask_tab10_continuous_cmap()
        sc = ax.scatter(
            df["mean_free"],
            df["chi_twist"],
            s=16,
            alpha=0.45,
            c=df["mean_info"],
            cmap=cmap,
            norm=norm,
            edgecolors="none",
        )
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label(module._LABEL_DI_MEAN)
        _style_colorbar_for_export(cbar)
        cax.yaxis.set_ticks_position("right")

        marker_size = 200
        baseline_marker_size = 110
        baseline_dot_size = REFERENCE_MARKER_DOT_SIZE
        marker_linewidth = REFERENCE_MARKER_LINEWIDTH

        _draw_reference_markers(
            ax,
            best_xy=(best_x, best_y),
            baseline_xy=None,
            chi0_xy=(chi0_x, 0.0) if chi0_x is not None else None,
            random_xy=(random_ref_x, random_ref_y) if random_ref_x is not None else None,
            line_width=marker_linewidth,
            star_size=marker_size,
            ref_size=92.0,
            baseline_size=baseline_marker_size,
            baseline_dot_size=baseline_dot_size,
            zorder_base=5,
            clip_on=False,
        )
        legend_handles = [
            Patch(facecolor="#8d79b8", edgecolor="#3c2457", linewidth=0.18, alpha=0.46, label="GA search"),
            _reference_marker_legend_handle("*", markersize=10.0, dot_markersize=1.8, line_width=REFERENCE_MARKER_LINEWIDTH),
            _reference_marker_legend_handle("o", markersize=8.0, dot_markersize=1.8, line_width=REFERENCE_MARKER_LINEWIDTH),
            _reference_marker_legend_handle("D", markersize=6.8, dot_markersize=1.8, line_width=REFERENCE_MARKER_LINEWIDTH),
        ]
        legend_labels = [
            "GA search",
            r"Best $\sigma$",
            r"$\chi=0$",
            r"Random $\sigma$",
        ]
        ax.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="upper left",
            fontsize=6.5,
            ncol=2,
            frameon=True,
            framealpha=0.9,
            columnspacing=1.0,
            handletextpad=0.6,
            borderpad=0.45,
            handler_map={tuple: HandlerTuple(ndivide=1)},
        )

        xr = float(df["mean_free"].max() - df["mean_free"].min())
        yr = float(df["chi_twist"].max() - df["chi_twist"].min())
        xpad_left = max(0.008 * xr, 0.01)
        xpad_right = max(0.055 * xr, 0.05)
        ypad = max(0.05 * yr, 0.02)
        iax = inset_axes(
            ax,
            width="24%",
            height="24%",
            loc="lower right",
            borderpad=0.8,
            bbox_to_anchor=(0.0, 0.07, 1.0, 1.0),
            bbox_transform=ax.transAxes,
        )
        iax.scatter(
            df["mean_free"],
            df["chi_twist"],
            s=10,
            alpha=0.30,
            c=df["mean_info"],
            cmap=cmap,
            norm=norm,
            edgecolors="none",
        )
        _draw_reference_markers(
            iax,
            best_xy=(best_x, best_y),
            baseline_xy=(identity_x, identity_y),
            chi0_xy=None,
            random_xy=None,
            line_width=marker_linewidth,
            star_size=marker_size,
            ref_size=92.0,
            baseline_size=baseline_marker_size,
            baseline_dot_size=baseline_dot_size,
            zorder_base=6,
            clip_on=False,
        )
        iax.set_xlim(best_x - xpad_left, best_x + xpad_right)
        iax.set_ylim(best_y - ypad, best_y + ypad)
        iax.grid(alpha=0.20)
        iax.tick_params(axis="both", labelsize=7)

        ax.set_xlabel(module._LABEL_FREE_MEAN, labelpad=10)
        ax.set_ylabel(module._LABEL_CHI_TWIST)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_box_aspect(1.0)
        ax.grid(alpha=0.25)

        if fmt == "pdf":
            from PIL import Image

            png_path = out_base.with_suffix(".png")
            if not png_path.exists():
                fig.savefig(png_path, dpi=240, bbox_inches="tight", pad_inches=0.02)
            with Image.open(png_path) as img:
                rgb = Image.new("RGB", img.size, (255, 255, 255))
                if "A" in img.getbands():
                    rgb.paste(img, mask=img.getchannel("A"))
                else:
                    rgb.paste(img)
                rgb.save(out_path, "PDF", resolution=240.0)
        else:
            fig.savefig(out_path, dpi=240, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        saved[fmt] = out_path
    return saved


def export_paper_random_diffusion_cloud(
    *,
    random_run_dirs: Sequence[str | Path],
    target_n_rows: int,
    out_base: str | Path,
    sample_csv_path: str | Path,
    search_root: str | Path,
    seed: int = 0,
    script_candidates: Iterable[Path] = DEFAULT_SCRIPT_CANDIDATES,
) -> dict[str, object]:
    module, script_path = load_shallow_report_module(script_candidates)
    out_base = Path(out_base).expanduser().resolve()
    sample_csv_path = Path(sample_csv_path).expanduser().resolve()
    search_root = Path(search_root).expanduser().resolve()

    pooled = load_pooled_sigma_aggregates(random_run_dirs)
    pooled = _finite_frontier_rows(pooled, require_mean_info=True)
    sampled = sample_sigma_cloud(pooled, n_rows=int(target_n_rows), seed=int(seed))

    sample_csv_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(sample_csv_path, index=False)

    import matplotlib

    matplotlib.use("Agg")
    if hasattr(module, "_apply_daily_plot_style"):
        module._apply_daily_plot_style(matplotlib)
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    summary_for_limits = None
    for run_dir_raw in random_run_dirs:
        try:
            summary_for_limits = load_run_summary(run_dir_raw)
            break
        except FileNotFoundError:
            continue
    if summary_for_limits is None:
        raise FileNotFoundError("Could not find any random run summary to resolve plotting limits")

    vmin, vmax, lim_source = _resolve_di_limits(module, summary_for_limits, sampled, search_root)
    x, y, xlim, ylim, xticks, yticks, _, _ = _normalize_free_chi_axes(sampled)

    saved: dict[str, object] = {
        "csv": sample_csv_path,
        "script": script_path,
        "n_rows": int(len(sampled)),
        "seed": int(seed),
        "di_lim_source": lim_source,
    }
    for fmt in ("png", "pdf"):
        out_path = out_base.with_suffix(f".{fmt}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7.0, 5.8))
        fig.subplots_adjust(left=0.12, right=0.88, bottom=0.10, top=0.98)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.8%", pad=0.05)

        _draw_random_density_hexbin(
            ax,
            x,
            y,
            gridsize=28,
            cmap="Oranges",
            alpha=0.72,
            mincnt=1,
            zorder=1,
        )
        norm = Normalize(vmin=float(vmin), vmax=float(vmax))
        cmap = module.gridflask_tab10_continuous_cmap()
        artist = ax.scatter(
            sampled["mean_free"],
            sampled["chi_twist"],
            s=7,
            alpha=0.08,
            c=sampled["mean_info"],
            cmap=cmap,
            norm=norm,
            edgecolors="none",
            zorder=2,
        )
        try:
            artist.set_rasterized(True)
        except Exception:
            pass
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cbar.set_label(module._LABEL_DI_MEAN)
        _style_colorbar_for_export(cbar)
        cax.yaxis.set_ticks_position("right")

        best_idx = int(sampled["mean_free"].idxmin())
        best_row = sampled.loc[best_idx]
        ax.scatter(
            [float(best_row["mean_free"])],
            [float(best_row["chi_twist"])],
            marker="*",
            s=200,
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.0,
            zorder=5,
        )

        ax.set_xlabel(module._LABEL_FREE_MEAN, labelpad=10)
        ax.set_ylabel(module._LABEL_CHI_TWIST)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_box_aspect(1.0)
        ax.grid(alpha=0.25)

        fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        saved[fmt] = out_path
    return saved


def export_paper_free_chi_di_scatter_with_random_cloud(
    *,
    ga_run_dir: str | Path,
    random_run_dirs: Sequence[str | Path],
    out_base: str | Path,
    sample_csv_path: str | Path,
    search_root: str | Path,
    gridfour_src: str | Path,
    seed: int = 0,
    target_n_rows: int | None = None,
    random_sigma_hash: str | None = None,
    script_candidates: Iterable[Path] = DEFAULT_SCRIPT_CANDIDATES,
) -> dict[str, object]:
    module, script_path = load_shallow_report_module(script_candidates)
    ga_run_dir = Path(ga_run_dir).expanduser().resolve()
    out_base = Path(out_base).expanduser().resolve()
    sample_csv_path = Path(sample_csv_path).expanduser().resolve()
    search_root = Path(search_root).expanduser().resolve()
    summary = load_run_summary(ga_run_dir)

    ga_df = load_sigma_aggregates(ga_run_dir)
    ga_df = _finite_frontier_rows(ga_df, require_mean_info=True)
    if target_n_rows is None:
        target_n_rows = int(len(ga_df))

    pooled_random = load_pooled_sigma_aggregates(random_run_dirs)
    pooled_random = _finite_frontier_rows(pooled_random, require_mean_info=True)
    sampled_random = sample_sigma_cloud(pooled_random, n_rows=int(target_n_rows), seed=int(seed))

    sample_csv_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_random.to_csv(sample_csv_path, index=False)
    if random_sigma_hash is not None:
        _ga_all = load_sigma_aggregates(ga_run_dir)
        _match = _ga_all[_ga_all["sigma_hash"] == random_sigma_hash]
        if not _match.empty:
            random_ref_row = _match.iloc[0]
            random_ref_x = float(random_ref_row["mean_free"])
            random_ref_y = float(random_ref_row["chi_twist"])
            random_ref_sigma = random_sigma_hash
        else:
            import warnings
            warnings.warn(f"random_sigma_hash {random_sigma_hash!r} not found in GA aggregates; using sampled_random.iloc[0]")
            random_ref_row = sampled_random.iloc[0]
            random_ref_x = float(random_ref_row["mean_free"])
            random_ref_y = float(random_ref_row["chi_twist"])
            random_ref_sigma = str(random_ref_row.get("sigma_hash", ""))
    else:
        random_ref_row = sampled_random.iloc[0]
        random_ref_x = float(random_ref_row["mean_free"])
        random_ref_y = float(random_ref_row["chi_twist"])
        random_ref_sigma = str(random_ref_row.get("sigma_hash", ""))

    best_idx = int(ga_df["mean_free"].idxmin())
    best_row = ga_df.loc[best_idx]
    best_x = float(best_row["mean_free"])
    best_y = float(best_row["chi_twist"])
    best_sigma = str(best_row.get("sigma_hash", ""))

    identity_result = load_or_compute_cartesian_baseline_reference(
        run_dir=ga_run_dir,
        gridfour_src=gridfour_src,
        script_candidates=script_candidates,
    )
    identity_x = float(identity_result["expected_free"])
    identity_y = 0.0

    chi0 = ga_df[np.isclose(ga_df["chi_twist"], 0.0)].sort_values("mean_free").head(1)
    chi0_x = None
    chi0_sigma = ""
    if not chi0.empty:
        chi0_x = float(chi0.iloc[0]["mean_free"])
        chi0_sigma = str(chi0.iloc[0].get("sigma_hash", ""))

    vmin, vmax, lim_source = _resolve_di_limits(module, summary, ga_df, search_root)

    import matplotlib

    matplotlib.use("Agg")
    if hasattr(module, "_apply_daily_plot_style"):
        module._apply_daily_plot_style(matplotlib)
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.patches import Patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    xrnd, yrnd, xlim, ylim, xticks, yticks, _, _ = _normalize_free_chi_axes(sampled_random)
    chi0_order_label = _format_action_order_for_sigma(ga_run_dir, chi0_sigma) if chi0_sigma else "[?, ?, ?, ?]"
    saved: dict[str, object] = {
        "csv": sample_csv_path,
        "script": script_path,
        "baseline_mean_free": identity_x,
        "baseline_cache": cartesian_baseline_cache_path(ga_run_dir),
        "target_n_rows": int(target_n_rows),
        "random_n_rows": int(len(sampled_random)),
        "seed": int(seed),
        "best_sigma": best_sigma,
        "di_lim_source": lim_source,
        "random_reference_sigma": random_ref_sigma,
        "random_reference_mean_free": random_ref_x,
        "random_reference_chi_twist": random_ref_y,
    }
    for fmt in ("png", "pdf"):
        out_path = out_base.with_suffix(f".{fmt}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7.0, 5.8))
        fig.subplots_adjust(left=0.12, right=0.88, bottom=0.10, top=0.98)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.8%", pad=0.05)

        _draw_random_density_hexbin(
            ax,
            xrnd,
            yrnd,
            gridsize=28,
            cmap="Oranges",
            alpha=0.52,
            mincnt=1,
            zorder=1,
        )
        ax.scatter(
            sampled_random["mean_free"],
            sampled_random["chi_twist"],
            s=7,
            alpha=0.07,
            color="#ef8c1f",
            edgecolors="none",
            zorder=1,
        )

        norm = Normalize(vmin=float(vmin), vmax=float(vmax))
        cmap = module.gridflask_tab10_continuous_cmap()
        ga_artist = ax.scatter(
            ga_df["mean_free"],
            ga_df["chi_twist"],
            s=16,
            alpha=0.45,
            c=ga_df["mean_info"],
            cmap=cmap,
            norm=norm,
            edgecolors="none",
            zorder=2,
        )
        try:
            ga_artist.set_rasterized(True)
        except Exception:
            pass
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cbar.set_label(module._LABEL_DI_MEAN)
        _style_colorbar_for_export(cbar)
        cax.yaxis.set_ticks_position("right")

        marker_size = 200
        baseline_marker_size = 110
        baseline_dot_size = REFERENCE_MARKER_DOT_SIZE
        marker_linewidth = REFERENCE_MARKER_LINEWIDTH

        _draw_reference_markers(
            ax,
            best_xy=(best_x, best_y),
            baseline_xy=None,
            chi0_xy=(chi0_x, 0.0) if chi0_x is not None else None,
            random_xy=(random_ref_x, random_ref_y),
            line_width=marker_linewidth,
            star_size=marker_size,
            ref_size=92.0,
            baseline_size=baseline_marker_size,
            baseline_dot_size=baseline_dot_size,
            zorder_base=5,
            clip_on=False,
        )
        legend_handles = [
            Patch(facecolor="#ef8c1f", edgecolor="#b56512", linewidth=0.18, alpha=0.53, label="Random search"),
            Patch(facecolor="#8d79b8", edgecolor="#3c2457", linewidth=0.18, alpha=0.46, label="GA search"),
            _reference_marker_legend_handle("*", markersize=10.0, dot_markersize=1.8, line_width=REFERENCE_MARKER_LINEWIDTH),
            _reference_marker_legend_handle("o", markersize=8.0, dot_markersize=1.8, line_width=REFERENCE_MARKER_LINEWIDTH),
            _reference_marker_legend_handle("D", markersize=8.0, dot_markersize=1.8, line_width=REFERENCE_MARKER_LINEWIDTH),
        ]
        legend_labels = [
            "Random search",
            "GA search",
            r"Best $\sigma$",
            r"$\chi=0$",
            r"Random $\sigma$",
        ]
        ax.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="upper left",
            fontsize=6.5,
            ncol=2,
            frameon=True,
            framealpha=0.9,
            columnspacing=1.0,
            handletextpad=0.6,
            borderpad=0.45,
            handler_map={tuple: HandlerTuple(ndivide=1)},
        )

        xr = float(ga_df["mean_free"].max() - ga_df["mean_free"].min())
        yr = float(ga_df["chi_twist"].max() - ga_df["chi_twist"].min())
        xpad_left = max(0.008 * xr, 0.01)
        xpad_right = max(0.055 * xr, 0.05)
        ypad = max(0.05 * yr, 0.02)
        iax = inset_axes(
            ax,
            width="24%",
            height="24%",
            loc="lower right",
            borderpad=0.8,
            bbox_to_anchor=(0.0, 0.07, 1.0, 1.0),
            bbox_transform=ax.transAxes,
        )
        _draw_random_density_hexbin(
            iax,
            xrnd,
            yrnd,
            gridsize=18,
            cmap="Oranges",
            alpha=0.45,
            mincnt=1,
            zorder=1,
        )
        ga_inset = iax.scatter(
            ga_df["mean_free"],
            ga_df["chi_twist"],
            s=10,
            alpha=0.30,
            c=ga_df["mean_info"],
            cmap=cmap,
            norm=norm,
            edgecolors="none",
            zorder=2,
        )
        try:
            ga_inset.set_rasterized(True)
        except Exception:
            pass
        _draw_reference_markers(
            iax,
            best_xy=(best_x, best_y),
            baseline_xy=(identity_x, identity_y),
            chi0_xy=(chi0_x, 0.0) if chi0_x is not None else None,
            random_xy=(random_ref_x, random_ref_y),
            line_width=marker_linewidth,
            star_size=marker_size,
            ref_size=92.0,
            baseline_size=baseline_marker_size,
            baseline_dot_size=baseline_dot_size,
            zorder_base=6,
            clip_on=False,
        )
        iax.set_xlim(best_x - xpad_left, best_x + xpad_right)
        iax.set_ylim(best_y - ypad, best_y + ypad)
        iax.grid(alpha=0.20)
        iax.tick_params(axis="both", labelsize=7)

        ax.set_xlabel(module._LABEL_FREE_MEAN, labelpad=10)
        ax.set_ylabel(module._LABEL_CHI_TWIST)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_box_aspect(1.0)
        ax.grid(alpha=0.25)

        fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        saved[fmt] = out_path
    if chi0_x is not None:
        saved["chi0_mean_free"] = chi0_x
        saved["chi0_sigma"] = chi0_sigma
    return saved


def export_paper_free_chi_2d_hexbin_density(
    *,
    run_dir: str | Path,
    out_base: str | Path,
    csv_path: str | Path | None = None,
    hex_gridsize: int = DEFAULT_HEX_GRIDSIZE,
    script_candidates: Iterable[Path] = DEFAULT_SCRIPT_CANDIDATES,
) -> dict[str, Path]:
    module, script_path = load_shallow_report_module(script_candidates)
    run_dir = Path(run_dir).expanduser().resolve()
    out_base = Path(out_base).expanduser().resolve()
    csv_path = Path(csv_path).expanduser().resolve() if csv_path is not None else out_base.with_suffix(".csv")
    summary = load_run_summary(run_dir)
    run_name = _run_name_for_dir(run_dir)

    df = load_sigma_aggregates(run_dir)
    required = ("mean_free", "chi_twist")
    if any(c not in df.columns for c in required):
        raise RuntimeError(f"sigma_aggregates missing required columns: {required}")
    df = df[np.isfinite(df["mean_free"]) & np.isfinite(df["chi_twist"])].copy()
    if df.empty:
        raise RuntimeError("No finite free/chi rows available")

    x, y, xlim, ylim, xticks, yticks, x_span, y_span = _normalize_free_chi_axes(df)

    import matplotlib

    matplotlib.use("Agg")
    if hasattr(module, "_apply_daily_plot_style"):
        module._apply_daily_plot_style(matplotlib)
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    saved: dict[str, Path] = {"csv": csv_path, "script": script_path}
    for fmt in ("png", "pdf"):
        out_path = out_base.with_suffix(f".{fmt}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7.0, 5.8))
        fig.subplots_adjust(left=0.14, right=0.88, bottom=0.10, top=0.98)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.8%", pad=0.05)

        hb = ax.hexbin(
            x,
            y,
            gridsize=int(max(6, hex_gridsize)),
            mincnt=1,
            extent=[0.0, 1.0, 0.0, 1.0],
            cmap="Purples",
            linewidths=0.0,
        )
        centers = np.asarray(hb.get_offsets(), dtype=float)
        counts = np.asarray(hb.get_array(), dtype=float)
        if centers.size == 0 or counts.size == 0:
            plt.close(fig)
            raise RuntimeError("Hexbin returned no populated cells")

        log_counts = np.log1p(np.maximum(counts, 0.0))
        hb.set_array(log_counts)
        vmin = float(np.nanmin(log_counts))
        vmax = float(np.nanmax(log_counts))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1e-9
        hb.set_clim(vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(hb, cax=cax)
        cbar.set_label("hex occupancy")
        _style_colorbar_for_export(cbar)
        if hasattr(module, "_log1p_count_tick_spec"):
            tick_pos, tick_labels = module._log1p_count_tick_spec(float(np.nanmax(counts)))
            cbar.set_ticks(tick_pos.tolist())
            cbar.set_ticklabels(tick_labels)
        cax.yaxis.set_ticks_position("right")

        ax.scatter(
            x,
            y,
            s=7,
            alpha=0.12,
            color="#111111",
            edgecolors="none",
            zorder=3,
        )
        ax.set_xlabel(module._LABEL_FREE_MEAN)
        ax.set_ylabel(module._LABEL_CHI_TWIST, labelpad=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks((xticks - float(xlim[0])) / x_span)
        ax.set_yticks((yticks - float(ylim[0])) / y_span)
        ax.set_xticklabels([f"{v:.1f}" for v in xticks])
        ax.set_yticklabels([f"{v:.1f}" for v in yticks])
        ax.grid(alpha=0.22)

        pd.DataFrame(
            {
                "hex_center_mean_free": float(xlim[0]) + centers[:, 0] * x_span,
                "hex_center_chi_twist": float(ylim[0]) + centers[:, 1] * y_span,
                "hex_count": counts,
                "hex_log1p_count": log_counts,
            }
        ).to_csv(csv_path, index=False)

        fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        saved[fmt] = out_path
    return saved


def export_paper_free_chi_2d_hexbin_density_with_random_overlay(
    *,
    ga_run_dir: str | Path,
    random_run_dirs: Sequence[str | Path],
    out_base: str | Path,
    sample_csv_path: str | Path,
    csv_path: str | Path | None = None,
    hex_gridsize: int = DEFAULT_HEX_GRIDSIZE,
    seed: int = 0,
    target_n_rows: int | None = None,
    random_sigma_hash: str | None = None,
    script_candidates: Iterable[Path] = DEFAULT_SCRIPT_CANDIDATES,
) -> dict[str, object]:
    module, script_path = load_shallow_report_module(script_candidates)
    ga_run_dir = Path(ga_run_dir).expanduser().resolve()
    out_base = Path(out_base).expanduser().resolve()
    sample_csv_path = Path(sample_csv_path).expanduser().resolve()
    csv_path = Path(csv_path).expanduser().resolve() if csv_path is not None else out_base.with_suffix(".csv")

    ga_df = _finite_frontier_rows(load_sigma_aggregates(ga_run_dir), require_mean_info=False)
    if target_n_rows is None:
        target_n_rows = int(len(ga_df))
    pooled_random = _finite_frontier_rows(load_pooled_sigma_aggregates(random_run_dirs), require_mean_info=False)
    sampled_random = sample_sigma_cloud(pooled_random, n_rows=int(target_n_rows), seed=int(seed))

    sample_csv_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_random.to_csv(sample_csv_path, index=False)
    if random_sigma_hash is not None:
        _match = ga_df[ga_df["sigma_hash"] == random_sigma_hash]
        if not _match.empty:
            random_ref_row = _match.iloc[0]
            random_ref_x = float(random_ref_row["mean_free"])
            random_ref_y = float(random_ref_row["chi_twist"])
            random_ref_sigma = random_sigma_hash
        else:
            import warnings
            warnings.warn(f"random_sigma_hash {random_sigma_hash!r} not found in GA aggregates; using sampled_random.iloc[0]")
            random_ref_row = sampled_random.iloc[0]
            random_ref_x = float(random_ref_row["mean_free"])
            random_ref_y = float(random_ref_row["chi_twist"])
            random_ref_sigma = str(random_ref_row.get("sigma_hash", ""))
    else:
        random_ref_row = sampled_random.iloc[0]
        random_ref_x = float(random_ref_row["mean_free"])
        random_ref_y = float(random_ref_row["chi_twist"])
        random_ref_sigma = str(random_ref_row.get("sigma_hash", ""))

    identity_result = load_or_compute_cartesian_baseline_reference(
        run_dir=ga_run_dir,
        gridfour_src="/home/karen/phd-marlyn/gridFour/src",
        script_candidates=script_candidates,
    )
    identity_x = float(identity_result["expected_free"])
    identity_y = 0.0

    chi0 = ga_df[np.isclose(ga_df["chi_twist"], 0.0)].sort_values("mean_free").head(1)
    chi0_x = None
    chi0_sigma = ""
    if not chi0.empty:
        chi0_x = float(chi0.iloc[0]["mean_free"])
        chi0_sigma = str(chi0.iloc[0].get("sigma_hash", ""))

    best_idx = int(ga_df["mean_free"].idxmin())
    best_row = ga_df.loc[best_idx]
    best_x = float(best_row["mean_free"])
    best_y = float(best_row["chi_twist"])
    best_sigma = str(best_row.get("sigma_hash", ""))

    x_ga, y_ga, xlim, ylim, xticks, yticks, x_span, y_span = _normalize_free_chi_axes(ga_df)
    x_r, y_r, _, _, _, _, _, _ = _normalize_free_chi_axes(sampled_random)

    import matplotlib

    matplotlib.use("Agg")
    if hasattr(module, "_apply_daily_plot_style"):
        module._apply_daily_plot_style(matplotlib)
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    chi0_order_label = _format_action_order_for_sigma(ga_run_dir, chi0_sigma) if chi0_sigma else "[?, ?, ?, ?]"

    saved: dict[str, object] = {
        "csv": csv_path,
        "sample_csv": sample_csv_path,
        "script": script_path,
        "target_n_rows": int(target_n_rows),
        "random_n_rows": int(len(sampled_random)),
        "seed": int(seed),
        "best_sigma": best_sigma,
        "baseline_mean_free": identity_x,
        "random_reference_sigma": random_ref_sigma,
        "random_reference_mean_free": random_ref_x,
        "random_reference_chi_twist": random_ref_y,
    }
    for fmt in ("png", "pdf"):
        out_path = out_base.with_suffix(f".{fmt}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7.0, 5.8))
        fig.subplots_adjust(left=0.14, right=0.86, bottom=0.10, top=0.98)

        hb_random = ax.hexbin(
            x_r,
            y_r,
            gridsize=int(max(6, hex_gridsize)),
            mincnt=1,
            extent=[0.0, 1.0, 0.0, 1.0],
            cmap="Oranges",
            linewidths=0.18,
            edgecolors="#b56512",
            alpha=0.44,
            zorder=1,
        )
        random_centers = np.asarray(hb_random.get_offsets(), dtype=float)
        random_counts = np.asarray(hb_random.get_array(), dtype=float)
        if random_centers.size == 0 or random_counts.size == 0:
            plt.close(fig)
            raise RuntimeError("Random hexbin returned no populated cells")
        # Display-only offset so the random and GA hex layers remain visually distinct.
        random_display_centers = random_centers.copy()
        random_dx = 0.18 / float(max(6, hex_gridsize))
        random_dy = -0.16 / float(max(6, hex_gridsize))
        random_display_centers[:, 0] = np.clip(random_display_centers[:, 0] + random_dx, 0.0, 1.0)
        random_display_centers[:, 1] = np.clip(random_display_centers[:, 1] + random_dy, 0.0, 1.0)
        hb_random.set_offsets(random_display_centers)

        random_log_counts = np.log1p(np.maximum(random_counts, 0.0))
        hb_random.set_array(random_log_counts)
        vmin = 0.0
        vmax = float(np.nanmax(random_log_counts))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1e-9
        hb_random.set_clim(vmin=vmin, vmax=vmax)
        try:
            hb_random.set_rasterized(True)
        except Exception:
            pass

        ax.scatter(
            x_ga,
            y_ga,
            s=8,
            alpha=0.68,
            color="steelblue",
            edgecolors="none",
            zorder=3,
        )

        legend_handles = [
            Patch(facecolor="#ef8c1f", edgecolor="#b56512", linewidth=0.18, alpha=0.44, label="Random search"),
            Line2D([], [], linestyle="None", marker="o", markersize=3.2, color="steelblue", alpha=0.85, label="GA search"),
            _reference_marker_legend_handle("*", markersize=10.0, dot_markersize=1.8, line_width=REFERENCE_MARKER_LINEWIDTH),
            _reference_marker_legend_handle("o", markersize=8.0, dot_markersize=1.8, line_width=REFERENCE_MARKER_LINEWIDTH),
            _reference_marker_legend_handle("D", markersize=6.8, dot_markersize=1.8, line_width=REFERENCE_MARKER_LINEWIDTH),
        ]
        legend_labels = [
            "Random search",
            "GA search",
            r"Best $\sigma$",
            r"$\chi=0$",
            r"Random $\sigma$",
        ]
        ax.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="upper left",
            fontsize=6.5,
            ncol=2,
            frameon=True,
            framealpha=0.9,
            columnspacing=1.0,
            handletextpad=0.6,
            borderpad=0.45,
            handler_map={tuple: HandlerTuple(ndivide=1)},
        )

        best_xy_n = _normalize_xy_point(best_x, best_y, xlim=xlim, ylim=ylim)
        base_xy_n = _normalize_xy_point(identity_x, identity_y, xlim=xlim, ylim=ylim)
        chi0_xy_n = _normalize_xy_point(chi0_x, 0.0, xlim=xlim, ylim=ylim) if chi0_x is not None else None
        random_xy_n = _normalize_xy_point(random_ref_x, random_ref_y, xlim=xlim, ylim=ylim)
        _draw_reference_markers(
            ax,
            best_xy=best_xy_n,
            baseline_xy=None,
            chi0_xy=chi0_xy_n,
            random_xy=random_xy_n,
            line_width=REFERENCE_MARKER_LINEWIDTH,
            star_size=200.0,
            ref_size=92.0,
            baseline_size=110.0,
            baseline_dot_size=REFERENCE_MARKER_DOT_SIZE,
            zorder_base=6,
            clip_on=False,
        )

        ax.set_xlabel(module._LABEL_FREE_MEAN)
        ax.set_ylabel(module._LABEL_CHI_TWIST, labelpad=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks((xticks - float(xlim[0])) / x_span)
        ax.set_yticks((yticks - float(ylim[0])) / y_span)
        ax.set_xticklabels([f"{v:.1f}" for v in xticks])
        ax.set_yticklabels([f"{v:.1f}" for v in yticks])
        ax.grid(alpha=0.22)

        fig.canvas.draw()
        bbox = ax.get_position()
        cbar_pad = 0.010
        cbar_width = 0.018
        cax_random = fig.add_axes([bbox.x1 + cbar_pad, bbox.y0, cbar_width, bbox.height])

        cbar_random = fig.colorbar(hb_random, cax=cax_random)
        cbar_random.set_label("hex occupancy")
        _style_colorbar_for_export(cbar_random)
        if hasattr(module, "_log1p_count_tick_spec"):
            tick_pos, tick_labels = module._log1p_count_tick_spec(float(np.nanmax(random_counts)))
            cbar_random.set_ticks(tick_pos.tolist())
            cbar_random.set_ticklabels(tick_labels)
        cax_random.yaxis.set_ticks_position("right")
        cax_random.tick_params(length=0)

        random_df = pd.DataFrame(
            {
                "hex_center_mean_free": float(xlim[0]) + random_centers[:, 0] * x_span,
                "hex_center_chi_twist": float(ylim[0]) + random_centers[:, 1] * y_span,
                "random_hex_count": random_counts,
                "random_hex_log1p_count": random_log_counts,
            }
        )
        random_df.sort_values(["hex_center_mean_free", "hex_center_chi_twist"]).to_csv(csv_path, index=False)

        if chi0_x is not None:
            saved["chi0_mean_free"] = chi0_x
            saved["chi0_sigma"] = chi0_sigma
        fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        saved[fmt] = out_path
    return saved
