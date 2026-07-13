"""
Pool generation-best four-rooms sigmas from GA runs and score simple room-structure motifs.

This is intentionally a proof-of-concept analysis tool. It does not write into the
grid-four schema-10 hive store yet; instead it copies the pooled sigma set into a
separate analysis directory and emits ranked CSV/JSON artifacts there.

Current motif scores are designed to be interpretable rather than final:
    - top-room attraction: for states outside the large top room, how often does a
      given label reduce graph distance to the top-room region?
    - region exit progress: for states in a region, how often does a given label
      reduce graph distance to the nearest frontier cell outside that region?

The output includes both the full per-generation-best history rows and a pooled
unique-sigma table with occurrence counts and structure scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import imageio.v2 as imageio
import networkx as nx
import numpy as np

_GRIDFOUR_SRC = Path(__file__).resolve().parents[2]
if str(_GRIDFOUR_SRC) not in sys.path:
    sys.path.insert(0, str(_GRIDFOUR_SRC))

import gridvis.display as di
from gridvis.display import plot_heatmap
from gridvis.display_twist import plot_twist_dual_landscape
from gridcore.envs import compute_four_room_walls
from gridcore.envs import GridRoom


LABELS = ("N", "E", "S", "W")


@dataclass(frozen=True)
class HistoryRow:
    run_name: str
    generation: int
    sigma_hash: str
    sigma_seed_equivalent: int
    best_mean_info: float
    best_mean_free: float
    best_mean_value: float
    best_failed_goals: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pool beta=0.5 four_rooms generation-best sigmas and score structural motifs."
    )
    p.add_argument(
        "--search-root",
        type=Path,
        default=Path(
            "/media/merlin/grid-twist/data-schema-10/multi/"
            "init_method=perm_balanced/env_id=four_rooms/shape=7x7/beta=0.5/det=0.97"
        ),
        help="Root containing the GA run_name=* directories.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/media/merlin/grid-twist/analysis/four_rooms-beta0p5-ga-pool"),
        help="Separate output directory for the pooled proof-of-concept artifacts.",
    )
    p.add_argument(
        "--run-glob",
        type=str,
        default="run_name=*",
        help="Glob used under --search-root to select run directories.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top-ranked unique sigmas to render and summarise.",
    )
    p.add_argument(
        "--gridtwist-src",
        type=Path,
        default=Path("/media/merlin/phd-marlyn/gridTwist/src"),
        help="Path to gridTwist/src for recomputing mean-over-goals metric heatmaps.",
    )
    p.add_argument(
        "--gridfour-src",
        type=Path,
        default=Path("/home/karen/phd-marlyn/gridFour/src"),
        help="Path to gridFour/src for cross-repo evaluation helpers.",
    )
    p.add_argument(
        "--copy-summaries",
        action="store_true",
        help="Copy sigma summaries and sigma.npy files into the output folder (default on).",
    )
    p.add_argument(
        "--no-copy-summaries",
        dest="copy_summaries",
        action="store_false",
        help="Do not copy sigma summaries and sigma.npy files into the output folder.",
    )
    p.set_defaults(copy_summaries=True)
    return p.parse_args()


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out


def _safe_int(value: object, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _find_history_rows(run_dir: Path) -> List[HistoryRow]:
    summary_path = next(run_dir.glob("*-multi-all.summary.json"))
    summary = json.loads(summary_path.read_text())
    history = summary.get("history", [])
    rows: List[HistoryRow] = []
    for snap in history:
        sigma_hash = str(snap.get("best_sigma_hash", "")).strip()
        if not sigma_hash:
            continue
        rows.append(
            HistoryRow(
                run_name=run_dir.name,
                generation=_safe_int(snap.get("gen"), default=-1),
                sigma_hash=sigma_hash,
                sigma_seed_equivalent=_safe_int(snap.get("best_sigma_seed_equivalent"), default=-1),
                best_mean_info=_safe_float(snap.get("best_expected_di", snap.get("best_mean_info"))),
                best_mean_free=_safe_float(snap.get("best_mean_free")),
                best_mean_value=_safe_float(snap.get("best_mean_value")),
                best_failed_goals=_safe_int(snap.get("best_failed_goals"), default=0),
            )
        )
    return rows


def discover_history_rows(search_root: Path, *, run_glob: str) -> List[HistoryRow]:
    out: List[HistoryRow] = []
    for run_dir in sorted(search_root.glob(run_glob)):
        if not run_dir.is_dir():
            continue
        out.extend(_find_history_rows(run_dir))
    return out


def _find_sigma_dir(run_dir: Path, sigma_hash: str) -> Path:
    matches = sorted(run_dir.glob(f"hive_sigma/**/sigma_id={sigma_hash}"))
    if not matches:
        raise FileNotFoundError(f"Could not locate sigma_id={sigma_hash} under {run_dir}")
    return matches[0]


def _build_sigma_dir_index(run_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for sigma_dir in sorted(run_dir.glob("hive_sigma/**/sigma_id=*")):
        sigma_hash = sigma_dir.name.removeprefix("sigma_id=")
        out.setdefault(sigma_hash, sigma_dir)
    return out


def build_base_env() -> GridRoom:
    shape = (7, 7)
    walls = compute_four_room_walls(shape[1], shape[0])
    return GridRoom(
        {
            "shape": shape,
            "goals": [],
            "manhattan": True,
            "determinism": 1.0,
            "epsilon": 0.0,
            "walls": walls,
        }
    )


def _summary_json_for_run(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("*-multi-all.summary.json"))
    if not matches:
        raise FileNotFoundError(f"Could not find *-multi-all.summary.json under {run_dir}")
    return matches[0]


def _frontier_targets(graph: nx.Graph, region_states: Sequence[int]) -> List[int]:
    region = set(int(s) for s in region_states)
    frontier: set[int] = set()
    for s in region:
        for nbr in graph.neighbors(int(s)):
            if nbr not in region:
                frontier.add(int(nbr))
    return sorted(frontier)


def define_regions(env: GridRoom) -> Dict[str, List[int]]:
    regions: Dict[str, List[int]] = {
        "top_room": [],
        "top_left": [],
        "top_right": [],
        "bottom_left": [],
        "bottom_right": [],
    }
    for s in env.available_states:
        y, x = env.get_multi_index(int(s))
        if y <= 3:
            regions["top_room"].append(int(s))
            if x <= 3:
                regions["top_left"].append(int(s))
            elif x >= 5:
                regions["top_right"].append(int(s))
        elif y >= 5 and x <= 2:
            regions["bottom_left"].append(int(s))
        elif y >= 5 and x >= 4:
            regions["bottom_right"].append(int(s))
    return {name: sorted(vals) for name, vals in regions.items()}


def _disjoint_room_regions(regions: Dict[str, List[int]]) -> Dict[str, List[int]]:
    return {
        "top_left": list(regions["top_left"]),
        "top_right": list(regions["top_right"]),
        "bottom_left": list(regions["bottom_left"]),
        "bottom_right": list(regions["bottom_right"]),
    }


def _all_pairs_dist(graph: nx.Graph) -> Dict[int, Dict[int, int]]:
    return {int(src): {int(dst): int(d) for dst, d in dmap.items()} for src, dmap in nx.all_pairs_shortest_path_length(graph)}


def _distance_to_targets(dist_map: Dict[int, Dict[int, int]], s: int, targets: Sequence[int]) -> int:
    vals = [dist_map[int(s)][int(t)] for t in targets if int(t) in dist_map[int(s)]]
    return min(vals) if vals else 10**9


def _successor_state(env: GridRoom, sigma: np.ndarray, s: int, label_idx: int) -> int:
    actual = int(sigma[int(s), int(label_idx)])
    ss = int(env.actions.get_successor_state(int(s), actual))
    return int(s) if ss in set(env.walls_flat) else ss


def _progress_score(
    env: GridRoom,
    sigma: np.ndarray,
    dist_map: Dict[int, Dict[int, int]],
    *,
    source_states: Sequence[int],
    target_states: Sequence[int],
    label_idx: int,
) -> float:
    source = [int(s) for s in source_states if int(s) not in set(target_states)]
    if not source or not target_states:
        return float("nan")
    better = 0
    for s in source:
        d0 = _distance_to_targets(dist_map, s, target_states)
        ss = _successor_state(env, sigma, s, label_idx)
        d1 = _distance_to_targets(dist_map, ss, target_states)
        if d1 < d0:
            better += 1
    return float(better / len(source))


def _entry_score(
    env: GridRoom,
    sigma: np.ndarray,
    *,
    source_states: Sequence[int],
    target_states: Sequence[int],
    label_idx: int,
) -> float:
    source = [int(s) for s in source_states if int(s) not in set(target_states)]
    if not source or not target_states:
        return float("nan")
    hits = 0
    target = set(int(t) for t in target_states)
    for s in source:
        ss = _successor_state(env, sigma, s, label_idx)
        if ss in target:
            hits += 1
    return float(hits / len(source))


def compute_structure_scores(env: GridRoom, sigma: np.ndarray) -> Dict[str, object]:
    graph = env.build_graph().subgraph(env.available_states).copy()
    dist_map = _all_pairs_dist(graph)
    regions = define_regions(env)

    out: Dict[str, object] = {}
    outside_top = [int(s) for s in env.available_states if int(s) not in set(regions["top_room"])]
    top_room = regions["top_room"]

    for label_idx, label in enumerate(LABELS):
        out[f"top_room_progress_{label}"] = _progress_score(
            env,
            sigma,
            dist_map,
            source_states=outside_top,
            target_states=top_room,
            label_idx=label_idx,
        )
        out[f"top_room_entry_{label}"] = _entry_score(
            env,
            sigma,
            source_states=outside_top,
            target_states=top_room,
            label_idx=label_idx,
        )

    exit_regions = ("top_left", "top_right", "bottom_left", "bottom_right")
    for region_name in exit_regions:
        region_states = regions[region_name]
        targets = _frontier_targets(graph, region_states)
        for label_idx, label in enumerate(LABELS):
            out[f"{region_name}_exit_progress_{label}"] = _progress_score(
                env,
                sigma,
                dist_map,
                source_states=region_states,
                target_states=targets,
                label_idx=label_idx,
            )
            out[f"{region_name}_exit_1step_{label}"] = _entry_score(
                env,
                sigma,
                source_states=region_states,
                target_states=targets,
                label_idx=label_idx,
            )

    def _best(prefix: str) -> tuple[str, float]:
        vals = {label: float(out[f"{prefix}_{label}"]) for label in LABELS}
        label_best = max(vals, key=vals.get)
        return label_best, vals[label_best]

    best_top_label, best_top_score = _best("top_room_progress")
    best_top_entry_label, best_top_entry_score = _best("top_room_entry")
    out["best_top_room_progress_label"] = best_top_label
    out["best_top_room_progress_score"] = best_top_score
    out["best_top_room_entry_label"] = best_top_entry_label
    out["best_top_room_entry_score"] = best_top_entry_score

    best_region = ""
    best_region_label = ""
    best_region_score = float("-inf")
    for region_name in exit_regions:
        label_best, score_best = _best(f"{region_name}_exit_progress")
        out[f"best_{region_name}_exit_progress_label"] = label_best
        out[f"best_{region_name}_exit_progress_score"] = score_best
        if score_best > best_region_score:
            best_region = region_name
            best_region_label = label_best
            best_region_score = score_best
    out["best_exit_region"] = best_region
    out["best_exit_label"] = best_region_label
    out["best_exit_progress_score"] = best_region_score
    out["interesting_structure_score"] = float(max(best_top_score, best_region_score))
    return out


def _copy_sigma_artifacts(sigma_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("sigma.npy", "summary.json", "metrics.npz", "goals.json"):
        src = sigma_dir / name
        if src.exists():
            shutil.copy2(src, target_dir / name)


def _add_gridtwist_to_path(gridtwist_src: Path) -> None:
    resolved = gridtwist_src.expanduser().resolve()
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))


def _load_eval_symbols(gridtwist_src: Path):
    _add_gridtwist_to_path(gridtwist_src)
    from evolution_core.gridfour_bridge import EvalConfig, evaluate_sigma

    return EvalConfig, evaluate_sigma


def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, :3]
    return img


def _crop_white_border(img: np.ndarray, threshold: int = 250) -> np.ndarray:
    rgb = _ensure_rgb(img)
    mask = np.any(rgb < threshold, axis=2)
    if not np.any(mask):
        return rgb
    ys, xs = np.where(mask)
    y0, y1 = int(np.min(ys)), int(np.max(ys)) + 1
    x0, x1 = int(np.min(xs)), int(np.max(xs)) + 1
    return rgb[y0:y1, x0:x1]


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    new_w = max(1, int(round((w / float(h)) * float(target_h))))
    yi = np.clip(np.round(np.linspace(0, h - 1, target_h)).astype(int), 0, h - 1)
    xi = np.clip(np.round(np.linspace(0, w - 1, new_w)).astype(int), 0, w - 1)
    return _ensure_rgb(img)[yi][:, xi]


def _add_outer_padding(img: np.ndarray, pad: int = 16, value: int = 255) -> np.ndarray:
    rgb = _ensure_rgb(img)
    return np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=value)


def _white_strip(height: int, width: int, value: int = 255) -> np.ndarray:
    return np.full((max(1, int(height)), max(1, int(width)), 3), int(value), dtype=np.uint8)


def _annotate_below(img: np.ndarray, lines: Sequence[str], pad: int, text_size: int = 18) -> np.ndarray:
    rgb = _ensure_rgb(img)
    try:
        from PIL import Image, ImageDraw, ImageFont

        line_gap = max(6, int(round(0.35 * text_size)))
        band_h = (len(lines) * text_size) + ((len(lines) - 1) * line_gap) + (2 * pad)
        canvas = np.concatenate([rgb, _white_strip(band_h, rgb.shape[1], value=255)], axis=0)
        pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", text_size)
        except Exception:
            font = ImageFont.load_default()
        y = rgb.shape[0] + pad
        for line in lines:
            draw.text((pad, y), line, fill=(0, 0, 0), font=font)
            y += text_size + line_gap
        return np.asarray(pil)
    except Exception:
        return rgb


def _plot_heatmap_without_goal_marker(env: GridRoom, vec: np.ndarray, title: str, cmap):
    original_goals = getattr(env, "goals", None)
    try:
        if original_goals is not None:
            setattr(env, "goals", [])
        return plot_heatmap(env, vec, title=title, label=True, cmap=cmap)
    finally:
        if original_goals is not None:
            setattr(env, "goals", original_goals)


def _candidate_path_variants(raw_path: str | None) -> List[Path]:
    if not raw_path:
        return []
    path = Path(raw_path).expanduser()
    variants = [path]
    text = str(path)
    rewrites = {
        "/home/karen/gridtwist-outputs": [
            "/media/panther/gridtwist-outputs",
            "/media/merlin/gridtwist-outputs",
        ],
    }
    for src_prefix, dst_prefixes in rewrites.items():
        if text.startswith(src_prefix):
            suffix = text[len(src_prefix):]
            for dst_prefix in dst_prefixes:
                variants.append(Path(dst_prefix + suffix))
    dedup: List[Path] = []
    seen: set[str] = set()
    for candidate in variants:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(candidate)
    return dedup


def _resolve_source_hive_root(cfg: Dict[str, object]) -> Path | None:
    candidates: List[Path] = []
    hive_store_root = cfg.get("hive_store_root")
    save_dir = cfg.get("save_dir")
    candidates.extend(_candidate_path_variants(str(hive_store_root)) if hive_store_root else [])
    if save_dir:
        for base in _candidate_path_variants(str(save_dir)):
            candidates.append(base / "hive_sigma")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _build_hive_sigma_index(hive_root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for sigma_dir in sorted(hive_root.glob("**/sigma_id=*")):
        sigma_hash = sigma_dir.name.removeprefix("sigma_id=")
        out.setdefault(sigma_hash, sigma_dir)
    return out


def _load_goal_arrays_from_sigma_dir(
    sigma_dir: Path,
    goal: int,
    *,
    expected_len: int,
) -> Dict[str, np.ndarray] | None:
    goal_path = sigma_dir / f"goal-{int(goal)}.npz"
    if not goal_path.exists():
        return None
    try:
        data = np.load(goal_path, allow_pickle=False)
    except Exception:
        return None
    out: Dict[str, np.ndarray] = {}
    for key in ("info", "free", "value"):
        if key not in data.files:
            return None
        arr = np.asarray(data[key], dtype=float).reshape(-1)
        if arr.shape[0] != expected_len:
            return None
        out[key] = arr
    return out


def _render_mean_metric_heatmaps(
    sigma: np.ndarray,
    *,
    env_id: str,
    shape: tuple[int, int],
    beta: float,
    determinism: float,
    theta: float,
    state_dist: str,
    goals: Sequence[int],
    sigma_dirs_for_arrays: Sequence[Path],
    gridtwist_src: Path,
    gridfour_src: Path,
    out_png: Path,
    footer_lines: Sequence[str],
) -> Dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    EvalConfig, evaluate_sigma = _load_eval_symbols(gridtwist_src)
    n_states = int(shape[0] * shape[1])
    infos: list[np.ndarray] = []
    values: list[np.ndarray] = []
    frees: list[np.ndarray] = []
    converged_goals = 0
    loaded_goal_count = 0
    recomputed_goal_count = 0
    loaded_by_goal: Dict[int, Dict[str, np.ndarray]] = {}
    for sigma_dir in sigma_dirs_for_arrays:
        for goal in goals:
            goal_i = int(goal)
            if goal_i in loaded_by_goal:
                continue
            loaded = _load_goal_arrays_from_sigma_dir(sigma_dir, goal_i, expected_len=n_states)
            if loaded is not None:
                loaded_by_goal[goal_i] = loaded

    for goal in goals:
        goal_i = int(goal)
        stored = loaded_by_goal.get(goal_i)
        if stored is not None:
            infos.append(np.asarray(stored["info"], dtype=float))
            values.append(np.asarray(stored["value"], dtype=float))
            frees.append(np.asarray(stored["free"], dtype=float))
            converged_goals += 1
            loaded_goal_count += 1
            continue
        cfg = EvalConfig(
            env_id=str(env_id),
            shape=tuple(int(x) for x in shape),
            goal=goal_i,
            beta=float(beta),
            determinism=float(determinism),
            manhattan=True,
            theta=float(theta),
            state_dist=str(state_dist),
            max_blahut_iterations=100_000,
            max_info_iterations=10_000,
        )
        res = evaluate_sigma(sigma, cfg, gridfour_src=str(gridfour_src.expanduser().resolve()))
        if not bool(res.get("ok", False)):
            continue
        infos.append(np.asarray(res["info"], dtype=float))
        values.append(np.asarray(res["value"], dtype=float))
        frees.append(np.asarray(res["free"], dtype=float))
        converged_goals += 1
        recomputed_goal_count += 1

    if not infos or not values or not frees:
        raise RuntimeError("No converged goal evaluations available for mean-over-goals heatmaps.")

    mean_info = np.nanmean(np.asarray(infos, dtype=float), axis=0)
    mean_value = np.nanmean(np.asarray(values, dtype=float), axis=0)
    mean_free = np.nanmean(np.asarray(frees, dtype=float), axis=0)

    env = build_base_env()
    items = [
        ("info", mean_info, "Mean Decision Information (all goals)", di.SOFT_ORANGES),
        ("value", mean_value, "Mean Value (all goals)", di.SOFT_BLUES),
        ("free", mean_free, "Mean Free Energy (all goals)", di.SOFT_PURPLES),
    ]

    panels: list[np.ndarray] = []
    for key, vec, title, cmap in items:
        tmp_png = out_png.parent / f"_tmp_mean_{key}.png"
        ax = _plot_heatmap_without_goal_marker(env, vec, title=title, cmap=cmap)
        ax.figure.savefig(tmp_png, dpi=180, bbox_inches="tight")
        plt.close(ax.figure)
        panels.append(_crop_white_border(_ensure_rgb(imageio.imread(tmp_png))))
        try:
            tmp_png.unlink()
        except Exception:
            pass

    max_h = max(p.shape[0] for p in panels)
    resized = [_resize_to_height(p, max_h) for p in panels]
    gap = np.full((max_h, 24, 3), 255, dtype=np.uint8)
    combo = np.concatenate([resized[0], gap, resized[1], gap, resized[2]], axis=1)
    combo = _add_outer_padding(combo, pad=18, value=255)
    combo = _annotate_below(combo, footer_lines, pad=20, text_size=24)
    imageio.imwrite(out_png, combo)

    return {
        "mean_info_vector": mean_info,
        "mean_value_vector": mean_value,
        "mean_free_vector": mean_free,
        "converged_goals": int(converged_goals),
        "loaded_goals_from_arrays": int(loaded_goal_count),
        "recomputed_goals": int(recomputed_goal_count),
    }


def _region_means(vec: np.ndarray, regions: Dict[str, List[int]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    arr = np.asarray(vec, dtype=float)
    for name, states in regions.items():
        if not states:
            out[name] = float("nan")
            continue
        vals = arr[np.asarray(states, dtype=int)]
        out[name] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    return out


def _peak_region(region_means: Dict[str, float]) -> str:
    finite = {k: float(v) for k, v in region_means.items() if np.isfinite(float(v))}
    if not finite:
        return ""
    return max(finite, key=finite.get)


def _render_rank_artifacts(
    ranked_rows: Sequence[Dict[str, object]],
    *,
    search_root: Path,
    out_dir: Path,
    run_cfg_cache: Dict[str, Dict[str, object]],
    gridtwist_src: Path,
    gridfour_src: Path,
    top_k: int,
) -> List[Dict[str, object]]:
    import matplotlib.pyplot as plt

    panel_dir = out_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: List[Dict[str, object]] = []
    room_regions = _disjoint_room_regions(define_regions(build_base_env()))
    source_hive_index_cache: Dict[str, Dict[str, Path]] = {}

    for rank, row in enumerate(ranked_rows[:top_k], start=1):
        sigma_path = Path(str(row["sigma_dir"])) / "sigma.npy"
        sigma = np.asarray(np.load(sigma_path), dtype=int)
        env = build_base_env()
        env.set_sigma(sigma)  # atomic sigma + sigma_inv update; see GridRoom.set_sigma

        run_name = str(row["first_seen_run"])
        cfg = dict(run_cfg_cache[run_name])
        sigma_hash = str(row["sigma_hash"])
        sigma_summary = json.loads((Path(str(row["sigma_dir"])) / "summary.json").read_text())
        goals = [int(g) for g in sigma_summary.get("goals", [])]
        sigma_dirs_for_arrays: List[Path] = [Path(str(row["sigma_dir"]))]
        source_hive_root = _resolve_source_hive_root(cfg)
        if source_hive_root is not None:
            cache_key = str(source_hive_root)
            if cache_key not in source_hive_index_cache:
                source_hive_index_cache[cache_key] = _build_hive_sigma_index(source_hive_root)
            source_sigma_dir = source_hive_index_cache[cache_key].get(sigma_hash)
            if source_sigma_dir is not None and source_sigma_dir not in sigma_dirs_for_arrays:
                sigma_dirs_for_arrays.append(source_sigma_dir)

        dual_name = f"rank-{rank:02d}-{str(row['sigma_hash'])[:12]}-dual-landscape.png"
        dual_path = panel_dir / dual_name
        dual_footer = (
            f"rank={rank} sigma={row['sigma_hash']}\n"
            f"score={float(row['interesting_structure_score']):.3f} info={float(row['best_mean_info']):.3f} "
            f"exit={row['best_exit_region']}:{row['best_exit_label']} top={row['best_top_room_progress_label']}"
        )
        fig = plot_twist_dual_landscape(
            env,
            policy=None,
            filename=str(dual_path),
            footer_text=dual_footer,
        )
        plt.close(fig)

        metrics_name = f"rank-{rank:02d}-{str(row['sigma_hash'])[:12]}-mean-info-value-free.png"
        metrics_path = panel_dir / metrics_name
        metrics_footer = [
            f"rank={rank} sigma={row['sigma_hash']}",
            (
                f"mean_info={float(row['best_mean_info']):.3f} "
                f"mean_free={float(row['best_mean_free']):.3f} "
                f"mean_value={float(row['best_mean_value']):.3f}"
            ),
        ]
        metric_payload = _render_mean_metric_heatmaps(
            sigma,
            env_id=str(cfg.get("env_id", "four_rooms")),
            shape=tuple(int(x) for x in cfg.get("shape", [7, 7])),
            beta=float(cfg.get("beta", 1.0)),
            determinism=float(cfg.get("determinism", 0.97)),
            theta=float(cfg.get("theta", 1e-5)),
            state_dist=str(cfg.get("state_dist", "uniform")),
            goals=goals,
            sigma_dirs_for_arrays=sigma_dirs_for_arrays,
            gridtwist_src=gridtwist_src,
            gridfour_src=gridfour_src,
            out_png=metrics_path,
            footer_lines=metrics_footer,
        )

        info_region_means = _region_means(np.asarray(metric_payload["mean_info_vector"], dtype=float), room_regions)
        value_region_means = _region_means(np.asarray(metric_payload["mean_value_vector"], dtype=float), room_regions)
        free_region_means = _region_means(np.asarray(metric_payload["mean_free_vector"], dtype=float), room_regions)
        peak_info_room = _peak_region(info_region_means)
        peak_value_room = _peak_region(value_region_means)
        peak_free_room = _peak_region(free_region_means)

        metric_row = {
            "rank": rank,
            "sigma_hash": row["sigma_hash"],
            "best_exit_region": row["best_exit_region"],
            "best_exit_label": row["best_exit_label"],
            "best_top_room_progress_label": row["best_top_room_progress_label"],
            "best_mean_info": row["best_mean_info"],
            "best_mean_free": row["best_mean_free"],
            "best_mean_value": row["best_mean_value"],
            "converged_goals": metric_payload["converged_goals"],
            "loaded_goals_from_arrays": metric_payload["loaded_goals_from_arrays"],
            "recomputed_goals": metric_payload["recomputed_goals"],
            "peak_mean_info_room": peak_info_room,
            "peak_mean_value_room": peak_value_room,
            "peak_mean_free_room": peak_free_room,
            "mean_info_matches_exit_room": peak_info_room == str(row["best_exit_region"]),
            "mean_value_matches_exit_room": peak_value_room == str(row["best_exit_region"]),
            "mean_free_matches_exit_room": peak_free_room == str(row["best_exit_region"]),
            "dual_landscape_png": dual_name,
            "mean_metric_png": metrics_name,
        }
        for room_name, value in info_region_means.items():
            metric_row[f"mean_info_room_{room_name}"] = value
        for room_name, value in value_region_means.items():
            metric_row[f"mean_value_room_{room_name}"] = value
        for room_name, value in free_region_means.items():
            metric_row[f"mean_free_room_{room_name}"] = value
        metric_rows.append(metric_row)

    return metric_rows


def _write_ranked_markdown(
    path: Path,
    ranked_rows: Sequence[Dict[str, object]],
    metric_rows: Sequence[Dict[str, object]],
    *,
    top_k: int,
    run_glob: str,
) -> None:
    by_hash = {str(row["sigma_hash"]): row for row in metric_rows}
    lines: List[str] = [
        "# Four-Rooms GA Sigma Structure Ranking",
        "",
        "This proof-of-concept pools generation-best sigmas across the selected runs and scores them for interpretable four-rooms structure.",
        "",
        "How to read the twist panels:",
        "- `best_exit_region` + `best_exit_label` means that label most often reduces graph distance to the exit frontier of that room.",
        "- In the dual landscape plot, inspect the inverse subplot titled `what's labelled <label>` to see the movement pattern for that label.",
        "- `best_top_room_progress_label` means that label most often reduces distance into the large top room from outside it.",
        "- `peak_mean_*_room` is computed from the mean-over-goals heatmaps for the disjoint room masks `top_left`, `top_right`, `bottom_left`, `bottom_right`.",
        "",
        f"Selected runs: `{run_glob}`",
        f"Ranked rows rendered: `{min(len(ranked_rows), top_k)}`",
        "",
        "## Ranked Sigma Notes",
        "",
    ]

    for rank, row in enumerate(ranked_rows[:top_k], start=1):
        sigma_hash = str(row["sigma_hash"])
        metric = by_hash[sigma_hash]
        dual_rel = f"panels/{metric['dual_landscape_png']}"
        heat_rel = f"panels/{metric['mean_metric_png']}"
        exit_label = str(row["best_exit_label"])
        top_label = str(row["best_top_room_progress_label"])
        lines.extend(
            [
                f"### Rank {rank} `{sigma_hash}`",
                "",
                f"- Exit motif: `{row['best_exit_region']}` driven by label `{exit_label}`. Inspect inverse panel `what's labelled {exit_label}` in [{metric['dual_landscape_png']}]({dual_rel}).",
                f"- Top-room attractor: label `{top_label}`. Inspect inverse panel `what's labelled {top_label}` in the same twist plot.",
                (
                    f"- Mean metrics: `mean_info={float(row['best_mean_info']):.3f}`, "
                    f"`mean_free={float(row['best_mean_free']):.3f}`, "
                    f"`mean_value={float(row['best_mean_value']):.3f}`."
                ),
                (
                    f"- Peak mean-metric rooms: "
                    f"`info={metric['peak_mean_info_room'] or 'n/a'}`, "
                    f"`free={metric['peak_mean_free_room'] or 'n/a'}`, "
                    f"`value={metric['peak_mean_value_room'] or 'n/a'}`."
                ),
                (
                    f"- Exit/info room match: "
                    f"`{metric['mean_info_matches_exit_room']}` "
                    f"(converged goals: `{metric['converged_goals']}`, "
                    f"stored arrays: `{metric['loaded_goals_from_arrays']}`, "
                    f"recomputed: `{metric['recomputed_goals']}`)."
                ),
                f"- Mean-over-goals heatmaps: [{metric['mean_metric_png']}]({heat_rel})",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Dict[str, object]], *, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    search_root = args.search_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    gridtwist_src = args.gridtwist_src.expanduser().resolve()
    gridfour_src = args.gridfour_src.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    history_rows = discover_history_rows(search_root, run_glob=args.run_glob)
    base_env = build_base_env()
    run_cfg_cache: Dict[str, Dict[str, object]] = {}
    sigma_dir_index_cache: Dict[str, Dict[str, Path]] = {}

    all_rows_out: List[Dict[str, object]] = []
    unique_rows_out: List[Dict[str, object]] = []
    pooled: Dict[str, Dict[str, object]] = {}

    for row in history_rows:
        run_dir = search_root / row.run_name
        if row.run_name not in run_cfg_cache:
            run_summary = json.loads(_summary_json_for_run(run_dir).read_text())
            run_cfg_cache[row.run_name] = dict(run_summary.get("config", {}))
        if row.run_name not in sigma_dir_index_cache:
            sigma_dir_index_cache[row.run_name] = _build_sigma_dir_index(run_dir)
        sigma_dir = sigma_dir_index_cache[row.run_name].get(row.sigma_hash)
        if sigma_dir is None:
            sigma_dir = _find_sigma_dir(run_dir, row.sigma_hash)
        sigma_summary = json.loads((sigma_dir / "summary.json").read_text())
        sigma = np.asarray(np.load(sigma_dir / "sigma.npy"), dtype=int)

        row_out = {
            "run_name": row.run_name,
            "generation": row.generation,
            "sigma_hash": row.sigma_hash,
            "sigma_seed_equivalent": row.sigma_seed_equivalent,
            "best_mean_info": row.best_mean_info,
            "best_mean_free": row.best_mean_free,
            "best_mean_value": row.best_mean_value,
            "best_failed_goals": row.best_failed_goals,
            "epsilon_actual": _safe_float(sigma_summary.get("epsilon_actual")),
            "epsilon_bin": _safe_float(sigma_summary.get("epsilon")),
            "chi_twist": _safe_float(sigma_summary.get("chi_twist")),
            "predominant_ordering": json.dumps(sigma_summary.get("predominant_ordering", [])),
            "sigma_dir": str(sigma_dir),
        }
        all_rows_out.append(row_out)

        cur = pooled.get(row.sigma_hash)
        if cur is None:
            structure = compute_structure_scores(base_env, sigma)
            cur = {
                "sigma_hash": row.sigma_hash,
                "sigma_seed_equivalent": row.sigma_seed_equivalent,
                "first_seen_run": row.run_name,
                "first_seen_generation": row.generation,
                "last_seen_generation": row.generation,
                "occurrence_count": 1,
                "source_runs": [row.run_name],
                "source_generations": [row.generation],
                "best_mean_info": row.best_mean_info,
                "best_mean_free": row.best_mean_free,
                "best_mean_value": row.best_mean_value,
                "best_failed_goals": row.best_failed_goals,
                "epsilon_actual": _safe_float(sigma_summary.get("epsilon_actual")),
                "epsilon_bin": _safe_float(sigma_summary.get("epsilon")),
                "chi_twist": _safe_float(sigma_summary.get("chi_twist")),
                "predominant_ordering": json.dumps(sigma_summary.get("predominant_ordering", [])),
                "sigma_dir": str(sigma_dir),
                "summary_path": str(sigma_dir / "summary.json"),
                **structure,
            }
            pooled[row.sigma_hash] = cur
            if args.copy_summaries:
                _copy_sigma_artifacts(sigma_dir, out_dir / "sigmas" / f"sigma_id={row.sigma_hash}")
        else:
            cur["occurrence_count"] = int(cur["occurrence_count"]) + 1
            cur["last_seen_generation"] = max(int(cur["last_seen_generation"]), row.generation)
            src_runs = list(cur["source_runs"])
            if row.run_name not in src_runs:
                src_runs.append(row.run_name)
                cur["source_runs"] = src_runs
            gens = list(cur["source_generations"])
            gens.append(row.generation)
            cur["source_generations"] = gens

    for item in pooled.values():
        item["source_runs"] = json.dumps(sorted(item["source_runs"]))
        item["source_generations"] = json.dumps(sorted(set(int(x) for x in item["source_generations"])))
        unique_rows_out.append(item)

    unique_rows_out.sort(key=lambda r: (-float(r["interesting_structure_score"]), float(r["best_mean_info"])))
    all_rows_out.sort(key=lambda r: (r["run_name"], int(r["generation"])))

    all_fieldnames = [
        "run_name",
        "generation",
        "sigma_hash",
        "sigma_seed_equivalent",
        "best_mean_info",
        "best_mean_free",
        "best_mean_value",
        "best_failed_goals",
        "epsilon_actual",
        "epsilon_bin",
        "chi_twist",
        "predominant_ordering",
        "sigma_dir",
    ]
    write_csv(out_dir / "pooled_generation_best_all.csv", all_rows_out, fieldnames=all_fieldnames)

    unique_fieldnames = [
        "sigma_hash",
        "sigma_seed_equivalent",
        "occurrence_count",
        "first_seen_run",
        "first_seen_generation",
        "last_seen_generation",
        "source_runs",
        "source_generations",
        "best_mean_info",
        "best_mean_free",
        "best_mean_value",
        "best_failed_goals",
        "epsilon_actual",
        "epsilon_bin",
        "chi_twist",
        "predominant_ordering",
        "best_top_room_progress_label",
        "best_top_room_progress_score",
        "best_top_room_entry_label",
        "best_top_room_entry_score",
        "best_top_left_exit_progress_label",
        "best_top_left_exit_progress_score",
        "best_top_right_exit_progress_label",
        "best_top_right_exit_progress_score",
        "best_bottom_left_exit_progress_label",
        "best_bottom_left_exit_progress_score",
        "best_bottom_right_exit_progress_label",
        "best_bottom_right_exit_progress_score",
        "best_exit_region",
        "best_exit_label",
        "best_exit_progress_score",
        "interesting_structure_score",
        "top_room_progress_N",
        "top_room_progress_E",
        "top_room_progress_S",
        "top_room_progress_W",
        "sigma_dir",
        "summary_path",
    ]
    extra_unique_fields = sorted(
        {
            key
            for row in unique_rows_out
            for key in row.keys()
            if key not in set(unique_fieldnames)
        }
    )
    unique_fieldnames = unique_fieldnames + extra_unique_fields
    write_csv(out_dir / "pooled_generation_best_unique.csv", unique_rows_out, fieldnames=unique_fieldnames)
    top_rows = unique_rows_out[: max(0, int(args.top_k))]
    write_csv(out_dir / f"top_{int(args.top_k)}_by_structure.csv", top_rows, fieldnames=unique_fieldnames)

    metric_rows = _render_rank_artifacts(
        top_rows,
        search_root=search_root,
        out_dir=out_dir,
        run_cfg_cache=run_cfg_cache,
        gridtwist_src=gridtwist_src,
        gridfour_src=gridfour_src,
        top_k=int(args.top_k),
    )
    if metric_rows:
        metric_fieldnames = [
            "rank",
            "sigma_hash",
            "best_exit_region",
            "best_exit_label",
            "best_top_room_progress_label",
            "best_mean_info",
            "best_mean_free",
            "best_mean_value",
            "converged_goals",
            "loaded_goals_from_arrays",
            "recomputed_goals",
            "peak_mean_info_room",
            "peak_mean_free_room",
            "peak_mean_value_room",
            "mean_info_matches_exit_room",
            "mean_free_matches_exit_room",
            "mean_value_matches_exit_room",
            "dual_landscape_png",
            "mean_metric_png",
        ]
        extra_metric_fields = sorted(
            {key for row in metric_rows for key in row.keys() if key not in set(metric_fieldnames)}
        )
        write_csv(
            out_dir / f"top_{int(args.top_k)}_metric_summary.csv",
            metric_rows,
            fieldnames=metric_fieldnames + extra_metric_fields,
        )
        _write_ranked_markdown(
            out_dir / "ranked_sigma_analysis.md",
            top_rows,
            metric_rows,
            top_k=int(args.top_k),
            run_glob=args.run_glob,
        )

    manifest = {
        "search_root": str(search_root),
        "out_dir": str(out_dir),
        "run_glob": str(args.run_glob),
        "history_rows_total": int(len(all_rows_out)),
        "unique_sigma_total": int(len(unique_rows_out)),
        "top_k": int(args.top_k),
        "copied_sigma_artifacts": bool(args.copy_summaries),
        "gridtwist_src": str(gridtwist_src),
        "gridfour_src": str(gridfour_src),
        "notes": [
            "Proof-of-concept pool outside grid-four schema-10.",
            "Interestingness currently uses max(top-room attraction, best region-exit progress).",
            "Scores are deterministic, geometry-based heuristics for four_rooms 7x7 only.",
            "Top-ranked sigmas also include recomputed mean-over-goals info/value/free heatmaps.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"out_dir: {out_dir}")
    print(f"history_rows_total: {len(all_rows_out)}")
    print(f"unique_sigma_total: {len(unique_rows_out)}")
    if unique_rows_out:
        top = unique_rows_out[0]
        print(
            "top_candidate:",
            top["sigma_hash"],
            f"interesting={float(top['interesting_structure_score']):.4f}",
            f"mean_info={float(top['best_mean_info']):.4f}",
            f"best_exit={top['best_exit_region']}:{top['best_exit_label']}",
            f"top_label={top['best_top_room_progress_label']}",
        )


if __name__ == "__main__":
    main()
