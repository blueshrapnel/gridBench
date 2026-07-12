"""Self-contained figure helpers for the ALife paper notebooks.

This is the paper-repo replacement for the frozen-gridFour notebook glue
(``alife_helpers`` + ``alife_paper_config``).  It imports **only** from the
public packages:

    * ``gridcore``  — environments and numerics
    * ``gridvis``   — plotting (imported by the notebooks directly)
    * ``gridbench`` — functional-graph analysis (imported lazily by gridvis)

Nothing here depends on gridFour.  Keep it small: environment construction,
figure saving, and run-config loading are all a notebook needs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

from gridbench import store
from gridcore.envs import (
    GridRoom,
    compute_corridor_1d_ring_walls,
    compute_corridor_four_rooms_walls,
    compute_pillar_walls,
    compute_pinwheel_walls,
    compute_plus_cross_walls,
    compute_x_wall_walls,
    compute_four_room_walls,
)

# Match the ALife template maths font (Computer Modern) across all figures.
mpl.rcParams["mathtext.fontset"] = "cm"

# ---------------------------------------------------------------------------
# Run references + paper constants (drop-in for the frozen alife_paper_config)
# ---------------------------------------------------------------------------
_NOTEBOOK_ROOT = Path(__file__).parent
FIGURES_DIR = _NOTEBOOK_ROOT / "figures"
PAPER_REFS_PATH = _NOTEBOOK_ROOT / "paper-refs.json"


def _load_paper_refs() -> dict:
    if not PAPER_REFS_PATH.exists():
        return {}
    return json.loads(PAPER_REFS_PATH.read_text())


_REFS = _load_paper_refs()

# Partition fields a run spec must carry to resolve a schema-11 run_dir.
_PARTITION_KEYS = (
    "init_method", "fitness_objective", "env_id", "shape", "beta", "det", "run_name",
)


def _resolve_run_dir(spec: dict):
    """Resolve a run spec to a schema-11 run_dir via gridbench.store.

    DATA_ROOT-relative (GRIDBENCH_DATA_ROOT), so the same spec resolves against
    the live store or a frozen paper bundle.  None if the spec is incomplete.
    """
    if not all(k in spec for k in _PARTITION_KEYS):
        return None
    return store.run_dir(**{k: spec[k] for k in _PARTITION_KEYS})


# --- Four rooms run (figures 1–5) ---
_FOUR_ROOMS = _REFS.get("four_rooms", {})
RUN_DIR = _resolve_run_dir(_FOUR_ROOMS)
BEST_SIGMA_HASH = _FOUR_ROOMS.get("best_sigma_hash", "24f551168e0469f95eea9fce2d976824")
RANDOM_SIGMA_SEED = _FOUR_ROOMS.get("random_sigma_seed", 20260319)
RANDOM_SIGMA_CHI_MIN = 0.65

# --- Wrap-grid run (figures 5–6) ---
_WRAP_GRID = _REFS.get("wrap_grid", {})
WRAP_GRID_RUN_DIR = _resolve_run_dir(_WRAP_GRID)

# --- Solve / plot constants ---
ENV_ID = "four_rooms"
SHAPE = (7, 7)
DETERMINISM = 0.97
BETA = 1.0
THETA = 1e-5
MAX_ITERATIONS = 100_000
MAX_INFO_ITERATIONS = 20_000

DI_CMAP = "Oranges"
DI_VMIN = 0.0
DI_VMAX = 15.0  # above GA range; accommodates random/β=0.3 twists


def read_sigma_refs() -> dict:
    """Read the four_rooms sigma refs. Raises if figure-1 has not been run."""
    refs = _load_paper_refs()
    fr = refs.get("four_rooms", {})
    if "random_sigma_hash" not in fr:
        raise FileNotFoundError(
            f"{PAPER_REFS_PATH} missing four_rooms.random_sigma_hash.\n"
            "Run figure-1 first to generate the canonical sigma selection."
        )
    return fr


# ---------------------------------------------------------------------------
# Environment construction (gridcore)
# ---------------------------------------------------------------------------
DEFAULT_SHAPES = {
    "open_grid": (7, 7),
    "four_rooms": (7, 7),
    "wrap_grid": (7, 7),
    "corr_1d_ring": (7, 7),
    "corr_four_rooms": (13, 13),
    "pillar_1": (7, 7),
    "pillar_2": (6, 6),
    "pillar_3": (7, 7),
    "plus_cross": (7, 7),
    "x_wall": (7, 7),
    "pinwheel": (7, 7),
    "wrap_pillar_3": (7, 7),
}


def default_shape(env_id: str) -> tuple[int, int]:
    if env_id not in DEFAULT_SHAPES:
        raise ValueError(f"Unsupported env_id={env_id!r}")
    return DEFAULT_SHAPES[env_id]


def _walls_for_env(env_id: str, shape: tuple[int, int]):
    width = int(shape[1])
    height = int(shape[0])
    if env_id == "four_rooms":
        return compute_four_room_walls(width, height)
    if env_id == "corr_1d_ring":
        return compute_corridor_1d_ring_walls(width, height)
    if env_id == "corr_four_rooms":
        return compute_corridor_four_rooms_walls(width, height)
    if env_id == "pillar_1":
        return compute_pillar_walls(width, height, 1)
    if env_id == "pillar_2":
        return compute_pillar_walls(width, height, 2)
    if env_id == "pillar_3":
        return compute_pillar_walls(width, height, 3)
    if env_id == "plus_cross":
        return compute_plus_cross_walls(width, height)
    if env_id == "x_wall":
        return compute_x_wall_walls(width, height)
    if env_id == "pinwheel":
        return compute_pinwheel_walls(width, height)
    if env_id == "wrap_pillar_3":
        # Same wall geometry as pillar_3; wrap=True is set in build_env.
        return compute_pillar_walls(width, height, 3)
    return None


def build_env(
    env_id: str,
    *,
    shape: tuple[int, int] | None = None,
    goal: int = 0,
    determinism: float = 0.97,
    epsilon: float = 0.0,
    twist_seed: int = 0,
    manhattan: bool = True,
) -> GridRoom:
    if shape is None:
        shape = default_shape(env_id)

    options = {
        "shape": tuple(int(v) for v in shape),
        "goals": [int(goal)],
        "manhattan": bool(manhattan),
        "determinism": float(determinism),
        "epsilon": float(epsilon),
        "twist_seed": int(twist_seed),
    }

    walls = _walls_for_env(env_id, shape)
    if walls is not None:
        options["walls"] = walls
    if env_id in {"wrap_grid", "wrap_pillar_3"}:
        options["wrap"] = True
    elif env_id not in {
        "open_grid", "four_rooms", "corr_1d_ring", "corr_four_rooms",
        "pillar_1", "pillar_2", "pillar_3", "plus_cross", "x_wall", "pinwheel",
    }:
        raise ValueError(f"Unsupported env_id={env_id!r}")

    return GridRoom(options)


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------
def save_figure_variants(
    fig,
    base_path: str | Path,
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 200,
    bbox_inches: str = "tight",
    pad_inches: float = 0.1,
) -> dict[str, Path]:
    """Save one matplotlib figure to multiple formats derived from a base path."""
    path = Path(base_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for fmt in formats:
        clean_fmt = str(fmt).strip().lower().lstrip(".")
        out_path = path.with_suffix(f".{clean_fmt}")
        fig.savefig(out_path, format=clean_fmt, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
        saved[clean_fmt] = out_path
    return saved


# ---------------------------------------------------------------------------
# Run-config loading (schema-10/11 run directory → best-sigma config)
# ---------------------------------------------------------------------------
def load_run_config(run_dir: Path) -> dict:
    """Load the best-sigma config from a run's summary JSON.

    Returns keys: run_dir, run_name, best_sigma_hash, sigma_path, hive_dir,
    env_id, shape, determinism, beta, chi_twist, mean_free, mean_info.
    """
    run_dir = Path(run_dir)
    summary_files = sorted(run_dir.glob("*-multi-all.summary.json"))
    if not summary_files:
        raise FileNotFoundError(f"No summary JSON in {run_dir}")
    with open(summary_files[0]) as f:
        summary = json.load(f)

    best_hash = summary["best_sigma_hash"]
    sigma_matches = list(run_dir.glob(f"hive_sigma/**/sigma_id={best_hash}/sigma.npy"))
    if not sigma_matches:
        raise FileNotFoundError(f"Sigma {best_hash} not found in hive at {run_dir}")

    cfg_run = summary.get("config", {})
    return {
        "run_dir": run_dir,
        "run_name": run_dir.name.replace("run_name=", ""),
        "best_sigma_hash": best_hash,
        "sigma_path": sigma_matches[0],
        "hive_dir": sigma_matches[0].parent,
        "env_id": cfg_run.get("env_id", "unknown"),
        "shape": tuple(cfg_run.get("shape", [7, 7])),
        "determinism": cfg_run.get("determinism", 0.97),
        "beta": cfg_run.get("beta", 1.0),
        "chi_twist": summary.get("best_epsilon_actual", 0),
        "mean_free": summary.get("best_mean_free", 0),
        "mean_info": summary.get("best_mean_info", 0),
    }
