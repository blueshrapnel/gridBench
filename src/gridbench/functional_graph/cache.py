"""
Ported from gridFour src/analysis/functional_graph (2026-07-11); gridFour copy is frozen.

On-disk cache for per-σ functional-graph fingerprints.

Layout under schema-10 (see ``gridFour/docs/schema.md``):

    .../data-schema-10/reports/_cache/functional_graph/
        env_id=<env>/shape=<HxW>/det=<d>/fingerprints.parquet

One parquet per ``(env_id, shape, determinism)`` partition; rows are
unique per ``sigma_hash``.  ``sigma_hash`` is the canonical join key
to ``sigma_aggregates.csv`` and ``run.parquet``.

Schema-10 itself is **read-only** for this module: the cache is a
sidecar, schema-10 directories are never mutated.

The pure compute (``fingerprint_for_sigma``) lives in
:mod:`gridbench.functional_graph.fingerprint`; this module is the I/O
layer plus the hive walker that locates ``sigma.npy`` files to feed
into it.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as exc:                                 # pragma: no cover
    pa = None
    pq = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from gridbench.functional_graph.fingerprint import (
    FINGERPRINT_FIELDS,
    fingerprint_for_sigma,
)


# ---------------------------------------------------------------------------
# Schema and version
# ---------------------------------------------------------------------------

FINGERPRINT_VERSION = 7
"""Bumped when the fingerprint definition, per-label aggregation, or the
set of cached columns changes.

History:
    1 — initial schema (8 fp_* + per-σ metrics baked in).
    2 — added ``run_name``, ``is_run_best``, ``run_type``, and
        ``fitness_objective`` columns for layered scatter rendering
        in gridTwistFlask:
    3 — walls-inert decomposition: ``decompose()`` now accepts a
        ``walls`` argument and ``fingerprint_for_sigma`` threads
        ``env.walls_flat`` through.  Wall states no longer
        contribute basins (state 0 absorption via argmax-of-zeros),
        cycles (singleton self-loops), terminal nodes, or basin
        sizes.  Every fp_* field for wall-bearing envs (four_rooms,
        plus_cross, x_wall, pinwheel, wrap_pillar_3, pillar_*,
        corridor_*) changes between v2 and v3; envs with no walls
        (open_grid, wrap_grid) are unchanged.

        - ``run_type`` is the first path segment under the schema-10
          root (``"multi"`` for GA runs, ``"random"`` for random-
          search runs); lets the dashboard use real evaluated random
          σs as the baseline.
        - ``fitness_objective`` is read from the run's
          ``*-multi-all.summary.json`` / ``*-random-all.summary.json``
          (``"decision_information"`` / ``"free_energy"`` / null for
          random or pre-field runs).  Necessary because GA runs were
          historically optimised against either DI or F; coloring a
          mixed cohort by ``mean_free`` makes the DI-best run markers
          look incorrect.
    4 — wrong-side σ indexing fix (2026-05-22).
        ``fingerprint_for_sigma`` was indexing ``base_succ[sigma[:, ell],
        ...]`` but the env convention is ``sigma[s, a_phys] -> label``
        (grid_room.py:147-152), so per-label functional graphs must use
        ``sigma_inv[:, ell]``.  Pre-fix fp_* described σ⁻¹ per-label
        graphs, NOT the per-label graphs the paper figures show.
        Spearman ρ between v3 (wrong) and v4 (right) on a 200-σ four_rooms
        7×7 sample: n_basins +0.20, tail −0.01, cycle +0.23 — essentially
        uncorrelated.  Every wall AND wall-free env partition changes.
        Backups of v3 caches available at ``.bak-wrong-side-2026-05-22``.
    5 — added ``fp_largest_basin_fraction`` (2026-06-19): the per-σ
        coverage scalar of the twists-home-vectors paper,
        ``coverage(σ) = max_ℓ ( b_ℓ / |S°| )`` (largest basin across the
        four labels / non-wall states).  The existing eight fp_* values
        are unchanged from v4 — this bump only ADDS a column — so a v5
        row equals its v4 row plus the new coverage field.  Every
        partition is recomputed so the column is populated everywhere.
        Backups of v4 caches at ``.bak-fp4-2026-06-19``.
    6 — added ``elite_full_eval`` (bool) and ``elite_full_eval_pool``
        (int) run-provenance columns (2026-06-20): whether a GA run used
        the full-goal elite re-evaluation ("J") under goal subsampling,
        read from the run's ``*-multi-all.summary.json`` ``config`` block.
        Like v5 this bump only ADDS columns — the existing fp_* and metric
        values are unchanged, so a v6 row equals its v5 row plus the two
        new fields (False/0 for every run predating the feature).  Every
        partition is recomputed so the columns are populated everywhere.
        Backups of v5 caches at ``.bak-fp5-2026-06-20``.
    7 — added ``init_method`` (string) run-provenance column (2026-07-11,
        schema-11): the initialisation strategy of the producing run.
        Read from the run's ``init_method=<value>`` path partition
        segment, with the summary.json ``init_mode_lineage`` value
        (top level or ``config`` block) preferred when present —
        lineage-aware for seeded runs.  Like v5/v6 this bump only ADDS
        a column: the existing fp_* and metric values are unchanged.
        First version produced by the gridbench port (gridFour copy of
        this module is frozen at v6).

Consumers should filter on ``fp_version`` when joining so stale cache
rows are not silently mixed with current ones.
"""


# ---------------------------------------------------------------------------
# Random-twist baseline cache
# ---------------------------------------------------------------------------

RANDOM_BASELINE_VERSION = 1
"""Bumped when the random-baseline schema or sampling protocol changes."""


def _build_random_baseline_schema():
    if pa is None:
        return None
    fp_columns = [(name, pa.float64()) for name in FINGERPRINT_FIELDS]
    return pa.schema([
        ("env_id", pa.string()),
        ("shape_h", pa.int32()),
        ("shape_w", pa.int32()),
        ("determinism", pa.float64()),
        ("neighbourhood", pa.string()),
        ("state_dist", pa.string()),
        ("n_states", pa.int32()),
        ("n_actions", pa.int32()),
        ("sample_index", pa.int32()),
        ("random_seed", pa.int64()),
        *fp_columns,
        ("rb_version", pa.int32()),
        ("rb_computed_at_utc", pa.string()),
    ])


RANDOM_BASELINE_SCHEMA = _build_random_baseline_schema()
"""pyarrow schema for the (env, shape, det) random-twist baseline.  The
random baseline is β-independent — fp_* depends only on σ and the
env's bare transition kernel — so one parquet per env config covers
all β partitions of that env."""


def random_baseline_path(
    cache_root: Path | str,
    env_id: str,
    shape: tuple[int, int],
    determinism: float,
) -> Path:
    """Full path to the random-baseline parquet for one ``(env, shape, det)``."""
    h, w = int(shape[0]), int(shape[1])
    return (
        Path(cache_root)
        / f"env_id={env_id}"
        / f"shape={h}x{w}"
        / f"det={_format_det(determinism)}"
        / "random_baseline.parquet"
    )


def _build_schema():
    if pa is None:
        return None
    fp_columns = [(name, pa.float64()) for name in FINGERPRINT_FIELDS]
    return pa.schema([
        ("sigma_hash", pa.string()),
        ("env_id", pa.string()),
        ("shape_h", pa.int32()),
        ("shape_w", pa.int32()),
        ("determinism", pa.float64()),
        ("beta", pa.float64()),
        ("neighbourhood", pa.string()),
        ("state_dist", pa.string()),
        ("n_states", pa.int32()),
        ("n_actions", pa.int32()),
        # Run provenance — added at fp_version=2 for the layered-scatter
        # rendering in gridTwistFlask.
        ("run_type", pa.string()),
        ("run_name", pa.string()),
        ("is_run_best", pa.bool_()),
        ("fitness_objective", pa.string()),
        # Run provenance — added at fp_version=6.  Whether the GA run used
        # full-goal elite re-evaluation ("J") under goal subsampling, and the
        # pool size; False/0 for runs predating the feature.  See
        # load_run_elite_full_eval.
        ("elite_full_eval", pa.bool_()),
        ("elite_full_eval_pool", pa.int32()),
        # Run provenance — added at fp_version=7 (schema-11): initialisation
        # strategy of the producing run (lineage-aware for seeded runs).
        ("init_method", pa.string()),
        *fp_columns,
        # Per-σ metrics baked in at build time from sigma_aggregates.csv.
        # See gridFour/docs/schema.md for provenance.
        ("mean_free", pa.float64()),
        ("chi_twist", pa.float64()),
        ("mean_info", pa.float64()),
        ("mean_value", pa.float64()),
        ("fp_version", pa.int32()),
        ("fp_computed_at_utc", pa.string()),
    ])


FINGERPRINT_SCHEMA = _build_schema()
"""pyarrow schema for the per-σ fingerprint parquet.  See
``gridFour/docs/schema.md`` for the column reference."""


# ---------------------------------------------------------------------------
# Partition layout
# ---------------------------------------------------------------------------

def cache_root_for(schema_10_root: Path | str) -> Path:
    """Return ``<schema_10_root>/reports/_cache/functional_graph``."""
    return Path(schema_10_root) / "reports" / "_cache" / "functional_graph"


def partition_path(
    cache_root: Path | str,
    env_id: str,
    shape: tuple[int, int],
    determinism: float,
    beta: float,
) -> Path:
    """Directory for one ``(env, shape, det, beta)`` partition.

    Beta joins the partition because the baked-in metrics
    (``mean_free`` etc.) are β-dependent.  The ``fp_*`` columns
    themselves are β-independent.
    """
    h, w = int(shape[0]), int(shape[1])
    return (
        Path(cache_root)
        / f"env_id={env_id}"
        / f"shape={h}x{w}"
        / f"det={_format_det(determinism)}"
        / f"beta={_format_det(beta)}"
    )


def parquet_path(
    cache_root: Path | str,
    env_id: str,
    shape: tuple[int, int],
    determinism: float,
    beta: float,
) -> Path:
    """Full path to the parquet file for one partition."""
    return (
        partition_path(cache_root, env_id, shape, determinism, beta)
        / "fingerprints.parquet"
    )


def _format_det(det: float) -> str:
    """Match the existing schema-10 token convention (e.g. ``0.97`` not ``0.970``)."""
    return f"{float(det):g}"


# ---------------------------------------------------------------------------
# Hive walker — read-only over schema-10
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HiveSigmaEntry:
    """One σ found in a run's hive directory.

    Attributes:
        sigma_hash: the ``sigma_id=<hash>`` token from the hive path.
        sigma_path: absolute path to the ``sigma.npy`` file.
        env_id: from the partition token.
        shape: ``(h, w)`` from the partition token.
        determinism: from the partition token.
        neighbourhood: from the partition token.
        state_dist: from the partition token.
        run_dir: the ``run_name=...`` directory containing this hive.
        run_type: ``"multi"`` (GA) or ``"random"`` (random search),
            taken from the first path segment under the schema-10
            root; ``"unknown"`` if neither.
    """

    sigma_hash: str
    sigma_path: Path
    env_id: str
    shape: tuple[int, int]
    determinism: float
    neighbourhood: str
    state_dist: str
    run_dir: Path
    run_type: str


_PART_RE = re.compile(r"([a-zA-Z_]+)=(.+)")


def _parse_partition_tokens(path: Path) -> dict[str, str]:
    """Pull ``key=value`` partition tokens from any ``Path`` ancestor list."""
    tokens: dict[str, str] = {}
    for part in path.parts:
        match = _PART_RE.fullmatch(part)
        if match:
            tokens[match.group(1)] = match.group(2)
    return tokens


def walk_hive_sigmas(
    schema_10_root: Path | str,
    *,
    env_id: str | None = None,
    shape: tuple[int, int] | None = None,
    determinism: float | None = None,
    beta: float | None = None,
) -> Iterator[HiveSigmaEntry]:
    """Yield every ``sigma.npy`` found in a schema-10 hive layout.

    Read-only: opens nothing under ``schema_10_root`` other than to
    list directories and resolve paths.  Optional filters trim the
    walk to one ``(env_id, shape, det, beta)`` partition for the smoke
    test.

    Args:
        schema_10_root: e.g. ``/media/merlin/grid-twist/data-schema-10``.
        env_id: keep only paths whose ``env_id=`` token matches.
        shape: keep only paths whose ``shape=`` token matches ``HxW``.
        determinism: keep only paths whose ``det=`` token matches.
        beta: keep only paths whose ``beta=`` token matches.
    """
    root = Path(schema_10_root)
    for sigma_path in root.rglob("hive_sigma/**/sigma.npy"):
        tokens = _parse_partition_tokens(sigma_path)
        if "sigma_id" not in tokens:
            continue
        if env_id is not None and tokens.get("env_id") != env_id:
            continue
        if shape is not None and tokens.get("shape") != f"{shape[0]}x{shape[1]}":
            continue
        if determinism is not None and tokens.get("det") != _format_det(determinism):
            continue
        if beta is not None and tokens.get("beta") != _format_det(beta):
            continue

        shape_token = tokens.get("shape", "0x0")
        h, w = (int(x) for x in shape_token.split("x"))

        # Locate the run_name=... ancestor.
        run_dir = sigma_path
        while run_dir != root and not run_dir.name.startswith("run_name="):
            run_dir = run_dir.parent
        if run_dir == root:
            run_dir = sigma_path.parent           # fallback: file's own dir

        # First path segment under the schema-10 root tells us whether
        # this σ came from a GA run (``multi/``) or a random-search
        # run (``random/``).
        try:
            rel_parts = sigma_path.relative_to(root).parts
            run_type = rel_parts[0] if rel_parts else "unknown"
        except ValueError:
            run_type = "unknown"

        yield HiveSigmaEntry(
            sigma_hash=tokens["sigma_id"],
            sigma_path=sigma_path,
            env_id=tokens.get("env_id", "unknown"),
            shape=(h, w),
            determinism=float(tokens.get("det", "0")),
            neighbourhood=tokens.get("neighbourhood", "unknown"),
            state_dist=tokens.get("state_dist", "unknown"),
            run_dir=run_dir,
            run_type=run_type,
        )


# ---------------------------------------------------------------------------
# Build and read
# ---------------------------------------------------------------------------

def _utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Metric columns baked into each row from sigma_aggregates.csv.
METRIC_FIELDS: tuple[str, ...] = ("mean_free", "chi_twist", "mean_info", "mean_value")


def _summary_path(run_dir: Path) -> Path | None:
    """Find the per-run summary JSON, agnostic of multi/random suffix.

    Both GA (``*-multi-all.summary.json``) and random-search
    (``*-random-all.summary.json``) runs carry the same per-run
    fields (``best_sigma_hash``, ``fitness_objective``, ...).
    """
    for pattern in ("*-multi-all.summary.json", "*-random-all.summary.json"):
        candidates = list(Path(run_dir).glob(pattern))
        if candidates:
            return candidates[0]
    return None


def _load_summary(run_dir: Path) -> dict | None:
    """Read the run's summary.json into a dict, or ``None`` on failure."""
    path = _summary_path(run_dir)
    if path is None:
        return None
    try:
        with path.open() as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def load_run_best_sigma_hash(run_dir: Path) -> str | None:
    """Read the best sigma hash for a run (GA-best or random-best).

    Args:
        run_dir: a ``run_name=...`` directory from schema-10.

    Returns:
        The hash string, or ``None`` if the summary file is missing,
        unparseable, or has no ``best_sigma_hash``.
    """
    data = _load_summary(run_dir)
    if data is None:
        return None
    value = data.get("best_sigma_hash")
    return str(value) if value else None


def load_run_fitness_objective(run_dir: Path) -> str | None:
    """Read the run's fitness objective from its summary.json.

    Returns the string (typically ``"decision_information"`` or
    ``"free_energy"``), or ``None`` for random-search runs (no
    objective recorded) or pre-field GA runs.
    """
    data = _load_summary(run_dir)
    if data is None:
        return None
    value = data.get("fitness_objective")
    return str(value) if value else None


def _init_method_from_path(path: Path | str) -> str | None:
    """Extract the ``init_method=<value>`` partition segment from a path.

    Schema-10/11 store paths carry the producing run's initialisation
    strategy as a directory segment (e.g.
    ``multi/init_method=shuffle/...``).  Returns ``None`` when no such
    segment is present (pre-field layouts).
    """
    tokens = _parse_partition_tokens(Path(path))
    value = tokens.get("init_method")
    return str(value) if value else None


def load_run_init_method(run_dir: Path) -> str | None:
    """Initialisation strategy of the producing run.

    Prefers ``init_mode_lineage`` from the run's summary.json (top level
    or ``config`` block — lineage-aware for seeded runs), falling back
    to the ``init_method=<value>`` path partition segment of
    ``run_dir``.  Returns ``None`` when neither source is available;
    like ``fitness_objective`` the column is nullable, so ``None`` is
    stored as null.
    """
    data = _load_summary(run_dir)
    if data is not None:
        cfg = data.get("config")
        cfg = cfg if isinstance(cfg, dict) else {}
        value = data.get("init_mode_lineage", cfg.get("init_mode_lineage"))
        if value:
            return str(value)
    return _init_method_from_path(run_dir)


def load_run_elite_full_eval(run_dir: Path) -> tuple[bool, int]:
    """Read the run's full-goal elite re-evaluation ("J") settings.

    Unlike ``fitness_objective`` (promoted to the summary top level), the
    ``elite_full_eval`` / ``elite_full_eval_pool`` fields are only
    serialized inside the summary's ``config`` block (``vars(args)`` in
    evolution_multi_goal.optim).  Returns ``(False, 0)`` for random-search
    runs, runs that did not enable the feature, and any run predating it.

    Returns:
        ``(elite_full_eval, elite_full_eval_pool)``.
    """
    data = _load_summary(run_dir)
    if data is None:
        return (False, 0)
    cfg = data.get("config")
    cfg = cfg if isinstance(cfg, dict) else {}
    # Prefer a top-level value if a future writer promotes it; fall back to
    # the config block where the fields currently live.
    raw_on = data.get("elite_full_eval", cfg.get("elite_full_eval", False))
    raw_pool = data.get("elite_full_eval_pool", cfg.get("elite_full_eval_pool", 0))
    try:
        pool = int(raw_pool or 0)
    except (TypeError, ValueError):
        pool = 0
    return (bool(raw_on), pool)


def load_run_metrics(run_dir: Path) -> dict[str, dict[str, float]]:
    """Read ``sigma_aggregates.csv`` from a run directory.

    Returns a mapping ``sigma_hash -> {mean_free, chi_twist, mean_info,
    mean_value}``.  Missing file or missing columns produce an empty
    dict; missing per-row values surface as ``float('nan')``.

    Args:
        run_dir: a ``run_name=...`` directory from schema-10.

    Returns:
        Dict keyed by sigma_hash; values are dicts with the four
        metric float fields.
    """
    csv_path = Path(run_dir) / "sigma_aggregates.csv"
    if not csv_path.exists():
        return {}
    out: dict[str, dict[str, float]] = {}
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sigma_hash = row.get("sigma_hash", "")
            if not sigma_hash:
                continue
            out[sigma_hash] = {
                field: _safe_float(row.get(field)) for field in METRIC_FIELDS
            }
    return out


def _safe_float(value) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def compute_fingerprint_row(
    entry: HiveSigmaEntry,
    env,
    *,
    beta: float,
    metrics: dict[str, float] | None = None,
    is_run_best: bool = False,
    fitness_objective: str | None = None,
    elite_full_eval: bool = False,
    elite_full_eval_pool: int = 0,
    init_method: str | None = None,
) -> dict:
    """Fingerprint one hive entry against a prepared env.

    The caller builds ``env`` consistently with ``entry.env_id /
    shape / determinism / neighbourhood`` (untwisted, no absorbing
    goal — see ``wrap_grid_helpers.build_twisted_env_no_goal``) and
    supplies the partition's ``beta``, the σ's metric row from
    :func:`load_run_metrics`, and whether this σ is the run's GA-best
    (from :func:`load_run_best_sigma_hash`).  ``init_method`` may be
    passed explicitly; when omitted it is resolved from
    ``entry.run_dir`` via :func:`load_run_init_method`.

    Returns a dict matching :data:`FINGERPRINT_SCHEMA`.
    """
    sigma = np.load(entry.sigma_path)
    fp = fingerprint_for_sigma(env, sigma)
    metrics = metrics or {}
    run_name = entry.run_dir.name
    if run_name.startswith("run_name="):
        run_name = run_name[len("run_name="):]
    return {
        "sigma_hash": entry.sigma_hash,
        "env_id": entry.env_id,
        "shape_h": int(entry.shape[0]),
        "shape_w": int(entry.shape[1]),
        "determinism": float(entry.determinism),
        "beta": float(beta),
        "neighbourhood": entry.neighbourhood,
        "state_dist": entry.state_dist,
        "n_states": int(env.nS),
        "n_actions": int(env.nA),
        "run_type": entry.run_type,
        "run_name": run_name,
        "is_run_best": bool(is_run_best),
        "fitness_objective": fitness_objective,
        "elite_full_eval": bool(elite_full_eval),
        "elite_full_eval_pool": int(elite_full_eval_pool),
        "init_method": (
            init_method if init_method is not None
            else load_run_init_method(entry.run_dir)
        ),
        **fp,
        "mean_free": _safe_float(metrics.get("mean_free")),
        "chi_twist": _safe_float(metrics.get("chi_twist")),
        "mean_info": _safe_float(metrics.get("mean_info")),
        "mean_value": _safe_float(metrics.get("mean_value")),
        "fp_version": int(FINGERPRINT_VERSION),
        "fp_computed_at_utc": _utc_stamp(),
    }


def write_partition(rows: list[dict], parquet_path: Path) -> int:
    """Write ``rows`` to ``parquet_path`` atomically.

    Returns the number of rows written.  Idempotent on identical
    inputs and identical ``FINGERPRINT_VERSION``.
    """
    if pa is None or pq is None:                          # pragma: no cover
        raise RuntimeError(
            f"pyarrow is required to write the fingerprint cache. {_IMPORT_ERROR!r}"
        )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Write an empty table so the partition's existence is recorded.
        table = FINGERPRINT_SCHEMA.empty_table()
    else:
        table = pa.Table.from_pylist(rows, schema=FINGERPRINT_SCHEMA)
    tmp = parquet_path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(parquet_path)
    return table.num_rows


def read_partition(parquet_path: Path):
    """Read a previously written fingerprint partition into a pyarrow Table.

    Reads the single file directly rather than via dataset discovery so
    the path-derived partition columns (``env_id=...`` etc.) don't
    clash with the columns of the same name inside the file.
    """
    if pq is None:                                        # pragma: no cover
        raise RuntimeError(
            f"pyarrow is required to read the fingerprint cache. {_IMPORT_ERROR!r}"
        )
    return pq.ParquetFile(parquet_path).read()


def existing_sigma_hashes(parquet_path: Path) -> set[str]:
    """Sigma hashes already cached at the current ``FINGERPRINT_VERSION``."""
    if not parquet_path.exists() or pq is None:
        return set()
    table = pq.ParquetFile(parquet_path).read(columns=["sigma_hash", "fp_version"])
    df = table.to_pydict()
    return {
        h for h, v in zip(df["sigma_hash"], df["fp_version"])
        if int(v) == FINGERPRINT_VERSION
    }
