"""Data-store layout + DATA_ROOT resolution for the schema-11 gridTwist store.

Single home for partition-path knowledge.  Schema-11 promoted both
``init_method`` and ``fitness_objective`` to first-class path levels

    <root>/multi/init_method=<X>/fitness_objective=<Y>/env_id=<e>/
           shape=<HxW>/beta=<b>/det=<d>/run_name=<run>/

so every consumer must state both explicitly — nothing here assumes a single
init strategy or objective.

``DATA_ROOT`` is configurable via the ``GRIDBENCH_DATA_ROOT`` environment
variable.  This is the dual-mode switch: notebooks run against the live lab
store by default, or against a frozen, self-contained paper bundle by setting
``GRIDBENCH_DATA_ROOT`` to the bundle's local ``data/`` directory.

Reads of per-run payload (summaries, ``hive_sigma/**``, ``sigma_aggregates``)
stay in their consumers: they recurse *under* an already-resolved ``run_dir``,
and the schema-11 levels sit *above* ``run_dir``, so they are unaffected.  Only
partition-path *assembly* belongs here.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_ROOT = "/media/merlin/grid-twist/data-schema-11"


def data_root() -> Path:
    """The store root, overridable via ``GRIDBENCH_DATA_ROOT`` (dual-mode)."""
    return Path(os.environ.get("GRIDBENCH_DATA_ROOT", DEFAULT_DATA_ROOT))


def _shape_token(shape) -> str:
    """Normalise a shape to its ``HxW`` path token.  Accepts ``'7x7'`` or ``(7, 7)``."""
    if isinstance(shape, (tuple, list)):
        return f"{int(shape[0])}x{int(shape[1])}"
    return str(shape)


def cohort_dir(
    *,
    init_method: str,
    fitness_objective: str,
    env_id: str,
    shape,
    beta,
    det,
    root: Path | str | None = None,
) -> Path:
    """The ``det=``-level directory holding the ``run_name=`` dirs of one cohort.

    ``beta`` / ``det`` are interpolated verbatim (pass the exact path tokens,
    e.g. ``beta='1'``, ``det='0.97'``) to avoid float-formatting drift
    (``1`` vs ``1.0``).
    """
    base = Path(root) if root is not None else data_root()
    return (
        base
        / "multi"
        / f"init_method={init_method}"
        / f"fitness_objective={fitness_objective}"
        / f"env_id={env_id}"
        / f"shape={_shape_token(shape)}"
        / f"beta={beta}"
        / f"det={det}"
    )


def run_dir(*, run_name: str, root: Path | str | None = None, **cohort) -> Path:
    """Assemble a schema-11 ``multi/`` run directory path.

    ``cohort`` = the :func:`cohort_dir` fields (init_method, fitness_objective,
    env_id, shape, beta, det).
    """
    return cohort_dir(root=root, **cohort) / f"run_name={run_name}"


def multi_root(root: Path | str | None = None) -> Path:
    """The ``multi/`` root; use with explicit init_method/fitness_objective when scanning."""
    base = Path(root) if root is not None else data_root()
    return base / "multi"
