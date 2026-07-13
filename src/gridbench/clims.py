"""Pooled per-family colour/axis limits from the functional-graph fingerprint cache.

A *family* is ``(env_id, shape, beta)`` (optionally pinned to a determinism).
Limits are **pooled across all ``init_method`` and ``fitness_objective``
cohorts** in the family, so different init strategies land on one comparable
scale (Karen's "common clims basis" for comparing init strategies).

Source: ``<DATA_ROOT>/cache/functional_graph/**/fingerprints.parquet`` (fp v7),
which carries ``init_method``/``fitness_objective`` + ``mean_free``/``mean_info``/
``mean_value``/``chi_twist``/``fp_*`` per sigma.
"""
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

from gridbench import store

DEFAULT_METRIC_COLUMNS = (
    "mean_free", "mean_info", "mean_value", "chi_twist",
    "fp_n_basins", "fp_largest_basin_fraction", "fp_cycle_basin_ratio",
)


def _fingerprint_cache_root(root=None) -> Path:
    base = Path(root) if root is not None else store.data_root()
    return base / "cache" / "functional_graph"


def pooled_family_clims(env_id, shape, beta, *, det=None,
                        columns=DEFAULT_METRIC_COLUMNS, root=None) -> dict:
    """Return ``{column: (vmin, vmax)}`` pooled over the (env, shape, beta) family.

    Pools every ``init_method``/``fitness_objective`` row in the family so the
    limits are common across init strategies.  A column maps to ``None`` if it
    is absent from the cache or has no finite values.  Returns ``{}`` if no
    fingerprint parquet is found.
    """
    if isinstance(shape, (tuple, list)):
        sh, sw = int(shape[0]), int(shape[1])
    else:
        sh, sw = (int(v) for v in str(shape).split("x"))

    files = [str(p) for p in _fingerprint_cache_root(root).rglob("fingerprints.parquet")]
    if not files:
        return {}
    dset = ds.dataset(files, format="parquet")

    flt = ((ds.field("env_id") == env_id)
           & (ds.field("shape_h") == sh)
           & (ds.field("shape_w") == sw)
           & (ds.field("beta") == float(beta)))
    if det is not None:
        flt = flt & (ds.field("determinism") == float(det))

    cols = [c for c in columns if c in dset.schema.names]
    if not cols:
        return {}
    df = dset.to_table(filter=flt, columns=cols).to_pandas()

    out = {}
    for c in cols:
        vals = df[c].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        out[c] = (float(vals.min()), float(vals.max())) if vals.size else None
    return out
