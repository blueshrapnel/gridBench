"""Expressive-range data loading over the run-manifest parquet store.

Ported from gridFour ``utility/expressive_range_helpers.py`` (the data half;
the plotting half lives in ``gridvis.expressive_range``).  These are scans over
``run.parquet`` manifests, so ``init_method`` and ``fitness_objective`` — the
schema-11 first-class fields — are filterable here (schema_version 10 manifests
carry both columns).
"""
from pathlib import Path
from typing import Iterable, Union

import pyarrow.dataset as ds


def load_manifest_dataset(root: Union[str, Path, Iterable[Union[str, Path]]]):
    """
    Build a Parquet dataset from manifest files only (run.parquet).

    This avoids accidental ingestion of non-Parquet files (e.g. logs/*.tsv)
    when pointing at a schema root.

    Note (schema-11): pass explicit run.parquet files or a root to rglob; do
    NOT let pyarrow infer hive partitioning off the ``init_method=`` path
    segments (that collides with the stored columns).
    """
    if isinstance(root, (str, Path)):
        root_path = Path(root).expanduser()
        if root_path.is_file():
            files = [str(root_path)]
        else:
            files = [str(p) for p in root_path.rglob("run.parquet")]
    else:
        files = [str(Path(p).expanduser()) for p in root]

    if not files:
        raise FileNotFoundError(f"No run.parquet files found for root={root}")
    return ds.dataset(files, format="parquet")


def load_runs(dataset, filter_expr):
    table = dataset.to_table(filter=filter_expr)
    return table.to_pandas()


def _value_expr(field_name, values):
    if values is None:
        return None
    if isinstance(values, (list, tuple, set)):
        values = list(values)
        if not values:
            return None
        expr = ds.field(field_name) == values[0]
        for val in values[1:]:
            expr = expr | (ds.field(field_name) == val)
        return expr
    return ds.field(field_name) == values


def filter_runs(shape=None, env_id=None, neighbourhood=None, state_dist=None,
                determinism=None, epsilon=None, beta=None,
                init_method=None, fitness_objective=None):
    exprs = []
    if shape is not None:
        h, w = shape
        exprs.append(ds.field('shape_h') == h)
        exprs.append(ds.field('shape_w') == w)
    for field, values in [
        ('env_id', env_id),
        ('neighbourhood', neighbourhood),
        ('state_dist', state_dist),
        ('determinism', determinism),
        ('epsilon', epsilon),
        ('beta', beta),
        ('init_method', init_method),
        ('fitness_objective', fitness_objective),
    ]:
        value_expr = _value_expr(field, values)
        if value_expr is not None:
            exprs.append(value_expr)
    if not exprs:
        return None
    expr = exprs[0]
    for extra in exprs[1:]:
        expr = expr & extra
    return expr


PARAM_COLUMNS = ['shape_h', 'shape_w', 'env_id', 'neighbourhood', 'state_dist',
                 'determinism', 'epsilon', 'beta', 'init_method', 'fitness_objective']


def unique_parameter_values(dataset, filter_expr=None, columns=PARAM_COLUMNS):
    # Tolerate manifests that predate a column (init_method/fitness_objective).
    available = [c for c in columns if c in dataset.schema.names]
    table = dataset.to_table(filter=filter_expr, columns=available)
    df = table.to_pandas()
    summary = {}
    if 'shape_h' in df.columns and 'shape_w' in df.columns:
        shapes = {(int(h), int(w)) for h, w in zip(df['shape_h'], df['shape_w'])}
        summary['shape'] = sorted(shapes)
    for col in available:
        if col in df.columns:
            values = sorted(df[col].dropna().unique().tolist())
            summary[col] = values
    return summary


def display_parameter_summary(dataset, filter_kwargs=None):
    filter_kwargs = filter_kwargs or {}
    if 'eps' in filter_kwargs and 'epsilon' not in filter_kwargs:
        filter_kwargs['epsilon'] = filter_kwargs.pop('eps')
    filter_expr = filter_runs(**filter_kwargs) if filter_kwargs else None
    summary = unique_parameter_values(dataset, filter_expr)
    for key, values in summary.items():
        print(f"{key}: {values}")
