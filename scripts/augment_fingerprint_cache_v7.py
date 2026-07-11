#!/usr/bin/env python3
"""Build the schema-11 fingerprint cache (fp_version 7) by augmentation.

fp_version 7 = fp_version 6 + the init_method provenance column; the
fingerprint values themselves are unchanged, so the v7 cache is built by
AUGMENTING the frozen schema-10 parquets rather than recomputing 3M rows:

  - new home:  <store>/cache/functional_graph/env_id=*/shape=*/det=*/beta=*/
               fingerprints.parquet   (out of reports/_cache: it is a
               derived cache, not a report)
  - init_method + fitness_objective_store joined per row via run_name
    against the schema-11 canonical partitions (single source of truth);
  - rows whose run_name is NOT in schema-11 (the excluded FEP+M family)
    are dropped, keeping cache and store consistent;
  - fp_version stamped 7 on every surviving row.

Old parquets under data-schema-10/reports/_cache stay untouched (frozen
history).  Consumers repoint to <store>/cache and read with
partitioning=None (hive-path collision note in gridBench/docs/schema.md).

Usage: augment_fingerprint_cache_v7.py [--source-cache PATH] [--store PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def store_partition_map(store: Path) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for run_dir in store.glob("multi/init_method=*/fitness_objective=*/env_id=*/shape=*/beta=*/det=*/run_name=*"):
        parts = {p.split("=", 1)[0]: p.split("=", 1)[1] for p in run_dir.parts if "=" in p}
        out[parts["run_name"]] = (parts["init_method"], parts["fitness_objective"])
    for run_dir in store.glob("random/init_method=*/env_id=*/shape=*/beta=*/det=*/run_name=*"):
        parts = {p.split("=", 1)[0]: p.split("=", 1)[1] for p in run_dir.parts if "=" in p}
        out[parts["run_name"]] = (parts["init_method"], "")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-cache", type=Path,
                    default=Path("/media/merlin/grid-twist/data-schema-10/reports/_cache/functional_graph"))
    ap.add_argument("--store", type=Path, default=Path("/media/merlin/grid-twist/data-schema-11"))
    args = ap.parse_args()

    runs = store_partition_map(args.store.resolve())
    print(f"store map: {len(runs)} run_names")

    dest_root = args.store / "cache" / "functional_graph"
    total_in = total_out = dropped = 0
    files = sorted(args.source_cache.rglob("fingerprints.parquet"))
    for i, src in enumerate(files, 1):
        rel = src.relative_to(args.source_cache)
        table = pq.ParquetFile(src).read()
        n = table.num_rows
        total_in += n
        names = table.column("run_name").to_pylist()
        keep, init_col, fo_col = [], [], []
        for j, rn in enumerate(names):
            hit = runs.get(rn or "")
            if hit is None:
                continue  # excluded from schema-11 (FEP+M family) or unknown
            keep.append(j)
            init_col.append(hit[0])
            fo_col.append(hit[1] or None)
        dropped += n - len(keep)
        if not keep:
            continue
        sub = table.take(keep)
        # place init_method after elite_full_eval_pool (end of provenance block)
        insert_at = sub.schema.get_field_index("elite_full_eval_pool") + 1
        sub = sub.add_column(insert_at, pa.field("init_method", pa.string()),
                             pa.array(init_col, type=pa.string()))
        sub = sub.add_column(insert_at + 1, pa.field("fitness_objective_store", pa.string()),
                             pa.array(fo_col, type=pa.string()))
        fpv = pa.array([7] * sub.num_rows, type=pa.int32())
        sub = sub.set_column(sub.schema.get_field_index("fp_version"),
                             pa.field("fp_version", pa.int32()), fpv)
        dest = dest_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(sub, dest)
        total_out += sub.num_rows
        if i % 10 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] rows in={total_in} out={total_out} dropped={dropped}", flush=True)
    print(f"done: {len(files)} parquets -> {dest_root}")
    print(f"rows: {total_in} in, {total_out} out, {dropped} dropped (not in schema-11)")
    return 0


if __name__ == "__main__":
    main()
