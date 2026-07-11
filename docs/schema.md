# Data store schema — schema-11 (and the schema-10 it supersedes)

Ported from `gridFour/docs/schema.md` (2026-07-11), which remains as the
frozen schema-10 reference.  This document is the live one: it defines
**schema-11**, the successor layout, and the manifest/fingerprint schema
changes that motivate it.

## Why schema-11 exists

Schema-10 partitions canonical runs by `init_method` but none of the
derived surfaces carried that field: the `schema10-export` pickle pool
flattened the partitions, and neither the export provenance dict nor
`fingerprints.parquet` had an `init_method` (or, at manifest level, a
`fitness_objective`) column.  Consequence, discovered 2026-07-10/11: a
row-shuffle, decision-information exemplar leaked into a
permutation-balanced free-energy paper cohort — and the two init
strategies turn out to produce *different solution families* (row-shuffle
runs retire an action label 17/18 and beat their perm-balanced twins on
fitness 11/13; perm-balanced runs never retire, 0/17).  Init strategy and
fitness objective are therefore first-class cohort axes and must be
non-optional at every level: paths, manifests, exports, and analytics
caches.

## Schema-11 layout

```
data-schema-11/
  multi/
    init_method=<perm_balanced|shuffle|...>/
      fitness_objective=<free_energy|decision_information|legacy_unknown>/
        env_id=<env>/shape=<HxW>/beta=<b>/det=<d>/run_name=<run>/   # per-run payload, unchanged
  random/
    init_method=<...>/env_id=/shape=/beta=/det=/run_name=/          # no objective level (no fitness)
  export/
    init_method=<...>/fitness_objective=<...>/shape=/env_id=/.../collated-*.pickle
  reports/
    _cache/functional_graph/env_id=/shape=/det=/beta=/fingerprints.parquet
    multi/..., import-manifests/...                                  # as schema-10
```

Changes vs schema-10:

1. **`fitness_objective` promoted to a path level** under `multi/`,
   between `init_method` and `env_id`.  Value comes from the run's
   `summary.json`; pre-field GA runs go to `legacy_unknown` rather than
   being silently pooled.
2. **Export partitioned like the store.**  `schema10-export`'s flat pool
   is replaced by `export/` mirroring the two provenance levels, and
   every collated pickle's `provenance` dict gains `init_method` and
   `fitness_objective` keys.  Cohort globs are then clean by
   construction; a `**` glob that crosses partitions has to say so.
3. **Analytics columns.**  `run.parquet` manifest goes to
   `schema_version = 10`: adds `init_method` (string) and
   `fitness_objective` (string, nullable for random).
   `FINGERPRINT_SCHEMA` gains `init_method` (string) at the next
   `fp_version` bump; `fitness_objective` is already present since v2.
4. **Seeded-run lineage rule.**  For runs started from a prior
   population (`--seed-population`), `init_method` is the *lineage* —
   the init_method of the run that produced the seed population — NOT
   the CLI `init_mode`, which the seeding path ignores but which still
   lands in `summary.json` (`init_mode='...' ignored` in the log).  The
   schema-10 importer (`import_run_to_ga_root.py:_build_destination`)
   partitions by `summary.config.init_mode` and would mis-file such
   runs; the schema-11 importer must prefer lineage when
   `seed_population_path` is set, and manifests must set per-run
   `init_mode` to lineage (playbook supports per-run override since
   2026-07-11).  Known affected runs (recorded `init_mode=shuffle`,
   true lineage `perm_balanced`): the three
   `g2000tot-pop-96-perm-bal-warm-11-07-...-s173/s316/s360` warm
   extensions.
5. **Everything else is unchanged**: per-run payload layout, eps
   binning, determinism-token note, and the derived-cache contract
   (`reports/_cache/` is disposable and rebuildable) carry over from
   schema-10 verbatim — see the schema-10 reference below.

## Reader gotcha: hive paths vs physical columns

`run.parquet` (v10) carries `init_method`/`fitness_objective` as physical
columns while also living under `init_method=…/fitness_objective=…` path
segments.  pyarrow's default `pq.read_table(path)` hive-infers partition
columns from the path and fails with a type-merge error on the collision.
Consumers must read manifests with `pq.ParquetFile(p).read()` or
`pq.read_table(p, partitioning=None)` (both verified).  Legacy consumers
that relied on the *inferred* `init_method` column from v9 paths get the
same column name, now physical and string-typed.  Audit consumers for
this during migration.

## Code ownership after the option-b port (2026-07-11)

- **gridTwist `src/schema_store/`** — the store schema: `run_artifacts`
  (MANIFEST_SCHEMA, schema_version 10), `metrics`, `env_bundle`.
  `export_schema10.py` (with `--layout schema11`) and
  `import_run_to_ga_root.py` (lineage-aware, `--init-method-override`,
  default root `data-schema-11`) consume it.  Branch
  `port/schema-store-v10`.
- **gridBench `src/gridbench/functional_graph/`** — ALL analysis code:
  `decomposition`, `fingerprint`, `probe_env` (now on `gridcore.envs`),
  `cache` (FINGERPRINT_VERSION 7, `init_method` column, lineage-aware
  via `load_run_init_method`).  Parity-verified against the frozen
  gridFour originals (identical T matrices and fingerprints).
  `pip install -e gridBench` (deps: numpy, pyarrow, local gridcore).
- **gridFour** — frozen; its `utility/` and `analysis/functional_graph/`
  copies stay at schema_version 9 / fp_version 6 for reproducing
  published figures only.
- **gridCore** — unchanged, kernel-only (it already carried the env
  classes `probe_env` needed).

## Migration (schema-10 → schema-11)

Constraints: the store is ~377G / ~13.8M files on ext4; a byte copy is
neither affordable nor needed.

1. **Backup first.**  `gridOps/scripts/desktop/backup-schema10` writes
   per-partition tar+zstd archives with stream sha256 and `zstd -t`
   verification to `/media/panther/backup/schema10-<date>/`.  Migration
   does not start until every partition reads `verified` in its
   `MANIFEST.tsv`.
2. **Hardlink projection.**  `data-schema-11/` is built next to
   schema-10 on the same filesystem with `cp -al` per run directory
   (hardlinks: new directory entries, zero data duplication).  The
   mapping run → `(init_method, fitness_objective)` comes from the
   schema-10 partition (init) and the run's `summary.json` (objective);
   the migration script writes a `MIGRATION-MAP.tsv` (src → dst) so the
   projection is auditable and reversible.
3. **Rebuild derived surfaces** under schema-11: export pickles
   regenerated with the enriched provenance; fingerprints parquet
   rebuilt with the `init_method` column (bump `fp_version`).
4. **Freeze schema-10.**  After validation (row counts, spot-check
   sigma_hash joins), schema-10 becomes read-only (`chmod -R a-w` on the
   canonical trees) and is retired from analytics; it is not deleted
   while any consumer still points at it.  Space is reclaimed later by
   removing the schema-10 *directory tree* only (the payload bytes
   survive via the schema-11 hardlinks).

## Cohort conventions (paper analytics)

- **Production cohort**: `init_method=perm_balanced`, objectives pooled
  only where the pooling is justified and stated.
- **Baseline arm**: `init_method=shuffle` prod-baseline runs (6 at
  schema-10 close) — exemplar use only, origin always disclosed.
- **Matched init batches** (silence-hunt 10-07/11-07 families): the
  controlled init comparison; import these into schema-11 directly
  rather than schema-10.
- K≠1 (`goal_subsample_fraction < 1`) runs stay out of the store
  entirely (annealing-paper-only), as before.

---

# Appendix: schema-10 reference (frozen)

# Hive manifest schema

This project stores one Parquet manifest per run under the schema-10 hive root:

```
.../shape=.../env_id=.../neighbourhood=.../state_dist=.../det=.../seed=.../eps=.../beta=.../run_id=.../run.parquet
```

The manifest schema lives in `gridFour/src/utility/run_artifacts.py`
(`MANIFEST_SCHEMA`).
Storage generation: **schema-10**.
Manifest table version (`schema_version` column): **9** (schema-11 bumps
this to 10; see above).

The full schema-10 column tables, eps-binning and determinism-token
notes, and the functional-graph fingerprint cache documentation remain
in `gridFour/docs/schema.md` and apply to schema-11 unchanged except as
amended above (added columns; export partitioning).
