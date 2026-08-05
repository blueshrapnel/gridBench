"""Relevant GridFour analysis invariants retained in GridBench."""

import json

import numpy as np
import pytest

from gridbench.functional_graph import cache
from gridbench.functional_graph.null_models import (
    classify_null_model,
    mapping_null,
    permutation_null,
    random_twist_baseline,
    sample_random_twist,
)
from gridbench.functional_graph.probe_env import build_goal_free_probe_env
from gridbench.functional_graph.single_label_reach import reach_sample, trajectory


def test_permutation_null_closed_form():
    null = permutation_null(49)
    assert null.n_basins == pytest.approx(sum(1 / k for k in range(1, 50)))
    assert null.n_basins * null.mean_cycle_length == pytest.approx(49)
    assert null.n_cyclic == 49
    assert null.n_terminal == null.mean_tail_length == 0


def test_mapping_null_reference_numbers():
    null = mapping_null(49)
    assert null.n_basins == pytest.approx(1.95, abs=0.05)
    assert null.n_cyclic == pytest.approx(8.8, abs=0.1)
    assert null.n_terminal == pytest.approx(18.0, abs=0.5)


@pytest.mark.parametrize("factory", [permutation_null, mapping_null])
def test_closed_form_nulls_reject_non_positive_size(factory):
    with pytest.raises(ValueError):
        factory(0)


def test_null_classification_uses_environment_dynamics():
    torus = build_goal_free_probe_env("wrap_grid", (7, 7), 0.97)
    rooms = build_goal_free_probe_env("four_rooms", (7, 7), 0.97)
    assert classify_null_model(torus) == "permutation"
    assert classify_null_model(rooms) == "mapping"


def test_random_twist_sampling_is_valid_and_seeded():
    first = sample_random_twist(20, 4, np.random.default_rng(7))
    second = sample_random_twist(20, 4, np.random.default_rng(7))
    assert np.array_equal(first, second)
    assert np.all(np.sort(first, axis=1) == np.arange(4))


def test_random_baseline_shapes_and_seed_reproducibility():
    env = build_goal_free_probe_env("wrap_grid", (7, 7), 0.97)
    first = random_twist_baseline(env, n_samples=8, rng=np.random.default_rng(42))
    second = random_twist_baseline(env, n_samples=8, rng=np.random.default_rng(42))
    for field, values in first.items():
        assert values.shape == (8, 4)
        assert np.array_equal(values, second[field])


def test_random_torus_twists_have_expected_terminal_fraction():
    env = build_goal_free_probe_env("wrap_grid", (7, 7), 0.97)
    baseline = random_twist_baseline(
        env, n_samples=200, rng=np.random.default_rng(0)
    )
    terminal_fraction = baseline["n_terminal"].mean() / env.nS
    assert 0.25 < terminal_fraction < 0.40


def test_random_baseline_rejects_zero_samples():
    env = build_goal_free_probe_env("wrap_grid", (3, 3), 1.0)
    with pytest.raises(ValueError):
        random_twist_baseline(env, n_samples=0)


def test_cache_schema_and_version_include_current_provenance():
    names = set(cache.FINGERPRINT_SCHEMA.names)
    assert cache.FINGERPRINT_VERSION == 7
    assert {"sigma_hash", "init_method", "elite_full_eval", "elite_full_eval_pool"} <= names
    assert set(cache.FINGERPRINT_FIELDS) <= names


def test_cache_partition_paths_and_parser(tmp_path):
    root = cache.cache_root_for(tmp_path / "schema-10")
    path = cache.partition_path(root, "wrap_grid", (7, 7), 0.97, beta=1.0)
    assert path.parts[-4:] == (
        "env_id=wrap_grid", "shape=7x7", "det=0.97", "beta=1"
    )
    tokens = cache._parse_partition_tokens(
        tmp_path / "multi" / "init_method=shuffle" / "run_name=foo"
    )
    assert tokens["init_method"] == "shuffle"
    assert tokens["run_name"] == "foo"


def test_run_provenance_prefers_summary_lineage(tmp_path):
    run_dir = tmp_path / "init_method=shuffle" / "run_name=foo"
    run_dir.mkdir(parents=True)
    (run_dir / "foo-multi-all.summary.json").write_text(
        json.dumps({
            "best_sigma_hash": "deadbeef",
            "fitness_objective": "free_energy",
            "init_mode_lineage": "perm_balanced",
            "config": {"elite_full_eval": True, "elite_full_eval_pool": 4},
        })
    )
    assert cache.load_run_best_sigma_hash(run_dir) == "deadbeef"
    assert cache.load_run_fitness_objective(run_dir) == "free_energy"
    assert cache.load_run_init_method(run_dir) == "perm_balanced"
    assert cache.load_run_elite_full_eval(run_dir) == (True, 4)


def test_hive_walker_filters_and_recovers_run_metadata(tmp_path):
    sigma_dir = (
        tmp_path
        / "multi"
        / "init_method=perm_balanced"
        / "env_id=wrap_grid"
        / "shape=7x7"
        / "beta=1"
        / "det=0.97"
        / "run_name=fake-r1"
        / "hive_sigma"
        / "shape=7x7"
        / "env_id=wrap_grid"
        / "neighbourhood=manhattan"
        / "state_dist=uniform"
        / "det=0.97"
        / "beta=1"
        / "sigma_id=deadbeef"
    )
    sigma_dir.mkdir(parents=True)
    np.save(sigma_dir / "sigma.npy", np.tile(np.arange(4), (49, 1)))

    entries = list(
        cache.walk_hive_sigmas(
            tmp_path,
            env_id="wrap_grid",
            shape=(7, 7),
            determinism=0.97,
            beta=1.0,
        )
    )
    assert len(entries) == 1
    entry = entries[0]
    assert entry.sigma_hash == "deadbeef"
    assert entry.run_type == "multi"
    assert entry.run_dir.name == "run_name=fake-r1"

    assert not list(cache.walk_hive_sigmas(tmp_path, env_id="four_rooms"))


def test_compute_fingerprint_row_uses_core_fingerprint_and_lineage(tmp_path):
    sigma_dir = tmp_path / "init_method=shuffle" / "run_name=fake" / "sigma_id=abc"
    sigma_dir.mkdir(parents=True)
    sigma_path = sigma_dir / "sigma.npy"
    np.save(sigma_path, np.tile(np.arange(4), (49, 1)))
    entry = cache.HiveSigmaEntry(
        sigma_hash="abc",
        sigma_path=sigma_path,
        env_id="wrap_grid",
        shape=(7, 7),
        determinism=0.97,
        neighbourhood="manhattan",
        state_dist="uniform",
        run_dir=sigma_dir.parent,
        run_type="multi",
    )
    env = build_goal_free_probe_env("wrap_grid", (7, 7), 0.97)
    row = cache.compute_fingerprint_row(
        entry,
        env,
        beta=1.0,
        metrics={"mean_free": 6.2, "chi_twist": 0.5},
        is_run_best=True,
    )
    assert row["fp_n_basins"] == pytest.approx(7.0)
    assert row["fp_largest_basin_fraction"] == pytest.approx(1 / 7)
    assert row["init_method"] == "shuffle"
    assert row["is_run_best"] is True
    assert row["mean_free"] == pytest.approx(6.2)


def test_cache_partition_roundtrip(tmp_path):
    row = {name: None for name in cache.FINGERPRINT_SCHEMA.names}
    row.update({
        "sigma_hash": "deadbeef",
        "env_id": "wrap_grid",
        "shape_h": 7,
        "shape_w": 7,
        "determinism": 0.97,
        "beta": 1.0,
        "n_states": 49,
        "n_actions": 4,
        "is_run_best": False,
        "elite_full_eval": False,
        "elite_full_eval_pool": 0,
        "fp_version": cache.FINGERPRINT_VERSION,
    })
    path = cache.parquet_path(
        cache.cache_root_for(tmp_path), "wrap_grid", (7, 7), 0.97, 1.0
    )
    assert cache.write_partition([row], path) == 1
    assert cache.read_partition(path).num_rows == 1
    assert cache.existing_sigma_hashes(path) == {"deadbeef"}


def test_reach_sample_excludes_walls_and_clips_walkable_values():
    env = build_goal_free_probe_env("four_rooms", (7, 7), 0.97)
    sample = reach_sample(env, label=0, cutoff=2)
    walls = set(env.walls_flat)
    assert all(sample.rho[state] < 1 for state in walls)
    assert all(1 <= sample.rho[state] <= 2 for state in range(env.nS) if state not in walls)


def test_trajectory_stops_before_first_revisit():
    env = build_goal_free_probe_env("wrap_grid", (3, 3), 1.0)
    path = trajectory(env, label=1, start=0, max_steps=20)
    assert path.tolist() == [0, 1, 2]
    assert len(path) == len(set(path.tolist()))
