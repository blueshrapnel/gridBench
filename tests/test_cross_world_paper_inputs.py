"""Tests for reproducible cross-world null and floor inputs."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gridbench.papers.cross_world_floors import (
    qualifying_floor_score,
    select_floor_candidate,
)
from gridbench.papers.cross_world_null import validated_free_energy_mean


def test_validated_free_energy_mean_records_finite_diagnostics():
    di = SimpleNamespace(
        converged=True,
        iteration_count=17,
        last_blahut_residual=2.5e-8,
    )
    mean, iterations, residual = validated_free_energy_mean(
        di, np.array([1.0, 2.0, 3.0]), [0, 2], context="draw=1 goal=2"
    )
    assert mean == 2.0
    assert iterations == 17
    assert residual == 2.5e-8


def test_validated_free_energy_mean_fails_loudly_on_bad_solver_output():
    unconverged = SimpleNamespace(
        converged=False,
        iteration_count=200_000,
        last_blahut_residual=1.0,
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        validated_free_energy_mean(
            unconverged, [1.0], [0], context="unconverged"
        )

    bad_residual = SimpleNamespace(
        converged=True,
        iteration_count=10,
        last_blahut_residual=float("nan"),
    )
    with pytest.raises(RuntimeError, match="non-finite residual"):
        validated_free_energy_mean(bad_residual, [1.0], [0], context="nan")

    finite = SimpleNamespace(
        converged=True,
        iteration_count=10,
        last_blahut_residual=1e-8,
    )
    with pytest.raises(RuntimeError, match="non-finite values"):
        validated_free_energy_mean(finite, [float("inf")], [0], context="inf")

    # A gridCore API rename must fail, not silently assume convergence.
    with pytest.raises(AttributeError):
        validated_free_energy_mean(
            SimpleNamespace(iteration_count=1, last_blahut_residual=0.0),
            [1.0],
            [0],
            context="missing flag",
        )


def _summary(*, score=8.0, fraction_marker=True, **config_changes):
    config = {
        "env_id": "four_rooms",
        "shape": [7, 7],
        "beta": 1.0,
        "determinism": 0.97,
        "state_dist": "uniform",
        "fitness_objective": "free_energy",
        "goal_mode": "all",
    }
    if fraction_marker:
        config["goal_subsample_fraction"] = 1.0
    config.update(config_changes)
    return {
        "status": "completed",
        "fitness_objective": "free_energy",
        "config": config,
        "goals": [0, 1, 2],
        "best_expected_free": score,
    }


def test_floor_qualifier_accepts_full_goal_runs_including_pre_feature_records():
    assert qualifying_floor_score(
        _summary(), env_id="four_rooms", expected_goals=[0, 1, 2]
    ) == 8.0
    assert qualifying_floor_score(
        _summary(fraction_marker=False),
        env_id="four_rooms",
        expected_goals=[0, 1, 2],
    ) == 8.0


@pytest.mark.parametrize(
    "changes",
    [
        {"beta": 0.3},
        {"goal_subsample_fraction": 0.25},
        {"goal_mode": "subset"},
        {"determinism": 0.9},
        {"state_dist": "live"},
    ],
)
def test_floor_qualifier_rejects_mismatched_settings(changes):
    assert qualifying_floor_score(
        _summary(**changes), env_id="four_rooms", expected_goals=[0, 1, 2]
    ) is None


def test_floor_qualifier_rejects_incomplete_or_partial_goal_records():
    running = _summary()
    running["status"] = "running"
    assert qualifying_floor_score(
        running, env_id="four_rooms", expected_goals=[0, 1, 2]
    ) is None

    partial = _summary()
    partial["goals"] = [0, 1]
    assert qualifying_floor_score(
        partial, env_id="four_rooms", expected_goals=[0, 1, 2]
    ) is None


def test_select_floor_candidate_returns_strict_minimum_and_count():
    summaries = [
        ("full-a.json", _summary(score=8.2)),
        ("subsampled.json", _summary(score=7.0, goal_subsample_fraction=0.25)),
        ("full-b.json", _summary(score=7.8)),
    ]
    best, count = select_floor_candidate(
        summaries, env_id="four_rooms", expected_goals=[0, 1, 2]
    )
    assert count == 2
    assert best.score == 7.8
    assert best.source == "full-b.json"
