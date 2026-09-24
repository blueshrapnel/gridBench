"""Strict selection of comparable single-world cross-world floor candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class FloorCandidate:
    env_id: str
    score: float
    source: str


def qualifying_floor_score(
    summary: Mapping[str, object],
    *,
    env_id: str,
    expected_goals: Sequence[int],
    beta: float = 1.0,
    determinism: float = 0.97,
    shape: tuple[int, int] = (7, 7),
    state_dist: str = "uniform",
) -> float | None:
    """Return the score only for completed, full-goal runs at frozen settings."""
    config = summary.get("config")
    if not isinstance(config, Mapping):
        return None
    if summary.get("status") != "completed":
        return None
    objective = summary.get("fitness_objective", config.get("fitness_objective"))
    if objective != "free_energy":
        return None
    if str(config.get("env_id")) != str(env_id):
        return None
    try:
        recorded_shape = tuple(int(v) for v in config.get("shape", ()))
        recorded_beta = float(config.get("beta", float("nan")))
        recorded_determinism = float(config.get("determinism", float("nan")))
        fraction = float(config.get("goal_subsample_fraction", 1.0))
    except (TypeError, ValueError):
        return None
    if recorded_shape != tuple(shape):
        return None
    if recorded_beta != float(beta):
        return None
    if recorded_determinism != float(determinism):
        return None
    if str(config.get("state_dist")) != state_dist:
        return None
    if str(config.get("goal_mode")) != "all":
        return None

    # Runs predating the subsampling feature have no field; goal_mode=all plus
    # an exact full goal list establishes their equivalent K=1 behaviour.
    if not math.isfinite(fraction) or fraction != 1.0:
        return None
    goals = summary.get("goals")
    if not isinstance(goals, Sequence) or isinstance(goals, (str, bytes)):
        return None
    if sorted(int(g) for g in goals) != sorted(int(g) for g in expected_goals):
        return None

    raw_score = summary.get("best_expected_free")
    if raw_score is None:
        return None
    score = float(raw_score)
    return score if math.isfinite(score) else None


def select_floor_candidate(
    summaries: Iterable[tuple[str, Mapping[str, object]]],
    *,
    env_id: str,
    expected_goals: Sequence[int],
) -> tuple[FloorCandidate, int]:
    """Return the best strict candidate and the qualifying cohort size."""
    candidates = []
    for source, summary in summaries:
        score = qualifying_floor_score(
            summary, env_id=env_id, expected_goals=expected_goals
        )
        if score is not None:
            candidates.append(FloorCandidate(env_id, score, str(source)))
    if not candidates:
        raise RuntimeError(f"no qualifying full-goal floor candidates for {env_id}")
    return min(candidates, key=lambda item: item.score), len(candidates)
