"""Goal-free analysis environment built by GridCore's canonical dispatcher."""

from typing import Iterable

from gridcore.bridge import build_goal_free_env_by_id


def build_goal_free_probe_env(
    env_id: str,
    shape: tuple[int, int],
    determinism: float,
    walls: Iterable[int] | None = None,
):
    """Build a validated, goal-free environment for structural analysis."""
    return build_goal_free_env_by_id(
        env_id=env_id,
        shape=tuple(shape),
        determinism=float(determinism),
        manhattan=True,
        walls=None if walls is None else list(walls),
    )
