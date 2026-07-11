"""
Ported from gridFour src/analysis/functional_graph (2026-07-11); gridFour copy is frozen.

Goal-free probe-env constructor for σ-only functional-graph statistics.

The structural summary (Q_modularity, dead_end_fraction, basin_entropy,
mean_pairwise_jaccard, union_coverage) and the eight ``fp_*``
fingerprint fields are pure functions of ``(env, sigma)`` *with no
goal absorption*.  A goalful env has the goal state absorbing in the
transition kernel — that changes the deterministic projection at the
goal state and shifts dead-end fractions, Jaccard, and Q.  Empirical
drift on four-rooms 7×7 identity σ: ``dead_end_fraction`` 0.0625
goalful vs 0.05 goal-free; ``mean_pairwise_jaccard`` 0.155 vs 0.118.

This module is the single source of truth for that env construction.
The GA's ``_build_probe_env`` in ``evolution_multi_goal.ga_optimizer``
builds a *goalful* probe env (with a walkable placeholder goal) for
the per-goal Blahut path; that env is wrong for the structural call
at fingerprint time.  Import :func:`build_goal_free_probe_env` from
here for any goal-free analysis path — fingerprint cache, structural
backfill, GA structural-summary site.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Optional, Tuple

from gridcore.envs import (
    GridRoom,
    compute_corridor_1d_ring_walls,
    compute_corridor_four_rooms_walls,
    compute_four_room_walls,
    compute_pillar_walls,
    compute_pinwheel_walls,
    compute_plus_cross_walls,
    compute_x_wall_walls,
)


# env_id → callable(width, height) -> list[int].  Mirrors the table in
# ``experiments.functional_graph.build_cache._WALL_BUILDERS`` and the
# one in ``gridtwistflask.drilldown._WALL_BUILDERS``.  Keep these three
# in sync if a new env layout is added.
_WALL_BUILDERS: Mapping[str, Callable[[int, int], list]] = {
    "four_rooms":      lambda w, h: compute_four_room_walls(w, h),
    "corr_1d_ring":    lambda w, h: compute_corridor_1d_ring_walls(w, h),
    "corr_four_rooms": lambda w, h: compute_corridor_four_rooms_walls(w, h),
    "pillar_1":        lambda w, h: compute_pillar_walls(w, h, 1),
    "pillar_2":        lambda w, h: compute_pillar_walls(w, h, 2),
    "pillar_3":        lambda w, h: compute_pillar_walls(w, h, 3),
    "plus_cross":      lambda w, h: compute_plus_cross_walls(w, h),
    "x_wall":          lambda w, h: compute_x_wall_walls(w, h),
    "pinwheel":        lambda w, h: compute_pinwheel_walls(w, h),
    # wrap_pillar_3 shares pillar_3's walls; wrap=True below.
    "wrap_pillar_3":   lambda w, h: compute_pillar_walls(w, h, 3),
    # helical: no internal walls; helical wrap is set in
    # build_goal_free_probe_env via the _HELICAL_ENVS hook.
    "helical":         lambda w, h: [],
}

_WRAP_ENVS = frozenset({"wrap_grid", "wrap_pillar_3", "helical"})

# env_id -> default seam_shift (k_v, k_h) for helical envs.  Plain
# ``wrap_grid``/``wrap_pillar_3`` use ``seam_shift=(0, 0)`` implicitly.
_HELICAL_ENVS: Mapping[str, Tuple[int, int]] = {
    "helical": (1, 0),
}


def build_goal_free_probe_env(
    env_id: str,
    shape: Tuple[int, int],
    determinism: float,
    walls: Optional[Iterable[int]] = None,
) -> GridRoom:
    """Untwisted, goal-free env for σ functional-graph statistics.

    Wraparound is derived from ``env_id``.  Walls come from one of two
    sources, in priority order: an explicit ``walls`` argument (used by
    the GA fitness path, which already has a goalful probe env with
    ``walls_flat`` populated and wants the goal-free env to mirror it
    exactly), or the ``env_id`` lookup in ``_WALL_BUILDERS`` (used by
    the standalone fingerprint cache builder, which has no pre-existing
    env to copy walls from).  If the wall builder rejects the shape
    (e.g., a unit test using a 1x3 grid for ``corr_1d_ring``), walls
    fall back to empty — production shapes never trip this path.

    Args:
        env_id: Environment identifier, e.g. ``"four_rooms"``.
        shape: ``(height, width)``.
        determinism: Transition determinism, passed through to
            ``GridRoom``.
        walls: Optional explicit list of wall-state flat indices.

    Returns:
        A ``GridRoom`` instance with ``goals=[]``; the env's transition
        kernel does not absorb any state.
    """
    options: dict = {
        "shape": tuple(shape),
        "goals": [],
        "manhattan": True,
        "determinism": float(determinism),
        "epsilon": 0.0,
        "twist_seed": 0,
    }
    if walls is not None:
        options["walls"] = [int(s) for s in walls]
    else:
        builder = _WALL_BUILDERS.get(env_id)
        if builder is not None:
            try:
                options["walls"] = builder(int(shape[1]), int(shape[0]))
            except Exception:
                # Wall builders enforce geometry constraints that don't
                # apply to tiny test shapes; fall back to no walls.
                pass
    if env_id in _WRAP_ENVS:
        options["wrap"] = True
    if env_id in _HELICAL_ENVS:
        options["seam_shift"] = _HELICAL_ENVS[env_id]
    return GridRoom(options)
