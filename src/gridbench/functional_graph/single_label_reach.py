"""
Single-label reach distributions for the per-label functional graph.

Reproduces the measurement behind Daniel's old random-twist diffusion
plots: for a twist sigma on an environment, pick one action label and
follow it deterministically from every starting state.  The trajectory
length until the first revisit (a.k.a. the F&O rho-value) is the
single-label reach.

This is the cheap, policy-free measurement that scales to large grids
where the full DI / GA pipeline is intractable.  The Cartesian
(untwisted) baseline saturates this axis -- every starting state has
rho = (n + 1) / 2 in expectation under the permutation null -- while
ever-more-twisted worlds collapse toward rho ~ sqrt(pi n / 2) under
the F&O random-mapping null.  See ``null_models.py`` for both.

This module is a thin wrapper around ``decomposition.decompose``: the
rho-array on the FunctionalGraph already *is* the per-start reach
vector.  Kept as a named entry point so notebook readers don't have to
know about the decomposition module to understand the measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from gridbench.functional_graph.decomposition import (
    FunctionalGraph,
    decompose,
    deterministic_successor,
)


@dataclass(frozen=True)
class ReachSample:
    """Per-start reach lengths for a single label.

    Attributes:
        rho: ``(n,)`` int array -- trajectory length from each starting
            state until the first revisit (F&O rho = lambda + mu_basin).
        label: action-label index that was followed.
        cutoff: applied clip on rho (passed through to the consumer);
            None means the raw rho is returned unclipped.
        fg: the underlying ``FunctionalGraph``, for callers that want to
            cross-reference cycle / basin structure with the reach
            distribution.  Cheap to keep around (~few arrays of size n).
    """

    rho: np.ndarray
    label: int
    cutoff: Optional[int]
    fg: FunctionalGraph


def reach_sample(env, label: int, *, cutoff: Optional[int] = None) -> ReachSample:
    """Return per-start reach lengths under a single committed label.

    For every starting state, follow ``label`` deterministically until
    the first revisited state.  The number of steps to that revisit is
    the F&O rho-value of the starting state in the per-label functional
    graph.  No simulation loop is needed -- ``decompose`` produces rho
    in O(n) for the whole state space.

    Args:
        env: GridRoom (or compatible) exposing ``T``, ``nS``, ``nA``.
            Twist must already be applied (i.e. ``env.twist_dynamics``
            was called or the env was constructed with ``epsilon > 0``).
        label: action-label index in ``[0, nA)`` to follow.
        cutoff: optional upper bound; rho values above this are clipped
            to ``cutoff``.  Use to reproduce histogram-style plots that
            truncated long trajectories (e.g. Daniel's plots cap at 500).
            ``None`` returns the raw (unclipped) rho values.

    Returns:
        A :class:`ReachSample` with the per-start rho array, the label
        followed, the applied cutoff, and the underlying FunctionalGraph.
    """
    succ = deterministic_successor(env, action=int(label))
    fg = decompose(succ)
    rho = np.asarray(fg.rho, dtype=int)
    if cutoff is not None:
        rho = np.minimum(rho, int(cutoff))
    return ReachSample(rho=rho, label=int(label), cutoff=cutoff, fg=fg)


def trajectory(env, label: int, start: int, *, max_steps: int) -> np.ndarray:
    """Trajectory of states visited by committing to ``label`` from ``start``.

    Used for the trajectory-streak panels (Daniel's ``twist_n_*.pdf``
    family).  Stops as soon as a state is revisited *or* ``max_steps``
    is reached, whichever comes first.  The cycle itself is *not*
    unrolled; the returned sequence ends at the first state that would
    close the cycle so it can be drawn as a single non-looping streak.

    Args:
        env: GridRoom with applied twist.
        label: action-label index to follow.
        start: starting state index.
        max_steps: hard upper bound on returned length (defensive only;
            the cycle-closure check normally fires first).

    Returns:
        Int array of state indices in visit order, including ``start``
        at index 0 and ending just before the first revisited state.
    """
    succ = deterministic_successor(env, action=int(label))
    seen = {int(start)}
    path = [int(start)]
    s = int(start)
    for _ in range(int(max_steps)):
        s = int(succ[s])
        if s in seen:
            break
        seen.add(s)
        path.append(s)
    return np.asarray(path, dtype=int)
