"""
Ported from gridFour src/analysis/functional_graph (2026-07-11); gridFour copy is frozen.

Functional-graph decomposition of finite deterministic dynamical systems.

For a finite state set S and a deterministic self-map f : S -> S the
functional graph has an edge s -> f(s) at every state.  Each weakly
connected component consists of a single cycle plus a forest of "tail"
trees draining into it.  This module exposes:

- ``deterministic_successor(env, action)`` — argmax-of-transition
  successor map for a single action under a (possibly twisted) GridRoom.
- ``decompose(succ)`` — full structural decomposition returning a
  :class:`FunctionalGraph` with cycles, basins, per-state tail length
  (lambda) and rho-value, terminal nodes, in-degree, and diameter.
- ``per_label_stats(fg)`` — eight summary scalars for one functional
  graph.  Same vocabulary as ``random_twist_baseline`` returns per
  (twist, label) so the per-σ fingerprint and the random baseline
  share a numeric language.

Vocabulary follows Flajolet & Odlyzko (1990), *Random Mapping Statistics*.

Usage
-----
    from gridbench.functional_graph.decomposition import (
        decompose, deterministic_successor,
    )

    succ = deterministic_successor(env, action=0)
    fg = decompose(succ)
    fg.n_basins, fg.diameter, fg.cycle_lengths
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Successor extraction
# ---------------------------------------------------------------------------

def deterministic_successor(env, action: int) -> np.ndarray:
    """Most-probable successor for each state under a fixed action.

    Returns ``argmax_{s'} T(s, a, s')`` per state.  For deterministic
    environments this is exact; under stochastic transitions
    (``determinism < 1``) the noise mass (e.g. 3% spread across the
    other three actions' resolved successors at ``determinism=0.97``)
    is silently discarded — what remains is the *deterministic
    projection* of the policy onto a function ``f : S → S``.

    This projection is what makes the Flajolet & Odlyzko (1990)
    functional-graph machinery (cycles, basins, lambda, rho, diameter)
    applicable at all: those statistics are defined for a deterministic
    self-map, not for a stochastic kernel.  Treating the fingerprint as
    a structural summary of the agent's *intended* dynamics — what σ
    is trying to do, before noise — is the deliberate choice; an
    analogous noise-aware variant would be a Markov chain analysis
    (stationary distributions, mixing times, hitting-time regions)
    rather than a functional graph, with different vocabulary and a
    different visualisation surface.  We do not anticipate needing one.

    For ``determinism > 0.25`` argmax always picks the intended
    successor (the intended action holds prob = ``det`` and the others
    split ``1 - det``; the intended successor dominates as long as
    ``det > (1 - det) / 3``).  So the fingerprint is *invariant* in
    ``determinism`` over the working range — the determinism axis
    matters for ``mean_free`` and ``chi_twist`` (continuous functionals
    of T), not for ``fp_*``.

    Args:
        env: GridRoom (or compatible) exposing ``T``, ``nS``, ``nA``.
            ``T`` is expected as a flat array of shape ``(nS * nA, nS)``
            with rows indexed by ``s * nA + a``.
        action: action index in ``[0, nA)``.

    Returns:
        Integer array of shape ``(nS,)`` — successor state for each state.
    """
    T = np.asarray(env.T)
    nS, nA = int(env.nS), int(env.nA)
    return T.reshape(nS, nA, nS)[:, action, :].argmax(axis=1).astype(int)


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FunctionalGraph:
    """Decomposition of the functional graph of a successor map.

    Field names follow Flajolet & Odlyzko (1990).  A *basin* (weakly
    connected component) is the set of states whose orbit eventually
    reaches a particular cycle; the cycle is the basin's recurrent core.

    Attributes:
        succ: ``(n,)`` int — the successor map.
        cycles: one list of states per basin, in traversal order.
            ``cycles[b][0]`` is the cycle's entry point as first seen
            during decomposition.
        basin_id: ``(n,)`` int — each state's basin index.
        basin_sizes: total number of states in each basin (cycle included).
        cycle_lengths: cycle length (mu) per basin.
        tail_length: ``(n,)`` int — lambda(s), steps from s to its cycle.
            Cyclic states have lambda = 0.
        rho: ``(n,)`` int — lambda(s) + mu_basin(s), the rho-value of s.
        terminal_nodes: ``(k,)`` int — states with no preimage.
        in_degree: ``(n,)`` int — preimage counts.

    Notes:
        ``decompose()`` is the canonical constructor.  Arrays are
        immutable; treat them as read-only.
    """

    succ: np.ndarray
    cycles: list[list[int]]
    basin_id: np.ndarray
    basin_sizes: list[int]
    cycle_lengths: list[int]
    tail_length: np.ndarray
    rho: np.ndarray
    terminal_nodes: np.ndarray
    in_degree: np.ndarray

    @property
    def n_basins(self) -> int:
        """Number of distinct cycles (= number of weakly-connected components)."""
        return len(self.cycles)

    @property
    def diameter(self) -> int:
        """Largest rho-value across states.

        F&O (1990) Theorem 7: for a uniform random mapping on n states the
        expected diameter is asymptotically ``c sqrt(n)`` with
        ``c approx 1.7374``.
        """
        return int(self.rho.max()) if self.rho.size else 0

    @property
    def cyclic_mask(self) -> np.ndarray:
        """Boolean mask: ``True`` where the state lies on a cycle."""
        return self.tail_length == 0

    @property
    def n_cyclic(self) -> int:
        """Number of cyclic states.  Equals ``sum(cycle_lengths)``."""
        return int(self.cyclic_mask.sum())

    @property
    def n_terminal(self) -> int:
        """Number of terminal (preimage-free) states."""
        return int(self.terminal_nodes.size)


def decompose(
    succ: np.ndarray,
    walls: Optional[Iterable[int]] = None,
) -> FunctionalGraph:
    """Decompose a successor map into its functional-graph structure.

    Single forward walk per unvisited state.  Each walk terminates when
    it hits either a state from its own path (new cycle) or a state from
    a previously decomposed basin (joining existing structure).  Runs in
    O(n) total: every state is touched once.

    Args:
        succ: ``(n,)`` int array.  ``succ[s]`` is the successor of state s;
            values must lie in ``[0, n)``.
        walls: optional iterable of state indices to treat as inert.
            Wall states never start a walk, never appear in any cycle,
            never join any basin, and are not counted in ``in_degree``
            / ``terminal_nodes``.  Their ``basin_id``, ``tail_length``,
            ``rho`` carry the sentinel value ``-1`` (or ``0`` for rho).
            Any non-wall state whose ``succ`` value points into a wall
            is defensively redirected to a self-loop on itself — well-
            formed envs should never produce such an edge, but the
            argmax-of-zeros behaviour of ``deterministic_successor`` on
            unreachable wall rows can.

    Returns:
        A :class:`FunctionalGraph` capturing cycles, basins, tail lengths,
        terminal nodes, in-degree, and rho-values over the *reachable*
        (non-wall) state space.
    """
    succ = np.asarray(succ, dtype=int).copy()
    n = succ.size

    # Wall mask: True where the state should be inert.
    wall_mask = np.zeros(n, dtype=bool)
    if walls is not None:
        wall_indices = np.asarray(list(walls), dtype=int)
        if wall_indices.size > 0:
            wall_mask[wall_indices] = True

    # Defensive: redirect any non-wall → wall edge to self.  This keeps
    # the walk strictly inside the non-wall subgraph regardless of what
    # the underlying env stored for wall transitions (e.g. argmax of an
    # all-zero T row returns 0, which is rarely a wall, but the same
    # row would silently fold any non-wall pointing into a wall into
    # wall territory).
    if wall_mask.any():
        target_in_wall = wall_mask[succ]
        bad = (~wall_mask) & target_in_wall
        if bad.any():
            succ = np.where(bad, np.arange(n), succ)

    visited = np.full(n, -1, dtype=int)       # walk-id that first visited s, or -1
    basin_id = np.full(n, -1, dtype=int)
    tail_length = np.full(n, -1, dtype=int)
    cycles: list[list[int]] = []
    cycle_lengths: list[int] = []
    current_basin = 0

    # Pre-mark walls as "visited but inert" so a walk can never enter
    # one and so the outer loop skips them as starts.  Use a sentinel
    # walk-id (n) that no real walk start can match.
    if wall_mask.any():
        visited[wall_mask] = n

    for start in range(n):
        if visited[start] >= 0:
            continue

        path: list[int] = []
        s = start
        while visited[s] < 0:
            visited[s] = start
            path.append(s)
            s = int(succ[s])

        if visited[s] == start:
            # Closed a new cycle inside this walk.
            cycle_start_idx = path.index(s)
            cycle = path[cycle_start_idx:]
            tail = path[:cycle_start_idx]

            bid = current_basin
            current_basin += 1
            cycles.append(cycle)
            cycle_lengths.append(len(cycle))

            for c in cycle:
                basin_id[c] = bid
                tail_length[c] = 0
            # Tail state immediately before the cycle has lambda = 1, etc.
            for offset, c in enumerate(reversed(tail), start=1):
                basin_id[c] = bid
                tail_length[c] = offset
        else:
            # Joined a previously decomposed basin via the tail of s.
            # Defensive: with the wall-redirect step above the walk can
            # never end on a wall, so basin_id[s] is always ≥ 0 here.
            bid = int(basin_id[s])
            base_lambda = int(tail_length[s])
            for offset, c in enumerate(reversed(path), start=1):
                basin_id[c] = bid
                tail_length[c] = base_lambda + offset

    # basin_sizes counts states per basin.  Walls have basin_id = -1 so
    # they're never matched by the ``b in range(current_basin)`` query.
    basin_sizes = [int(np.sum(basin_id == b)) for b in range(current_basin)]
    cycle_mu = np.asarray(cycle_lengths, dtype=int)
    # rho = lambda + mu_basin, but only meaningful where basin_id ≥ 0;
    # walls keep rho = 0 so ``diameter = max(rho)`` is unaffected.
    if n and current_basin:
        non_wall = basin_id >= 0
        rho = np.zeros(n, dtype=int)
        rho[non_wall] = tail_length[non_wall] + cycle_mu[basin_id[non_wall]]
    else:
        rho = np.zeros(n, dtype=int)

    # in_degree counts edges from non-wall sources only.  Wall edges
    # were redirected to self above where they pointed into walls, but
    # walls' own succ values still exist and would otherwise pollute
    # the bincount.
    non_wall_mask = ~wall_mask
    if non_wall_mask.any():
        in_degree = np.bincount(succ[non_wall_mask], minlength=n).astype(int)
    else:
        in_degree = np.zeros(n, dtype=int)
    # Terminal nodes: non-wall states with no preimage from non-walls.
    # Walls always have in_degree = 0 under this counting but are
    # excluded from terminal_nodes because they're not part of the
    # reachable functional graph.
    terminal_nodes = np.flatnonzero((in_degree == 0) & non_wall_mask).astype(int)

    return FunctionalGraph(
        succ=succ,
        cycles=cycles,
        basin_id=basin_id,
        basin_sizes=basin_sizes,
        cycle_lengths=cycle_lengths,
        tail_length=tail_length,
        rho=rho,
        terminal_nodes=terminal_nodes,
        in_degree=in_degree,
    )


# ---------------------------------------------------------------------------
# Per-label summary scalars
# ---------------------------------------------------------------------------

# Field names mirror the keys returned by ``random_twist_baseline``.
PER_LABEL_FIELDS: tuple[str, ...] = (
    "n_basins",
    "n_cyclic",
    "n_terminal",
    "mean_cycle_length",
    "cycle_basin_ratio",
    "mean_tail_length",
    "mean_rho",
    "diameter",
)


def per_label_stats(fg: FunctionalGraph) -> dict[str, float]:
    """Eight summary scalars for one functional graph.

    Returned keys match :data:`PER_LABEL_FIELDS` and the keys returned
    by ``random_twist_baseline``, so the per-σ fingerprint and the
    random baseline cloud can be plotted in the same space.

    Args:
        fg: a :class:`FunctionalGraph`.

    Returns:
        Dict with float values for n_basins, n_cyclic, n_terminal,
        mean_cycle_length, cycle_basin_ratio, mean_tail_length,
        mean_rho, diameter.  When fg was produced with ``walls``, the
        means are taken over non-wall states only (wall states carry
        sentinel ``-1`` in ``tail_length`` and ``0`` in ``rho`` that
        would otherwise drag the means).
    """
    cycle_lengths = np.asarray(fg.cycle_lengths, dtype=float)
    basin_sizes = np.asarray(fg.basin_sizes, dtype=float)
    non_wall = fg.basin_id >= 0
    if non_wall.any():
        tail_mean = float(fg.tail_length[non_wall].mean())
        rho_mean = float(fg.rho[non_wall].mean())
    else:
        tail_mean = 0.0
        rho_mean = 0.0
    return {
        "n_basins": float(fg.n_basins),
        "n_cyclic": float(fg.n_cyclic),
        "n_terminal": float(fg.n_terminal),
        "mean_cycle_length": float(cycle_lengths.mean()) if cycle_lengths.size else 0.0,
        "cycle_basin_ratio": (
            float((cycle_lengths / basin_sizes).mean())
            if cycle_lengths.size else 0.0
        ),
        "mean_tail_length": tail_mean,
        "mean_rho": rho_mean,
        "diameter": float(fg.diameter),
    }
