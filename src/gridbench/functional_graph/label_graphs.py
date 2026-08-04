"""Functional graphs induced by repeatedly issuing each action label."""

from dataclasses import dataclass

import numpy as np

from gridbench.functional_graph.decomposition import (
    decompose, deterministic_successor, validate_sigma,
)


@dataclass(frozen=True)
class LabelGraph:
    """One label's deterministic successor map and decomposition."""

    label: int
    succ: np.ndarray
    fg: object
    cycle_set: frozenset[int]


def walls_and_nonwalls(env) -> tuple[set[int], list[int]]:
    """Return flat wall indices and the complementary walkable-state list."""
    raw_walls = getattr(env, "walls_flat", None)
    walls = (
        set()
        if raw_walls is None
        else {int(state) for state in np.asarray(raw_walls).ravel()}
    )
    return walls, [state for state in range(int(env.nS)) if state not in walls]


def label_graphs(env, sigma: np.ndarray) -> list[LabelGraph]:
    """Decompose the repeated-label graph for every label in ``sigma``.

    ``sigma[state, physical_action]`` gives the issued label, so its row-wise
    inverse maps a repeated label back to the physical action executed at that
    state.
    """
    n_states, n_actions = int(env.nS), int(env.nA)
    sigma = validate_sigma(sigma, n_actions)
    if sigma.shape != (n_states, n_actions):
        raise ValueError(
            f"expected sigma shape {(n_states, n_actions)}, got {sigma.shape}"
        )
    raw_walls = getattr(env, "walls_flat", None)
    walls = [] if raw_walls is None else np.asarray(raw_walls).ravel().tolist()
    states = np.arange(n_states)
    base = np.stack(
        [deterministic_successor(env, action) for action in range(n_actions)]
    )
    sigma_inverse = np.argsort(sigma, axis=1)

    graphs = []
    for label in range(n_actions):
        successor = base[sigma_inverse[:, label], states]
        graph = decompose(successor, walls=walls)
        cycle_set = frozenset(state for cycle in graph.cycles for state in cycle)
        graphs.append(LabelGraph(label, successor, graph, cycle_set))
    return graphs
