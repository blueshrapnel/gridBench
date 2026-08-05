from copy import deepcopy

import networkx as nx
import numpy as np
import pytest

from gridbench.functional_graph.decomposition import decompose, deterministic_successor
from gridbench.functional_graph.modularity import (
    _adjusted_rand_index,
    basin_comembership_graph,
    modularity_diagnostics_for_graph,
    modularity_diagnostics_for_sigma,
    modularity_for_sigma,
)
from gridbench.functional_graph.probe_env import build_goal_free_probe_env


def _non_involution_sigma(n_states: int) -> np.ndarray:
    sigma = np.tile(np.arange(4, dtype=int), (n_states, 1))
    rows = (
        np.array([1, 2, 0, 3]),
        np.array([2, 0, 1, 3]),
        np.array([0, 2, 3, 1]),
    )
    for state in range(n_states):
        sigma[state] = rows[state % len(rows)]
    return sigma


def _graph_from_twisted_env(env, sigma: np.ndarray) -> nx.Graph:
    """Independent reference using gridCore's actual twist-dynamics path."""

    twisted = deepcopy(env)
    twisted.set_sigma(sigma)
    twisted.twist_dynamics()
    twisted.update_dynamics_for_goals([])

    walls = set(int(state) for state in twisted.walls_flat)
    nonwall = [state for state in range(twisted.nS) if state not in walls]
    basin_ids = []
    for label in range(twisted.nA):
        successor = deterministic_successor(twisted, label)
        basin_ids.append(decompose(successor, walls=walls).basin_id)

    graph = nx.Graph()
    graph.add_nodes_from(nonwall)
    for index, source in enumerate(nonwall):
        for target in nonwall[index + 1 :]:
            weight = sum(ids[source] == ids[target] for ids in basin_ids)
            if weight:
                graph.add_edge(source, target, weight=weight)
    return graph


def test_graph_matches_gridcore_twisted_dynamics_for_non_involution_sigma():
    env = build_goal_free_probe_env("four_rooms", (7, 7), 0.97)
    sigma = _non_involution_sigma(env.nS)

    actual = basin_comembership_graph(env, sigma)
    expected = _graph_from_twisted_env(env, sigma)

    assert nx.utils.graphs_equal(actual, expected)
    assert not set(actual.nodes) & set(env.walls_flat)


def test_identity_four_rooms_matches_corrected_gridfour_reference():
    env = build_goal_free_probe_env("four_rooms", (7, 7), 0.97)
    sigma = np.tile(np.arange(env.nA), (env.nS, 1))

    assert modularity_for_sigma(env, sigma) == pytest.approx(
        0.4580016562289737, abs=1e-12
    )


def test_diagnostics_are_deterministic_and_report_partition_stability():
    env = build_goal_free_probe_env("four_rooms", (7, 7), 0.97)
    sigma = _non_involution_sigma(env.nS)

    first = modularity_diagnostics_for_sigma(env, sigma, n_seeds=5)
    second = modularity_diagnostics_for_sigma(env, sigma, n_seeds=5)

    assert first == second
    assert 0.0 <= first.mean_q <= 1.0
    assert first.min_q <= first.mean_q <= first.max_q
    assert -1.0 <= first.min_pairwise_ari <= 1.0
    assert len(first.n_communities) == 5


def test_edgeless_graph_has_defined_zero_diagnostics():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2, 3])

    result = modularity_diagnostics_for_graph(graph, n_seeds=3)

    assert result.mean_q == 0.0
    assert result.median_pairwise_ari == 1.0
    assert result.n_communities == (3, 3, 3)


def test_adjusted_rand_index_distinguishes_same_and_crossed_partitions():
    assert _adjusted_rand_index([0, 0, 1, 1], [4, 4, 9, 9]) == pytest.approx(1.0)
    assert _adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]) < 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"n_seeds": 0}, "n_seeds"), ({"resolution": 0.0}, "resolution")],
)
def test_invalid_louvain_parameters_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        modularity_diagnostics_for_graph(nx.path_graph(3), **kwargs)
