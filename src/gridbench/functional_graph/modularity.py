"""Louvain modularity of cross-label basin agreement.

For a twist ``sigma``, repeatedly issuing each label induces one functional
graph and hence one basin partition of the walkable states.  The basin
co-membership graph places a weighted, undirected edge between two states for
each label under which they share a basin.  Louvain modularity on that graph is
therefore a scalar measure of *agreement between the labels' basin
partitions*.

The graph contains no spatial-adjacency term.  A high score does not by itself
show that the detected communities are compact rooms or that they predict
policy boundaries; those are separate hypotheses requiring external tests.

This module is the corrected, gridCore-compatible part of the older gridFour
modularity work.  It deliberately excludes the historical FEP+M optimisation
and K-annealing analyses.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np

from gridbench.functional_graph.label_graphs import (
    label_graphs,
    walls_and_nonwalls,
)


@dataclass(frozen=True)
class ModularityDiagnostics:
    """Seed-restart diagnostics for one basin co-membership graph.

    ``mean_q`` is the scalar used by the expressive-range analysis.  The Q
    spread records heuristic variability; adjusted Rand indices record whether
    the actual node partitions agree across restarts.
    """

    mean_q: float
    std_q: float
    min_q: float
    max_q: float
    median_pairwise_ari: float
    min_pairwise_ari: float
    n_communities: tuple[int, ...]


def basin_comembership_graph(env, sigma: np.ndarray) -> nx.Graph:
    """Build the weighted graph of cross-label basin agreement.

    Nodes are walkable state indices.  Edge weight ``(s, t)`` is the number of
    labels whose repeated-label functional graphs place ``s`` and ``t`` in the
    same basin.  With four labels, weights therefore lie in ``[1, 4]``; pairs
    that never share a basin have no edge.
    """

    graphs = label_graphs(env, sigma)
    _, nonwall = walls_and_nonwalls(env)

    graph = nx.Graph()
    graph.add_nodes_from(nonwall)

    weights: dict[tuple[int, int], int] = {}
    for labelled in graphs:
        basin_id = labelled.fg.basin_id
        for basin in range(labelled.fg.n_basins):
            members = [state for state in nonwall if basin_id[state] == basin]
            for source, target in combinations(members, 2):
                edge = (source, target) if source < target else (target, source)
                weights[edge] = weights.get(edge, 0) + 1

    # NetworkX's Louvain traversal is sensitive to graph insertion order even
    # with a fixed random seed.  Insert lexicographically so corrected
    # gridFour references and independent rebuilds produce identical results.
    graph.add_weighted_edges_from(
        (source, target, weights[(source, target)])
        for source, target in sorted(weights)
    )
    return graph


def _adjusted_rand_index(first: list[int], second: list[int]) -> float:
    """Adjusted Rand index without adding a scikit-learn dependency."""

    a = np.asarray(first, dtype=int)
    b = np.asarray(second, dtype=int)
    if a.shape != b.shape:
        raise ValueError(f"partition shapes differ: {a.shape} != {b.shape}")
    if a.size < 2:
        return 1.0

    _, a_inverse = np.unique(a, return_inverse=True)
    _, b_inverse = np.unique(b, return_inverse=True)
    contingency = np.zeros(
        (int(a_inverse.max()) + 1, int(b_inverse.max()) + 1), dtype=int
    )
    np.add.at(contingency, (a_inverse, b_inverse), 1)

    def choose_two(values):
        values = np.asarray(values, dtype=float)
        return values * (values - 1.0) / 2.0

    index = float(choose_two(contingency).sum())
    row_pairs = float(choose_two(contingency.sum(axis=1)).sum())
    column_pairs = float(choose_two(contingency.sum(axis=0)).sum())
    total_pairs = float(choose_two(np.array([a.size]))[0])
    expected = row_pairs * column_pairs / total_pairs if total_pairs else 0.0
    maximum = (row_pairs + column_pairs) / 2.0
    if maximum == expected:
        return 1.0
    return (index - expected) / (maximum - expected)


def modularity_diagnostics_for_graph(
    graph: nx.Graph,
    *,
    n_seeds: int = 10,
    resolution: float = 1.0,
) -> ModularityDiagnostics:
    """Run Louvain repeatedly and report scalar and partition stability."""

    if n_seeds < 1:
        raise ValueError(f"n_seeds must be positive, got {n_seeds}")
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")

    nodes = tuple(sorted(graph.nodes))
    if graph.number_of_edges() == 0:
        n_communities = len(nodes)
        return ModularityDiagnostics(
            mean_q=0.0,
            std_q=0.0,
            min_q=0.0,
            max_q=0.0,
            median_pairwise_ari=1.0,
            min_pairwise_ari=1.0,
            n_communities=(n_communities,) * n_seeds,
        )

    qs: list[float] = []
    memberships: list[list[int]] = []
    community_counts: list[int] = []
    for seed in range(n_seeds):
        communities = nx.community.louvain_communities(
            graph,
            weight="weight",
            resolution=resolution,
            seed=seed,
        )
        qs.append(
            float(
                nx.community.modularity(
                    graph,
                    communities,
                    weight="weight",
                    resolution=resolution,
                )
            )
        )
        membership = {
            node: community_index
            for community_index, community in enumerate(communities)
            for node in community
        }
        memberships.append([membership[node] for node in nodes])
        community_counts.append(len(communities))

    pairwise_ari = [
        _adjusted_rand_index(first, second)
        for first, second in combinations(memberships, 2)
    ]
    if not pairwise_ari:
        pairwise_ari = [1.0]

    q_values = np.asarray(qs, dtype=float)
    return ModularityDiagnostics(
        mean_q=float(q_values.mean()),
        std_q=float(q_values.std()),
        min_q=float(q_values.min()),
        max_q=float(q_values.max()),
        median_pairwise_ari=float(np.median(pairwise_ari)),
        min_pairwise_ari=float(np.min(pairwise_ari)),
        n_communities=tuple(community_counts),
    )


def modularity_diagnostics_for_sigma(
    env,
    sigma: np.ndarray,
    *,
    n_seeds: int = 10,
    resolution: float = 1.0,
) -> ModularityDiagnostics:
    """Build ``sigma``'s co-membership graph and run Louvain diagnostics."""

    return modularity_diagnostics_for_graph(
        basin_comembership_graph(env, sigma),
        n_seeds=n_seeds,
        resolution=resolution,
    )


def modularity_for_sigma(
    env,
    sigma: np.ndarray,
    *,
    n_seeds: int = 10,
    resolution: float = 1.0,
) -> float:
    """Mean Louvain Q over deterministic seeds ``0 .. n_seeds - 1``."""

    return modularity_diagnostics_for_sigma(
        env,
        sigma,
        n_seeds=n_seeds,
        resolution=resolution,
    ).mean_q
