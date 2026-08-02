"""
Null-model baselines for per-label functional-graph statistics.

Provides:

- ``permutation_null(n)`` — closed-form expectations under a uniform
  random permutation of ``{0, ..., n-1}``.  Appropriate as a coarse
  reference for the Cartesian (untwisted) torus, where each bare action
  is a bijection of S.
- ``mapping_null(n)`` — Flajolet-Odlyzko asymptotics under a uniform
  random mapping ``{0,...,n-1} -> {0,...,n-1}``.  Appropriate as a
  coarse reference for walled environments where per-label dynamics is
  many-to-one.
- ``classify_null_model(env)`` — heuristic that returns
  ``"permutation"`` if every bare action of ``env`` is a bijection of
  states, else ``"mapping"``.
- ``sample_random_twist(nS, nA, rng)`` — draw a uniform valid twist.
- ``random_twist_baseline(env, n_samples)`` — empirical baseline cloud,
  one record per (twist, label) pair.  This is the *trusted* reference
  for the journal paper; the closed-form nulls above are coarse and
  only loosely match because a random twist on a structured environment
  produces a *constrained* random mapping (each state picks among its
  geometric neighbours, not uniformly across states).

Reference: Flajolet, P. & Odlyzko, A. M. (1990).  Random mapping
statistics. *Lecture Notes in Computer Science*, 434, 329-354.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gridbench.functional_graph.decomposition import (
    PER_LABEL_FIELDS,
    decompose,
    deterministic_successor,
    per_label_stats,
)


# ---------------------------------------------------------------------------
# Constants from Flajolet & Odlyzko (1990) and adjacent literature
# ---------------------------------------------------------------------------

# Expected diameter of a uniform random mapping is ~ c sqrt(n).
_DIAMETER_CONST = 1.7374

# Expected largest component / n under uniform random mapping.
_GIANT_COMPONENT = 0.7578

# Golomb-Dickman constant: E[longest cycle in a random permutation] / n.
_GOLOMB_DICKMAN = 0.62433


# ---------------------------------------------------------------------------
# Closed-form null expectations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NullExpectations:
    """Expectations of functional-graph summary statistics under a named
    null model on ``n`` states.

    Fields are scalar expectations; per-state values are averaged over
    states (not basins).  Use as coarse reference points only — see the
    module docstring for why the constrained-mapping induced by a random
    twist on a structured environment doesn't exactly match either null.

    Attributes:
        name: ``"permutation"`` or ``"mapping"``.
        n: number of states.
        n_basins: expected number of cycles / weakly-connected components.
        n_cyclic: expected number of states on a cycle.
        n_terminal: expected number of preimage-free states.
        mean_cycle_length: average cycle length over basins.
        mean_tail_length: average lambda(s) over states.
        mean_rho: average rho(s) = lambda(s) + mu_basin(s) over states.
        diameter: expected max rho across states.
        largest_component: expected largest basin size.
    """

    name: str
    n: int
    n_basins: float
    n_cyclic: float
    n_terminal: float
    mean_cycle_length: float
    mean_tail_length: float
    mean_rho: float
    diameter: float
    largest_component: float


def permutation_null(n: int) -> NullExpectations:
    """Expected statistics under a uniform random permutation of ``n`` states.

    Exact closed-form where available, asymptotic constants otherwise.

    Notes:
        E[# cycles] = H_n = sum_{k=1}^n 1/k.
        E[longest cycle] / n -> Golomb-Dickman constant ≈ 0.62433.
        Every state is cyclic, no terminal nodes, lambda = 0 everywhere.
        E[cycle length of the basin of a uniformly chosen state] = (n+1)/2.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    H_n = float(np.sum(1.0 / np.arange(1, n + 1)))
    return NullExpectations(
        name="permutation",
        n=n,
        n_basins=H_n,
        n_cyclic=float(n),
        n_terminal=0.0,
        mean_cycle_length=float(n) / H_n,
        mean_tail_length=0.0,
        mean_rho=float(n + 1) / 2.0,
        diameter=float(n) * _GOLOMB_DICKMAN,
        largest_component=float(n) * _GOLOMB_DICKMAN,
    )


def mapping_null(n: int) -> NullExpectations:
    """Expected statistics under a uniform random mapping on ``n`` states.

    Flajolet & Odlyzko (1990) asymptotics.  Accuracy improves with n;
    treat as order-of-magnitude reference points at moderate n.

    Notes:
        E[# components] ~ (1/2) log n.
        E[# terminal nodes] ~ n / e ≈ 0.368 n.
        E[# cyclic nodes] ~ sqrt(pi n / 2).
        E[mu] = E[lambda] ~ sqrt(pi n / 8).
        E[rho] ~ sqrt(pi n / 2).
        E[diameter] ~ c sqrt(n), c ≈ 1.7374.
        E[largest component] / n -> 0.7578.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    sqrt_pi_n_over_8 = float(np.sqrt(np.pi * n / 8.0))
    return NullExpectations(
        name="mapping",
        n=n,
        n_basins=0.5 * float(np.log(n)),
        n_cyclic=2.0 * sqrt_pi_n_over_8,          # = sqrt(pi n / 2)
        n_terminal=float(n) / float(np.e),
        mean_cycle_length=sqrt_pi_n_over_8,
        mean_tail_length=sqrt_pi_n_over_8,
        mean_rho=2.0 * sqrt_pi_n_over_8,
        diameter=_DIAMETER_CONST * float(np.sqrt(n)),
        largest_component=float(n) * _GIANT_COMPONENT,
    )


# ---------------------------------------------------------------------------
# Environment classification
# ---------------------------------------------------------------------------

def classify_null_model(env) -> str:
    """Return ``"permutation"`` if every bare action of ``env`` is a
    bijection of states, else ``"mapping"``.

    Heuristic for picking which closed-form null to compare against.
    Note: a random *twist* on a permutation-class env still induces a
    constrained random mapping (not a permutation), since at each state
    the twist picks one of |A| distinct geometric successors uniformly.
    The classification reflects the *bare* env, not the twisted one.

    Args:
        env: GridRoom (or compatible) exposing ``T``, ``nS``, ``nA``.

    Returns:
        ``"permutation"`` or ``"mapping"``.
    """
    nS = int(env.nS)
    for a in range(int(env.nA)):
        succ = deterministic_successor(env, a)
        if int(np.unique(succ).size) != nS:
            return "mapping"
    return "permutation"


# ---------------------------------------------------------------------------
# Sampling and empirical baseline
# ---------------------------------------------------------------------------

def sample_random_twist(
    nS: int,
    nA: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a uniform valid twist of shape ``(nS, nA)``.

    A valid twist ``sigma(s, .)`` is a permutation of ``{0, ..., nA-1}``
    at each state, drawn independently across states.

    Args:
        nS: number of states.
        nA: number of actions / labels.
        rng: numpy ``Generator`` for reproducibility.

    Returns:
        Integer array of shape ``(nS, nA)`` with each row a permutation.
    """
    sigma = np.empty((nS, nA), dtype=int)
    for s in range(nS):
        sigma[s] = rng.permutation(nA)
    return sigma


def _precompute_base_successors(env) -> np.ndarray:
    """Stack of per-action deterministic successor maps for an untwisted env.

    Returns array of shape ``(nA, nS)`` where row ``a`` is the successor
    map under bare action ``a``.  Applying a twist sigma then amounts to
    indexing through the cached inverse:
    ``f_label(s) = base_succ[sigma_inv[s, label], s]`` where
    ``sigma_inv[s, label] -> a_phys`` (see fingerprint_for_sigma
    docstring for the convention reference).
    """
    return np.stack(
        [deterministic_successor(env, a) for a in range(int(env.nA))],
        axis=0,
    )


def random_twist_baseline(
    env,
    n_samples: int = 1000,
    *,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Empirical baseline cloud over random twists.

    Samples ``n_samples`` uniform random twists on ``env`` and runs the
    per-label functional-graph decomposition for each ``(twist, label)``
    pair.  Returns a dict of ``(n_samples, nA)`` arrays for downstream
    aggregation.

    The env is assumed to be the *untwisted* environment with no goal
    absorbing — use ``build_twisted_env_no_goal(cfg, identity_sigma)`` or
    the bare ``GridRoom(...)`` constructor with ``goals=[]``.  Twists are
    applied by indexing into a precomputed successor table, not by
    rebuilding the env per sample.

    Args:
        env: untwisted, no-goal GridRoom (or compatible).
        n_samples: number of twists to sample.
        rng: numpy ``Generator``; defaults to a fresh ``default_rng()``.

    Returns:
        Dict with one entry per statistic; values are ``(n_samples, nA)``
        float arrays indexed as ``[sample, label]``:

        - ``n_basins``: number of cycles.
        - ``n_cyclic``: number of states on a cycle.
        - ``n_terminal``: number of preimage-free states.
        - ``mean_cycle_length``: average cycle length over basins.
        - ``cycle_basin_ratio``: average ``cycle_length / basin_size``
          over basins.
        - ``mean_tail_length``: average lambda(s) over states.
        - ``mean_rho``: average rho(s) over states.
        - ``diameter``: max rho across states.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    rng = np.random.default_rng() if rng is None else rng

    nS = int(env.nS)
    nA = int(env.nA)
    base_succ = _precompute_base_successors(env)         # (nA, nS)
    state_idx = np.arange(nS)

    # Walls are inert: decompose excludes them so the null cloud
    # describes only the agent's reachable graph (same contract as
    # fingerprint_for_sigma — see decomposition.decompose docstring).
    walls = getattr(env, "walls_flat", None)

    out = {f: np.zeros((n_samples, nA), dtype=float) for f in PER_LABEL_FIELDS}

    for i in range(n_samples):
        sigma = sample_random_twist(nS, nA, rng)
        # Per-label graph uses sigma_inv (label -> a_phys); convention
        # reference in fingerprint_for_sigma docstring.
        sigma_inv = np.argsort(sigma, axis=1)
        for ell in range(nA):
            succ = base_succ[sigma_inv[:, ell], state_idx]
            stats = per_label_stats(decompose(succ, walls=walls))
            for field, value in stats.items():
                out[field][i, ell] = value

    return out
