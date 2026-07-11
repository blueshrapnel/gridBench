"""
Ported from gridFour src/analysis/functional_graph (2026-07-11); gridFour copy is frozen.

Per-σ fingerprint: nine summary scalars derived from the per-label
functional-graph decomposition, aggregated across action labels.  Eight
are mean/max reductions of the per-label stats; the ninth,
``fp_largest_basin_fraction``, is the paper's coverage scalar (max over
labels of the largest basin as a fraction of non-wall states).

The fingerprint is a pure function of ``(env, sigma)``: same inputs
produce identical numbers, no randomness, no policy, no fitness, no
beta.  This is the function the cache builder calls; the cache I/O
itself lives in :mod:`gridbench.functional_graph.cache`.

Per-label statistics are produced by
:func:`gridbench.functional_graph.decomposition.per_label_stats`.  Mean-
type stats (number of basins, cycle length, etc.) are averaged across
the labels of a single twist; ``diameter`` is taken as the *max*
across labels because the longest reach of any one label is what
controls how far the agent can be from its eventual habit cycle.

Deterministic projection
------------------------
The fingerprint is computed from the **deterministic projection** of
``T`` (one ``argmax`` per state-action row in
``deterministic_successor``).  Noise mass is dropped — see that
function's docstring for the full motivation.  Briefly: Flajolet &
Odlyzko's (1990) *Random Mapping Statistics* defines cycles, basins,
lambda, rho, and diameter on a deterministic self-map ``f : S → S``;
the projection is what makes the F&O machinery applicable and what
gives us a structural summary of σ's *intended* dynamics.  The
fingerprint is therefore invariant in ``determinism`` over the
working range (>0.25); noise affects ``mean_free`` and ``chi_twist``,
not ``fp_*``.  A noise-aware variant is not planned.
"""

from __future__ import annotations

import numpy as np

from gridbench.functional_graph.decomposition import (
    PER_LABEL_FIELDS,
    decompose,
    deterministic_successor,
    per_label_stats,
)


# Field names emitted by :func:`fingerprint_for_sigma`.  Mirror the
# per-label keys with an ``fp_`` prefix so a joined DataFrame stays
# self-documenting next to mean_free / chi_twist / mean_info.
#
# ``fp_largest_basin_fraction`` (added at fp_version=5) is the per-σ
# *coverage* scalar of the twists-home-vectors paper (Eq. coverage):
# ``coverage(σ) = max_ℓ ( b_ℓ / |S°| )`` — the largest single basin across
# all four labels, as a fraction of non-wall states.  Aggregated by MAX
# across labels (not mean), so a value of 1.0 means at least one label's
# deterministic skeleton drains the whole walkable env into one basin.
FINGERPRINT_FIELDS: tuple[str, ...] = (
    tuple(f"fp_{name}" for name in PER_LABEL_FIELDS) + ("fp_largest_basin_fraction",)
)


def _aggregate_across_labels(per_label: dict[str, list[float]]) -> dict[str, float]:
    """Reduce per-label lists to scalars.

    Mean for most stats; max for ``diameter`` because a single
    long-reach label is the informative signal.
    """
    out: dict[str, float] = {}
    for field, values in per_label.items():
        arr = np.asarray(values, dtype=float)
        if field == "diameter":
            out[f"fp_{field}"] = float(arr.max())
        else:
            out[f"fp_{field}"] = float(arr.mean())
    return out


def fingerprint_for_sigma(env, sigma: np.ndarray) -> dict[str, float]:
    """Aggregate per-label functional-graph statistics for one twist.

    Applies the twist by indexing into the bare-action successor table.
    The codebase convention (see grid_room.py:147-152) is
    ``sigma[s, a_phys] -> label``, with cached inverse
    ``sigma_inv[s, label] -> a_phys``.  The per-label functional graph
    is therefore::

        f_label(s) = base_succ[sigma_inv[s, label], s]

    i.e., emitting label ``ell`` at state ``s`` fires the physical
    action ``sigma_inv[s, ell]``, whose deterministic successor we look
    up.  This matches what ``apply_twist`` then
    ``decompose_all_labels`` compute on the twisted env (verified by
    regression test ``test_fingerprint_matches_decompose_all_labels``).

    Args:
        env: GridRoom (or compatible) — the *untwisted* env with no
            absorbing goal.  See ``build_twisted_env_no_goal`` for the
            canonical construction.
        sigma: ``(nS, nA)`` int array; each row is a permutation of
            ``{0, ..., nA-1}`` in the forward convention
            ``sigma[s, a_phys] -> label``.

    Returns:
        Dict with nine ``fp_*`` keys: ``fp_n_basins``, ``fp_n_cyclic``,
        ``fp_n_terminal``, ``fp_mean_cycle_length``,
        ``fp_cycle_basin_ratio``, ``fp_mean_tail_length``,
        ``fp_mean_rho``, ``fp_diameter``, and
        ``fp_largest_basin_fraction`` (the coverage scalar).
    """
    sigma = np.asarray(sigma, dtype=int)
    nS = int(env.nS)
    nA = int(env.nA)
    if sigma.shape != (nS, nA):
        raise ValueError(
            f"sigma shape {sigma.shape} does not match env (nS={nS}, nA={nA})"
        )

    base_succ = np.stack(
        [deterministic_successor(env, a) for a in range(nA)], axis=0,
    )                                                    # (nA, nS)
    state_idx = np.arange(nS)
    sigma_inv = np.argsort(sigma, axis=1)                # label -> a_phys

    # Walls are inert: they don't contribute to any fingerprint number.
    # Without this, deterministic_successor's argmax-of-zeros on an
    # unreachable wall row returns state 0 and silently inflates state
    # 0's basin (see decomposition.decompose docstring for the
    # defensive redirect that prevents this).
    walls = getattr(env, "walls_flat", None)

    per_label: dict[str, list[float]] = {f: [] for f in PER_LABEL_FIELDS}
    largest_basin_fracs: list[float] = []
    for ell in range(nA):
        succ = base_succ[sigma_inv[:, ell], state_idx]
        fg = decompose(succ, walls=walls)
        stats = per_label_stats(fg)
        for field, value in stats.items():
            per_label[field].append(value)
        # Per-label largest basin as a fraction of non-wall states.
        # sum(basin_sizes) == |S°| because every non-wall state lands in
        # exactly one basin (walls carry basin_id == -1, excluded).
        basin_sizes = np.asarray(fg.basin_sizes, dtype=float)
        n_nonwall = float(basin_sizes.sum())
        largest_basin_fracs.append(
            float(basin_sizes.max() / n_nonwall)
            if basin_sizes.size and n_nonwall > 0 else 0.0
        )

    out = _aggregate_across_labels(per_label)
    # Coverage (paper Eq. coverage): the single largest basin fraction across
    # all four labels, i.e. max_ℓ ( b_ℓ / |S°| ).  MAX, not mean.
    out["fp_largest_basin_fraction"] = (
        max(largest_basin_fracs) if largest_basin_fracs else 0.0
    )
    return out
