"""
Goal-marginalised policy computation.

Given a directory of per-goal npz files produced by the hive_sigma evaluation
pipeline, computes the action distribution at each state marginalised over all
evaluated goals:

    π_marg(a | s)  =  (1 / |G|)  Σ_{g ∈ G}  π*(a | s, g)

where π*(·|s, g) is the Blahut–Arimoto optimal policy for goal g under the
twisted MDP.

The result is a (nS, nA) numpy array whose rows are valid probability
distributions (sum to 1) for non-terminal, non-wall states.  Wall and
terminal (goal) states are zeroed out by the solver and remain zero in the
marginal.

Usage
-----
    from utility.marginal_policy import load_goal_marginal_policy

    policy_marg = load_goal_marginal_policy(sigma_hive_dir)
    # shape: (nS, nA)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_goal_policy_stack(sigma_hive_dir: str | Path) -> tuple[np.ndarray, list[int]]:
    """Load per-goal policy arrays from a sigma hive directory.

    Parameters
    ----------
    sigma_hive_dir:
        Path to the ``sigma_id=<hash>`` directory that contains
        ``goal-<N>.npz`` files.

    Returns
    -------
    policies : ndarray, shape (n_goals, nS, nA)
        Stacked per-goal optimal policies.
    goals : list[int]
        Goal state indices in the order they were stacked.

    Raises
    ------
    FileNotFoundError
        If no ``goal-*.npz`` files are found.
    ValueError
        If the policy arrays have inconsistent shapes.
    """
    sigma_hive_dir = Path(sigma_hive_dir).expanduser().resolve()
    goal_files = sorted(sigma_hive_dir.glob("goal-*.npz"),
                        key=lambda p: int(p.stem.split("-")[1]))

    if not goal_files:
        raise FileNotFoundError(
            f"No goal-*.npz files found in {sigma_hive_dir}"
        )

    policies: list[np.ndarray] = []
    goals:    list[int]        = []

    for gf in goal_files:
        goal_idx = int(gf.stem.split("-")[1])
        data = np.load(gf)
        if "policy" not in data.files:
            continue
        policies.append(np.asarray(data["policy"], dtype=float))
        goals.append(goal_idx)

    if not policies:
        raise ValueError(f"No 'policy' arrays found in {sigma_hive_dir}")

    shapes = {p.shape for p in policies}
    if len(shapes) > 1:
        raise ValueError(
            f"Inconsistent policy shapes in {sigma_hive_dir}: {shapes}"
        )

    return np.stack(policies, axis=0), goals


def compute_goal_marginal_policy(
    policies: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Average per-goal policies into a single goal-marginal policy.

    Parameters
    ----------
    policies : ndarray, shape (n_goals, nS, nA)
    weights : optional 1-D array, length n_goals
        Goal weights.  Uniform if None.

    Returns
    -------
    ndarray, shape (nS, nA)
        π_marg(a | s) = Σ_g w_g · π*(a | s, g), normalised so rows sum to 1
        wherever any probability mass exists.
    """
    n_goals, nS, nA = policies.shape
    if weights is None:
        weights = np.ones(n_goals, dtype=float) / n_goals
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

    marginal = np.einsum("g,gas->as", weights, policies)  # (nS, nA)

    # Re-normalise rows that have any mass (handles rounding safely)
    row_sums = marginal.sum(axis=1, keepdims=True)
    safe = row_sums > 1e-12
    marginal = np.where(safe, marginal / np.where(safe, row_sums, 1.0), 0.0)

    return marginal


def load_goal_marginal_policy(
    sigma_hive_dir: str | Path,
    *,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Convenience wrapper: load + compute goal-marginal policy in one call.

    Returns
    -------
    marginal : ndarray, shape (nS, nA)
    goals    : list[int]  — goal indices that were averaged over
    """
    policies, goals = load_goal_policy_stack(sigma_hive_dir)
    marginal = compute_goal_marginal_policy(policies, weights=weights)
    return marginal, goals


def find_sigma_hive_dir(run_dir: str | Path, sigma_hash: str) -> Path | None:
    """Glob for the ``sigma_id=<hash>`` directory inside a run's hive_sigma tree."""
    run_dir = Path(run_dir).expanduser().resolve()
    matches = list(run_dir.glob(f"hive_sigma/**/sigma_id={sigma_hash}"))
    return matches[0] if matches else None
