from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class FreeEnergyStats:
    """Summary statistics derived from a square matrix."""

    mean_free: float
    variance_free: float
    mean_relative_asymmetry: float
    symmetric: np.ndarray
    directed_count: int

    def as_tuple(self) -> Tuple[float, float, float]:
        """Return (mean, variance, relative asymmetry) for quick unpacking."""
        return self.mean_free, self.variance_free, self.mean_relative_asymmetry


def matrix_stats(matrix: np.ndarray, *, ignore_diagonal: bool = True) -> FreeEnergyStats:
    """
    Calculate summary statistics for an asymmetric square matrix.

    Parameters
    ----------
    frees:
        Square matrix whose (i, j) entry encodes the free energy of reaching goal i
        when starting from state j. The function expects ``frees`` to be coercible
        to ``np.ndarray`` and to have matching dimensions along both axes.
    ignore_diagonal:
        When ``True`` (default) the ``i == j`` entries are excluded from the averages.

    Returns
    -------
    FreeEnergyStats
        Dataclass containing the mean and variance of the directed matrix entries,
        the average percentage asymmetry between pairs ``(i, j)`` and ``(j, i)``, and
        the symmetric matrix ``(frees + frees.T) / 2`` that is typically fed into MDS.

    Raises
    ------
    ValueError
        If ``frees`` is not a two-dimensional square matrix.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix_stats expects a square 2D matrix.")

    n_states = matrix.shape[0]
    sym = 0.5 * (matrix + matrix.T)
    denom = np.maximum(sym, 1e-12)  # guard against division by zero
    percent_diff = np.abs(matrix - matrix.T) / denom

    if ignore_diagonal and n_states:
        mask = ~np.eye(n_states, dtype=bool)
    else:
        mask = np.ones_like(matrix, dtype=bool)

    directed_values = matrix[mask]
    percent_values = percent_diff[mask]

    if directed_values.size == 0:
        mean = variance = relative = 0.0
    else:
        mean = float(directed_values.mean())
        variance = float(directed_values.var())  # population variance keeps behaviour defined
        relative = float(percent_values.mean())

    return FreeEnergyStats(
        mean_free=mean,
        variance_free=variance,
        mean_relative_asymmetry=relative,
        symmetric=sym,
        directed_count=int(directed_values.size),
    )


def free_energy_stats(frees: np.ndarray, *, ignore_diagonal: bool = True) -> FreeEnergyStats:
    """Backwards-compatible wrapper for matrix_stats."""
    return matrix_stats(frees, ignore_diagonal=ignore_diagonal)
