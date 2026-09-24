"""Validation helpers for the frozen cross-world null calculation."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def validated_free_energy_mean(
    decision_information,
    free_energy,
    states: Sequence[int],
    *,
    context: str,
) -> tuple[float, int, float]:
    """Return a finite mean and diagnostics, or fail loudly on a bad solve."""
    if not bool(decision_information.converged):
        raise RuntimeError(
            f"free-energy solve did not converge: {context} "
            f"iterations={decision_information.iteration_count}"
        )

    iterations = int(decision_information.iteration_count)
    residual = float(decision_information.last_blahut_residual)
    if iterations < 0:
        raise RuntimeError(f"free-energy solve reported negative iterations: {context}")
    if not math.isfinite(residual):
        raise RuntimeError(
            f"free-energy solve reported non-finite residual {residual}: {context}"
        )

    state_ids = np.asarray(list(states), dtype=int)
    if state_ids.size == 0:
        raise ValueError(f"free-energy mean received no states: {context}")
    values = np.asarray(free_energy, dtype=float)[state_ids]
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"free-energy solve returned non-finite values: {context}")
    return float(values.mean()), iterations, residual
