"""Compatibility exports for the GridCore functional-graph implementation.

The implementation moved to GridCore in August 2026 so GridBench and
GridTwist share one tested definition.  Existing notebooks may retain their
GridBench imports.
"""

from gridcore.functional_graph import (
    FunctionalGraph,
    PER_LABEL_FIELDS,
    decompose,
    deterministic_successor,
    per_label_stats,
)
from gridcore.twists import validate_sigma

__all__ = [
    "FunctionalGraph",
    "PER_LABEL_FIELDS",
    "decompose",
    "deterministic_successor",
    "per_label_stats",
    "validate_sigma",
]
