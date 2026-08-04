"""Guards added after the 2026-08-04 functional-graph review.

Findings covered: (1) unknown env ids must not silently become open
grids and the fr- stack family must build with the evaluator's walls;
(2) malformed twists must be rejected; (4) single-label reach must
exclude walls on walled envs.
"""

import numpy as np
import pytest

from gridbench.functional_graph.decomposition import validate_sigma
from gridbench.functional_graph.fingerprint import fingerprint_for_sigma
from gridbench.functional_graph.probe_env import build_goal_free_probe_env
from gridbench.functional_graph.single_label_reach import reach_sample


def _identity_sigma(n_states, n_actions=4):
    return np.tile(np.arange(n_actions), (n_states, 1))


def test_unknown_env_id_raises():
    with pytest.raises(ValueError, match="unknown env_id"):
        build_goal_free_probe_env("four_roomz", (7, 7), 0.97)


def test_wall_builder_shape_rejection_raises():
    with pytest.raises(ValueError, match="rejected shape"):
        build_goal_free_probe_env("corr_four_rooms", (7, 7), 0.97)


def test_fr_stack_env_matches_gridcore_walls():
    from gridcore.bridge import EvalConfig, build_twisted_env_from_sigma
    for env_id in ("fr-R4Cu4Cd3-rot0", "fr-R2Cu2Cd3-rot90"):
        probe = build_goal_free_probe_env(env_id, (7, 7), 0.97)
        cfg = EvalConfig(env_id=env_id, shape=(7, 7), goal=0, beta=1.0,
                         determinism=0.97, manhattan=True, theta=1e-5,
                         state_dist="uniform")
        core = build_twisted_env_from_sigma(_identity_sigma(49), cfg)
        assert set(np.ravel(probe.walls_flat).tolist()) == \
            set(range(49)) - set(int(s) for s in core.available_states)


def test_fr_template_member_matches_four_rooms():
    a = build_goal_free_probe_env("four_rooms", (7, 7), 0.97)
    b = build_goal_free_probe_env("fr-R4Cu4Cd3-rot0", (7, 7), 0.97)
    assert set(np.ravel(a.walls_flat).tolist()) == set(np.ravel(b.walls_flat).tolist())


def test_validate_sigma_rejects_non_permutations():
    with pytest.raises(ValueError, match="not a permutation"):
        validate_sigma(np.zeros((49, 4), dtype=int))
    with pytest.raises(ValueError, match="must be"):
        validate_sigma(np.zeros((49, 3), dtype=int))
    validate_sigma(_identity_sigma(49))  # sane


def test_fingerprint_rejects_malformed_sigma():
    env = build_goal_free_probe_env("four_rooms", (7, 7), 0.97)
    with pytest.raises(ValueError, match="not a permutation"):
        fingerprint_for_sigma(env, np.zeros((49, 4), dtype=int))


def test_reach_sample_excludes_walls():
    env = build_goal_free_probe_env("four_rooms", (7, 7), 0.97)
    sample = reach_sample(env, label=0)
    walls = set(np.ravel(env.walls_flat).tolist())
    # wall states carry the decomposition's sentinel, not ordinary reach
    walkable_rho = [int(sample.rho[s]) for s in range(49) if s not in walls]
    wall_rho = [int(sample.rho[s]) for s in walls]
    assert min(walkable_rho) >= 1
    assert all(r < 1 for r in wall_rho)
