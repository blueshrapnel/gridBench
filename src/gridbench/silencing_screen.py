"""Silencing screen: per-run label usage-share and coverage over a GA fan.

Promoted from the scratchpad ``screen_catchment_scale.py`` / ``screen_mechanism_fan.py``
pattern (2026-07-16) so it survives across sessions and can be pointed at
any fan, not just the four_rooms catchment ladder.

For each run-best twist in a fan it computes, per action *label*:

- **usage share** -- the fraction of goal-optimal greedy decisions that
  issue that label, averaged over every walkable start/goal pair.  The
  policies are solved fresh with gridcore at ``beta`` on the twisted env,
  so the greedy action index *is* the label (the twisted env's action
  axis is the label axis; see the twisted-env convention).
- **coverage** -- the largest single-label basin as a fraction of the
  walkable cells, from the repeat-label functional graph
  ``base_succ[sigma_inv[:, label], state]`` (the paper's "repeat one
  label" graph, built from ``sigma_inv`` so it is the paper-figure graph,
  not the wrong-side sigma of the cached ``fp_*`` fields).

**Silenced definition (canonical):** a run is flagged ``SILENCED`` when
its minimum per-label usage share falls below ``silence_threshold``
(default ``0.05`` = 5%): at least one label is issued in under 5% of
goal-optimal greedy decisions.  This is a property of the goal-optimal
*policy ensemble*, not of the twist geometry -- a silenced label still
moves the agent whenever pressed.

The env is read per-run from the summary ``config.env_id`` / ``config.shape``,
so a mixed-environment fan (e.g. the geometry-compression screen) works
without per-call configuration.  Runs whose dir lacks a
``*-multi-all.summary.json`` or ``*-multi-all.sigma.npy`` are skipped, so
an incomplete cell (e.g. the 7x7 ``corr_four_rooms`` that only builds at
13x13) drops out rather than raising.
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass

import numpy as np

from gridcore.bridge import (
    EvalConfig,
    _state_dist_class,
    build_twisted_env_from_sigma,
)
from gridcore.info import DecisionInformation
from gridbench.functional_graph.decomposition import decompose, deterministic_successor
from gridbench.functional_graph.probe_env import build_goal_free_probe_env

SILENCE_THRESHOLD = 0.05  # min per-label greedy usage share below which a run is SILENCED


@dataclass
class SilencingResult:
    """One run's silencing screen record."""

    name: str  # run dir tail after '-core-ga-'
    env_id: str
    shape: tuple[int, int]
    free_energy: float  # best_expected_free from the summary
    usage: np.ndarray  # per-label greedy usage share, sums to 1
    coverage: np.ndarray  # per-label largest-basin fraction of walkable cells
    silenced: bool

    @property
    def min_share(self) -> float:
        return float(self.usage.min())

    @property
    def dom_coverage(self) -> float:
        return float(self.coverage.max())

    @property
    def verdict(self) -> str:
        return "SILENCED" if self.silenced else "balanced"

    def format_row(self) -> str:
        return (
            f"{self.name:38} F={self.free_energy:7.3f} "
            f"usage={np.round(self.usage, 2)} min={self.min_share:5.1%} "
            f"cov={np.round(self.coverage, 2)} domcov={self.dom_coverage:.2f}  "
            f"{self.verdict}"
        )


def _probe_pack_cache() -> dict:
    return {}


def _probe_pack(cache: dict, env_id: str, shape: tuple[int, int], determinism: float):
    """Walls, state count, and per-action successors for (env_id, shape)."""
    key = (env_id, shape)
    if key not in cache:
        e = build_goal_free_probe_env(env_id, shape, determinism)
        wf = getattr(e, "walls_flat", None)
        walls = set(int(w) for w in np.ravel(wf)) if wf is not None else set()
        n_states = shape[0] * shape[1]
        succ = np.stack([deterministic_successor(e, a) for a in range(4)], axis=0)
        cache[key] = (walls, n_states, succ)
    return cache[key]


def screen_run(
    run_dir: str,
    *,
    beta: float = 1.0,
    silence_threshold: float = SILENCE_THRESHOLD,
    _pack_cache: dict | None = None,
) -> SilencingResult | None:
    """Screen one run dir; return ``None`` if it has no summary+sigma."""
    summary = glob.glob(run_dir + "/*-multi-all.summary.json")
    sigma_path = glob.glob(run_dir + "/*-multi-all.sigma.npy")
    if not summary or not sigma_path:
        return None

    s = json.load(open(summary[0]))
    cfg = s["config"]
    env_id = cfg["env_id"]
    shape = tuple(cfg["shape"])
    determinism = float(cfg.get("determinism", 0.97))

    cache = _pack_cache if _pack_cache is not None else _probe_pack_cache()
    walls, n_states, base_succ = _probe_pack(cache, env_id, shape, determinism)
    walkable = [x for x in range(n_states) if x not in walls]

    sigma = np.load(sigma_path[0]).astype(int)

    # Per-label greedy usage share over all walkable start/goal pairs.
    usage = np.zeros(4)
    for goal in walkable:
        c = EvalConfig(
            env_id=env_id, shape=shape, goal=int(goal), beta=beta,
            determinism=determinism, manhattan=True, theta=1e-5, state_dist="uniform",
        )
        e = build_twisted_env_from_sigma(sigma, c)
        di = DecisionInformation(
            e, _state_dist_class("uniform")(e), 1e-5,
            max_iterations=200_000, max_info_iterations=10_000,
        )
        pi, _, _ = di.get_opt_policy_Z_free_vector(beta)
        greedy = np.argmax(np.asarray(pi, dtype=float), axis=1)
        for st in walkable:
            if st != goal:
                usage[greedy[st]] += 1
    usage /= usage.sum()

    # Per-label coverage from the repeat-label graph (sigma_inv side).
    sigma_inv = np.argsort(sigma, axis=1)
    idx = np.arange(n_states)
    coverage = np.empty(4)
    for label in range(4):
        d = decompose(base_succ[sigma_inv[:, label], idx], walls=walls)
        bs = np.asarray(d.basin_sizes, dtype=int)
        coverage[label] = bs.max() / len(walkable) if bs.size else 0.0

    name = run_dir.rstrip("/").split("/")[-1].split("-core-ga-")[-1]
    return SilencingResult(
        name=name, env_id=env_id, shape=shape,
        free_energy=float(s["best_expected_free"]),
        usage=usage, coverage=coverage,
        silenced=bool(usage.min() < silence_threshold),
    )


def screen_fan(
    base: str,
    *,
    beta: float = 1.0,
    silence_threshold: float = SILENCE_THRESHOLD,
    verbose: bool = True,
) -> list[SilencingResult]:
    """Screen every run dir under ``base``; skip incomplete cells.

    Prints one row per run as it completes when ``verbose`` (the screen
    is slow -- a full per-goal policy solve per run -- so streaming keeps
    the operator informed).  Returns the results in dir-sorted order.
    """
    cache = _probe_pack_cache()
    results: list[SilencingResult] = []
    for run in sorted(glob.glob(base + "/*")):
        r = screen_run(
            run, beta=beta, silence_threshold=silence_threshold, _pack_cache=cache,
        )
        if r is None:
            continue
        results.append(r)
        if verbose:
            print(r.format_row(), flush=True)
    return results
