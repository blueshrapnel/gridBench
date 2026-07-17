# %% [markdown]
# # Single-label reach by class, 7x7 and 9x9
#
# Rebuilds the paper's reach-recovery comparison from the current gridBench
# functional-graph implementation.  The retained environment palette excludes
# the legacy x_wall robustness case.

# %%
from pathlib import Path
import sys

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path("/media/merlin/phd-marlyn/gridBench/notebooks/single-label-reach")

ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridbench.functional_graph.decomposition import (
    decompose,
    deterministic_successor,
    per_label_stats,
)
from gridbench.functional_graph.probe_env import build_goal_free_probe_env


CACHE_ROOT = Path("/media/merlin/grid-twist/data-schema-10/reports/_cache/functional_graph")
PAPER_FIGURE = Path(
    "/home/karen/Dropbox/phd/writing/twists-home-vectors/figures/F-reach-by-class.png"
)
DET = 0.97
ENVS = [
    "wrap_grid",
    "helical",
    "open_grid",
    "pinwheel",
    "four_rooms",
    "pillar_3",
]
SHAPES = [(7, 7), (9, 9)]
N_RANDOM = 2_000
COHORT_OBJECTIVES = {"decision_information", "free_energy"}


def walls_of(env):
    walls = getattr(env, "walls_flat", None)
    if walls is not None:
        return list(walls)
    return [s for s in range(int(env.nS)) if not env.T[s].any()]


def mean_rho_of_sigma(base, sigma, walls, n_states):
    n_actions = base.shape[0]
    sigma_inv = np.argsort(sigma, axis=1)
    state_idx = np.arange(n_states)
    means = []
    for label in range(n_actions):
        succ = base[sigma_inv[:, label], state_idx]
        means.append(per_label_stats(decompose(succ, walls=walls))["mean_rho"])
    return float(np.mean(means))


def cartesian_and_random(env_id, shape, seed=20260704):
    env = build_goal_free_probe_env(env_id, shape, DET)
    walls = walls_of(env)
    n_states, n_actions = int(env.nS), int(env.nA)
    base = np.stack(
        [deterministic_successor(env, action) for action in range(n_actions)]
    )
    identity = np.tile(np.arange(n_actions, dtype=int), (n_states, 1))
    cartesian = mean_rho_of_sigma(base, identity, walls, n_states)
    rng = np.random.default_rng(seed)
    random_reach = np.asarray(
        [
            mean_rho_of_sigma(
                base,
                np.stack([rng.permutation(n_actions) for _ in range(n_states)]),
                walls,
                n_states,
            )
            for _ in range(N_RANDOM)
        ]
    )
    return cartesian, random_reach


def ga_run_bests(env_id, shape):
    cache = (
        CACHE_ROOT
        / f"env_id={env_id}"
        / f"shape={shape[0]}x{shape[1]}"
        / f"det={DET}"
        / "beta=1"
        / "fingerprints.parquet"
    )
    if not cache.exists():
        return None
    frame = pd.read_parquet(
        cache,
        columns=[
            "fp_mean_rho",
            "is_run_best",
            "run_type",
            "fitness_objective",
            "run_name",
        ],
    )
    full_goal = ~(
        frame.run_name.str.contains("gss")
        | frame.run_name.str.contains(r"k0\d\d", regex=True)
    )
    selected = (
        frame.is_run_best
        & (frame.run_type == "multi")
        & (
            frame.fitness_objective.isin(COHORT_OBJECTIVES)
            | frame.fitness_objective.isna()
        )
        & full_goal
    )
    return frame.loc[selected, "fp_mean_rho"].to_numpy()


# %%
results = {}
for shape in SHAPES:
    for env_id in ENVS:
        run_bests = ga_run_bests(env_id, shape)
        if run_bests is None or len(run_bests) == 0:
            print(f"skip {env_id} {shape}: no cohort")
            continue
        cartesian, random_reach = cartesian_and_random(env_id, shape)
        results[(env_id, shape)] = {
            "cartesian": cartesian,
            "random": random_reach,
            "run_bests": run_bests,
        }
        print(
            f"{env_id} {shape}: Cartesian={cartesian:.2f}, "
            f"random median={np.median(random_reach):.2f}, "
            f"GA median={np.median(run_bests):.2f} (n={len(run_bests)})"
        )


# %%
figure, axes = plt.subplots(1, 2, figsize=(14.4, 5.0), dpi=150, sharey=True)
for axis, shape in zip(axes, SHAPES):
    environments = [env_id for env_id in ENVS if (env_id, shape) in results]
    positions = np.arange(len(environments))
    for position, env_id in enumerate(environments):
        result = results[(env_id, shape)]
        jitter = np.random.default_rng(7).uniform(
            -0.10, 0.10, len(result["run_bests"])
        )
        axis.scatter(
            np.full(len(result["run_bests"]), position) + jitter,
            result["run_bests"],
            s=14,
            color="#228833",
            alpha=0.5,
            zorder=3,
        )
        axis.scatter(
            [position],
            [np.median(result["run_bests"])],
            marker="*",
            s=200,
            facecolors="none",
            edgecolors="#228833",
            linewidths=1.6,
            zorder=4,
            label="GA run-bests (median $\\bigstar$, points)"
            if position == 0
            else None,
        )
        axis.scatter(
            [position],
            [result["cartesian"]],
            marker="o",
            s=110,
            facecolors="none",
            edgecolors="black",
            linewidths=1.8,
            zorder=5,
            label="Cartesian identity" if position == 0 else None,
        )
        low, high = np.percentile(result["random"], [5, 95])
        axis.plot(
            [position, position],
            [low, high],
            color="#ee6677",
            lw=3.0,
            alpha=0.45,
            solid_capstyle="butt",
            zorder=2,
            label=f"uniform-random null 5–95% (n={N_RANDOM:,})"
            if position == 0
            else None,
        )
        axis.scatter(
            [position],
            [np.median(result["random"])],
            marker="s",
            s=70,
            facecolors="none",
            edgecolors="#ee6677",
            linewidths=1.6,
            zorder=5,
            label="uniform-random median" if position == 0 else None,
        )
    first_walled = environments.index("pinwheel")
    axis.axvline(first_walled - 0.5, color="grey", lw=0.6, ls=":")
    axis.set_xticks(positions)
    axis.set_xticklabels(environments, rotation=12)
    axis.set_yscale("log")
    axis.set_title(f"${shape[0]} \\times {shape[1]}$", fontsize=12)
    axis.set_ylabel(r"mean single-label reach $\langle \rho \rangle$")
    axis.legend(fontsize=8, loc="upper right")

figure.suptitle(
    "Single-label reach by twist class: the walled-world recovery grows with room size",
    fontsize=12,
)
figure.tight_layout()
output = HERE / "figures" / "F-reach-by-class.png"
output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(output, bbox_inches="tight")
figure.savefig(PAPER_FIGURE, bbox_inches="tight")
print(f"saved {output}")
print(f"saved {PAPER_FIGURE}")
