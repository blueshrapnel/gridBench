# %% [markdown]
# # 16 — Generation-zero populations on the (alignment, free energy) plane
#
# Question (2026-07-15, Karen): visualise the CORE difference between the
# two initialisers.  Everything measured says the bulks are identical and
# the difference is a tail: perm_balanced puts near-compasses in support
# by construction; shuffle does not.  Selection at generation zero is a
# max-operation, so the tail decides the founder and canalisation does
# the rest.  One figure: 96 individuals per arm (matched seed s173),
# alignment (1 - chi_twist, best-of-24) against full-goal mean free
# energy, gen-0 winner starred.
#
# Take-home the figure carries: an initialiser is a prior over coherent
# structures; report it like the fitness function.

# %%
import random
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "/media/merlin/phd-marlyn/gridTwist/src")
from evolution_core.initial_population import (  # noqa: E402
    build_individual_genes,
    build_population_perm_balanced,
)
from gridcore.bridge import EvalConfig, _state_dist_class, build_twisted_env_from_sigma  # noqa: E402
from gridcore.info import DecisionInformation as GC_DI  # noqa: E402
from gridbench.functional_graph.probe_env import build_goal_free_probe_env  # noqa: E402
from itertools import permutations  # noqa: E402

POP, SEED = 96, 173205080
WALLS = {4, 11, 25, 28, 29, 31, 32, 33, 38}
STATE_ORDER = [s for s in range(49) if s not in WALLS]
ORDS = np.array(list(permutations(range(4))))
FIG_DIR = Path(__file__).resolve().parent / "figs"

env = build_goal_free_probe_env("four_rooms", (7, 7), 0.97)
GOALS = STATE_ORDER


def genes_to_sigma(genes):
    sigma = np.tile(np.arange(4), (49, 1))
    rows = np.asarray(genes, dtype=int).reshape(len(STATE_ORDER), 4)
    for i, s in enumerate(STATE_ORDER):
        sigma[s] = rows[i]
    return sigma


def alignment(sigma):
    rows = sigma[STATE_ORDER]
    return float((rows[None, :, :] == ORDS[:, None, :]).mean(axis=(1, 2)).max())


def mean_free(sigma):
    tot = 0.0
    for g in GOALS:
        cfg = EvalConfig(env_id="four_rooms", shape=(7, 7), goal=int(g), beta=1.0,
                         determinism=0.97, manhattan=True, theta=1e-5, state_dist="uniform")
        e = build_twisted_env_from_sigma(sigma, cfg)
        di = GC_DI(e, _state_dist_class("uniform")(e), 1e-5,
                   max_iterations=200_000, max_info_iterations=10_000)
        _, _, F = di.get_opt_policy_Z_free_vector(1.0)
        tot += float(np.mean(F))
    return tot / len(GOALS)


rng = random.Random(SEED)
pb, _ = build_population_perm_balanced(
    state_order=STATE_ORDER, n_actions=4, population_size=POP,
    base_init_mode="hybrid_schedule", init_schedule="uniform",
    init_derangement_prob=0.5, init_derangement_power=1.0, dedupe=True, rng=rng)
rng2 = random.Random(SEED)
sh = [build_individual_genes(state_order=STATE_ORDER, n_actions=4, init_mode="shuffle",
                             target_epsilon=None, init_derangement_prob=0.5,
                             init_derangement_power=1.0, rng=rng2) for _ in range(POP)]

ARMS = {"perm_balanced": [genes_to_sigma(g) for g in pb],
        "shuffle": [genes_to_sigma(g) for g in sh]}

fig, ax = plt.subplots(figsize=(8.4, 6.4), dpi=150)
STYLE = {"perm_balanced": ("tab:blue", "o"), "shuffle": ("tab:orange", "^")}
for arm, sigmas in ARMS.items():
    als = np.array([alignment(s) for s in sigmas])
    print(f"{arm}: alignment min/med/max = {als.min():.2f}/{np.median(als):.2f}/{als.max():.2f}",
          flush=True)
    Fs = np.array([mean_free(s) for s in sigmas])
    c, m = STYLE[arm]
    ax.scatter(als, Fs, s=26, alpha=0.65, color=c, marker=m, label=f"{arm} (n={POP})")
    w = int(np.argmin(Fs))
    ax.scatter([als[w]], [Fs[w]], s=340, facecolors="none", edgecolors=c,
               linewidths=2.4, marker="*", zorder=5,
               label=f"{arm} gen-0 winner  (align {als[w]:.2f}, F {Fs[w]:.2f})")
    print(f"{arm}: winner alignment={als[w]:.3f} F={Fs[w]:.3f}", flush=True)
ax.axvline(0.32, color="grey", lw=0.8, ls=":", label="alignment chance level")
ax.set_xlabel("alignment  (1 − χ_twist, best-of-24 ordering match)")
ax.set_ylabel("full-goal mean free energy  (β = 1)")
ax.set_title("Generation-zero populations, four_rooms 7×7, matched seed s173")
ax.legend(fontsize=8)
out = FIG_DIR / "F-gen0-tail-plane.png"
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}")
