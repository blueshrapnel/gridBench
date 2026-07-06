# %% [markdown]
# # Goal-geometry series, 00 — the figure-3 benchmark
#
# Durable home for the 2026-07-06 benchmark that ran as session scratch:
# the space-of-goals paper's figure-3 computation (per-goal free-energy
# fields for all 121 goals; 11x11 open grid, MOORE moves, beta=100,
# LIVE state distribution, det=1.0, theta=1e-5) timed on the modern
# gridCore kernel and on the gridFour kernel, both validated against
# the paper's own 2021 pickle.
#
# Recorded results (2026-07-06, gridCore d432c4b = fast-path items 1-5
# + guarded Anderson):
#   gridFour kernel:            804 ms/goal  (~97 s projected full sweep)
#   gridCore plain:             328 ms/goal  ( 39.7 s full sweep)  2.45x
#   gridCore + Anderson:        341 ms/goal  -- AA LOSES on live dist
#   deviation vs 2021 pickle:   max ~2.0e-2 for BOTH kernels (identical),
#     i.e. the paper-era live-distribution path difference, not a
#     kernel artefact: the modern kernels agree with each other.
#
# NOTE this notebook deliberately imports the gridFour kernel for the
# comparison — a workbench exception to the porting rule, confined to
# the benchmark.
#
# WALL-TIME CAVEAT: the header numbers above are the canonical
# measurements (idle machine, 2026-07-06 afternoon).  Re-executions
# report correct deviations and iteration counts regardless of load
# (verified: a rerun on a load-40 box reproduced 8.9e-3/2.0e-2 and the
# exact iteration counts while wall times inflated 30x) — but quote
# timings only from an idle machine.

# %%
import pickle
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

import gridcore

repo = Path(gridcore.__file__).resolve().parents[2]
try:
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], text=True
    ).strip()
except Exception:
    commit = "unknown"
print(f"gridcore: {repo.name} @ {commit}")

# %% [markdown]
# ## Reference: the paper's pickle (pandas shim for pre-2.0 indexes)

# %%
import pandas as pd

_shim = types.ModuleType("pandas.core.indexes.numeric")
_shim.Int64Index = pd.Index
_shim.Float64Index = pd.Index
_shim.UInt64Index = pd.Index
sys.modules["pandas.core.indexes.numeric"] = _shim
sys.path.insert(0, "/media/merlin/phd-marlyn/cognitive-geometry/src")

PICKLE = Path("/media/merlin/phd-marlyn/cognitive-geometry/data/11-11-det"
              "/data-11-11-moo-liv-det-1.0-b-100-Z.pickle")
with open(PICKLE, "rb") as fh:
    ref = pickle.load(fh)
frees_ref = np.asarray(ref["frees"], dtype=float)   # frees[g][s]
BETA, THETA, SHAPE = float(ref["beta"]), float(ref["theta"]), tuple(ref["shape"])
print(f"reference: {PICKLE.name}, frees {frees_ref.shape}, "
      f"beta={BETA}, theta={THETA}")

# %% [markdown]
# ## gridCore: full 121-goal sweep, plain and accelerated

# %%
from gridcore.bridge import build_open_grid_env
from gridcore.info import DecisionInformation
from gridcore.planning.state_distribution import LiveStateDistribution


def gridcore_sweep(accelerate):
    t0 = time.perf_counter()
    iters = 0
    devs = []
    for goal in range(121):
        env = build_open_grid_env(shape=(11, 11), goal=int(goal),
                                  determinism=1.0, manhattan=False)
        di = DecisionInformation(env, LiveStateDistribution(env), THETA,
                                 max_iterations=200_000, max_info_iterations=10_000,
                                 accelerate=accelerate)
        _, _, F = di.get_opt_policy_Z_free_vector(BETA)
        assert di.converged
        iters += di.iteration_count
        devs.append(float(np.max(np.abs(F - frees_ref[goal]))))
    return time.perf_counter() - t0, iters, float(np.median(devs)), float(np.max(devs))


dt_p, it_p, med_p, max_p = gridcore_sweep(False)
dt_a, it_a, med_a, max_a = gridcore_sweep(True)
print(f"gridCore plain: {dt_p:6.1f} s ({dt_p/121*1000:.0f} ms/goal, {it_p} iters) "
      f"| dev vs pickle med {med_p:.1e} max {max_p:.1e}")
print(f"gridCore accel: {dt_a:6.1f} s ({dt_a/121*1000:.0f} ms/goal, {it_a} iters) "
      f"| dev med {med_a:.1e} max {max_a:.1e} | AA factor {dt_p/dt_a:.2f}x wall")

# %% [markdown]
# ## gridFour kernel: sampled goals (the old-code baseline)

# %%
# The cognitive-geometry path was only needed to unpickle; remove it and
# purge its top-level modules (it shares bare package names with
# gridFour: planning/env/utility) before importing the gridFour kernel.
# This collision is exactly what the namespaced gridcore/gridvis family
# eliminates; it survives only in this deliberately-two-legacy notebook.
sys.path.remove("/media/merlin/phd-marlyn/cognitive-geometry/src")
for mod in [m for m in list(sys.modules)
            if m == "planning" or m.startswith(("planning.", "env", "utility"))]:
    del sys.modules[mod]
sys.path.insert(0, "/media/merlin/phd-marlyn/gridFour/src")
from env.grid_room import GridRoom as GF_GridRoom                    # noqa: E402
from planning.state_distribution import LiveStateDistribution as GF_Live  # noqa: E402
from planning.decision_information import DecisionInformation as GF_DI    # noqa: E402

sample = list(range(0, 121, 10))
t0 = time.perf_counter()
devs = []
for g in sample:
    env = GF_GridRoom({"shape": (11, 11), "goals": [g], "walls": [],
                       "manhattan": False, "determinism": 1.0, "epsilon": 0.0})
    di = GF_DI(env, GF_Live(env), THETA,
               max_iterations=200_000, max_info_iterations=10_000)
    _, _, F = di.get_opt_policy_Z_free_vector(BETA)
    devs.append(float(np.max(np.abs(np.asarray(F, dtype=float) - frees_ref[g]))))
dt_gf = time.perf_counter() - t0
print(f"gridFour: {dt_gf/len(sample)*1000:.0f} ms/goal over {len(sample)} sampled goals "
      f"| dev vs pickle max {max(devs):.1e} "
      f"| projected full sweep {dt_gf/len(sample)*121:.0f} s "
      f"| gridCore speedup {(dt_gf/len(sample)*121)/dt_p:.2f}x")

# %% [markdown]
# ## Discernments
#
# - gridCore runs the paper's flagship computation ~2.5x faster than the
#   gridFour kernel (fast-path items 1-5; the margin grows with grid
#   size, since item 5 removed a per-iteration temporary that scales
#   with nS^2).
# - BOTH kernels deviate from the 2021 pickle by the same ~2e-2 — the
#   modern kernels agree with each other; the difference is paper-era
#   live-distribution path dependence, not an extraction artefact.
# - Anderson LOSES slightly on live-distribution solves: p_s mutates
#   every iteration, the F map is non-stationary throughout, and the
#   accelerator's history keeps invalidating.  Selective-enablement
#   policy: AA on for uniform-distribution work, off for live.
