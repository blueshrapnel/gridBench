# %% [markdown]
# # 18 — Silencing screen: per-run label usage shares across a fan
#
# **Question:** across a fan of GA run-bests, which runs evolve a
# *retired* (silenced) label, and where does it happen?  A silencing
# screen answers this per run, so it extends the retirement-affordance
# map beyond the four_rooms cohort to the untested geometry palette.
#
# **Silenced definition (canonical, 2026-07-16):** a run is `SILENCED`
# when its **minimum per-label greedy usage share falls below 5%** —
# at least one action label is issued in under 5% of goal-optimal greedy
# decisions, averaged over every walkable start/goal pair.  This is a
# property of the goal-optimal *policy ensemble*, not of the twist: a
# silenced label still moves the agent whenever pressed; the policies
# simply stop choosing it.  It matches the `<5%`-usage POLICY property
# pinned in twists-home-vectors sec:cycle-escape / sec:label-roles.
#
# Two quantities per label, both solved fresh (no cached `fp_*`, which
# describe the wrong-side sigma):
#
# 1. **usage share** — greedy-decision fraction on the twisted env
#    (gridcore, beta=1); the greedy action index is the label because
#    the twisted env's action axis is the label axis.
# 2. **coverage** — largest single-label basin as a fraction of walkable
#    cells, from the repeat-label graph `base_succ[sigma_inv[:, l], s]`
#    (the paper-figure graph, built from `sigma_inv`).
#
# The engine lives in `gridbench.silencing_screen` (promoted from the
# scratchpad `screen_catchment_scale.py` / `screen_mechanism_fan.py` so
# it survives across sessions).  It reads `env_id` / `shape` per run from
# the summary, so a mixed-environment fan works without configuration,
# and skips any dir lacking a summary+sigma — e.g. the 7x7
# `corr_four_rooms` cell, which only builds at 13x13 and is dropped from
# the geometry screen (see project-geometry-fan-corr-four-rooms-dropped).

# %%
from gridbench.silencing_screen import screen_fan

BASE = "/media/merlin/grid-twist/gridtwist-outputs"


def _summarise(results):
    n_sil = sum(r.silenced for r in results)
    print(f"--- {len(results)} runs screened, "
          f"{n_sil} SILENCED, {len(results) - n_sil} balanced")
    return results


# %% [markdown]
# ## Geometry-compression fan (2026-07-12) — 12-environment 7x7 screen
#
# Shuffle x free-energy g500 across the untested palette (x_wall,
# plus_cross, pillars 1–3, corr_1d_ring, wrap_pillar_3, pinwheel,
# open_grid, helical).  Does the retirement affordance appear off
# four_rooms?

# %%
geometry = _summarise(screen_fan(f"{BASE}/core-geometry-compression-12-07"))

# %% [markdown]
# ## Catchment-scale ladder (2026-07-15) — four_rooms 11x11 / 13x13
#
# Shuffle x free-energy g500 at larger scales, the matched contrast for
# the 9x9+ perm_balanced cohorts (see the TODO(init-check) in
# sec:label-roles).

# %%
catchment = _summarise(screen_fan(f"{BASE}/core-catchment-scale-15-07"))

# %% [markdown]
# ## Per-environment silenced tally (geometry fan)

# %%
from collections import Counter

sil = Counter(r.env_id for r in geometry if r.silenced)
tot = Counter(r.env_id for r in geometry)
for env in sorted(tot):
    print(f"  {env:16} silenced {sil[env]}/{tot[env]}")
