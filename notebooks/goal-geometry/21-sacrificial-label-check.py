# %% [markdown]
# # Goal-geometry series, 21 — is there a sacrificial label in the evolved 13×13 four rooms?
#
# The habits note (workbench 07-habits-twists-frames) says that in some twisted worlds
# one action is *sacrificial*: never used by the policy and therefore never refined.
# At the goal-prior level the dropped labels of the evolved 13×13 four-rooms twists are
# spread over all four labels (Mac session, 7 Oct 2026), so no label is sacrificed
# across the world there. This notebook asks the question at three levels, for each
# world, reference and beta:
#
#   1. prior level   per label: goals whose prior drops it (< 1e-6), mean prior mass over
#                    goals, mass in the average of the goals' priors;
#   2. policy level  per label: share of states where the goal policy puts mass > 1e-3 on
#                    it (mean over goals), and share of (start, goal) journeys that use it
#                    (expected number of decisions with that label above 1e-3);
#   3. label graphs  per label: coverage of its largest basin under the intended move
#                    (the twists work's dominant/silent ordering), for context.
#
# A uniform-reference sacrificial claim names a label dropped by more than 80 % of goals
# or used on fewer than 1 % of journeys; the check is whether the pooled solve drops the
# same label.
#
# Data: the corrected-solver files in /media/merlin/fixed-point-sweep/ (the evolved
# twists' pooled AND uniform files were re-solved on 7 Oct with gridcore.info.fixed_point,
# so the recheck JSON's corrections are already in them; untwisted and random are the
# 2 Oct files, which have no spurious drops). Twist convention: label b at state s
# executes argsort(sigma[s])[b].
#
# Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=<gridCore fixed-point branch>/src python 21-...py

# %%
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import glob, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/media/merlin/Dropbox/phd/code/fieldInfo/stage2/ba_comparison")
from twist_compare import base_world, twisted, chi_twist, EVOLVED, ROOT, SIDE   # noqa: E402
from gridcore.info.fixed_point import absorb, occupancy                           # noqa: E402

SWEEP = Path("/media/merlin/fixed-point-sweep")
OUT = Path("/media/merlin/Dropbox/workbench/topics/twisted-switching/results/sacrificial-label-check.json")
WORLDS = {"untwisted": None, "random_twist": "random", "evolved_lowest_F": "evolved, lowest F",
          "evolved_high_chi_warm": "evolved, high χ, warm", "evolved_highest_chi": "evolved, highest χ"}
T0, walk = base_world(); n = len(walk); idx = np.arange(n)

def sigma_of(tag):
    key = WORLDS[tag]
    if key is None: return np.tile(np.arange(4), (SIDE * SIDE, 1))
    if key == "random":
        rng = np.random.default_rng(20261002); return np.array([rng.permutation(4) for _ in range(SIDE * SIDE)])
    return np.load(glob.glob(ROOT + EVOLVED[key])[0])

def label_coverage(T):
    """Largest-basin coverage of each label's intended-move functional graph on the walkable states."""
    succ = T.argmax(2)                       # [s, b] -> most likely successor
    cov = []
    for b in range(4):
        f = succ[:, b]
        # basin id = the cycle reached; find cycles by iterating
        cycle_of = np.full(n, -1); cycles = {}
        for s in range(n):
            path = []; x = s; seen = {}
            while cycle_of[x] < 0 and x not in seen:
                seen[x] = len(path); path.append(x); x = int(f[x])
            if cycle_of[x] >= 0: cid = cycle_of[x]
            else:
                cyc = tuple(sorted(path[seen[x]:])); cid = cycles.setdefault(cyc, len(cycles))
            for p in path: cycle_of[p] = cid
        sizes = np.bincount(cycle_of)
        cov.append(float(sizes.max() / n))
    return cov

results = []
for tag, key in WORLDS.items():
    sigma = sigma_of(tag); T = T0 if key is None else twisted(T0, walk, sigma)
    chi = 0.0 if key is None else chi_twist(sigma, walk)
    cov = label_coverage(T)
    for ref in ("uniform", "pooled"):
        for beta in (1.0, 0.3):
            d = np.load(SWEEP / f"four_rooms-13x13-det-0.97-{tag}-{ref}-b-{beta:g}.npz")
            pol, Q = d["policies"], d["q"]
            dropped = (Q < 1e-6).sum(0)
            used_states = np.zeros(4); used_journeys = np.zeros(4)
            for g in range(n):
                keep = idx != g
                used_states += (pol[g][keep] > 1e-3).mean(0)
                N = occupancy(np.einsum("sa,saj->sj", pol[g], absorb(T, g)), g)
                expected_label_decisions = N @ pol[g]              # [s, b]
                used_journeys += (expected_label_decisions[keep] > 1e-3).mean(0)
            used_states /= n; used_journeys /= n
            row = dict(world=tag, chi=round(chi, 3), reference=ref, beta=beta,
                       goals_dropping=dropped.tolist(), mean_prior=Q.mean(0).round(4).tolist(),
                       share_of_states_using=used_states.round(3).tolist(), share_of_journeys_using=used_journeys.round(3).tolist(),
                       label_coverage=np.round(cov, 3).tolist(),
                       sacrificial_by_prior=[int(b) for b in range(4) if dropped[b] > 0.8 * n],
                       sacrificial_by_journeys=[int(b) for b in range(4) if used_journeys[b] < 0.01])
            results.append(row)
            print(f"{tag:22s} χ {chi:.2f} {ref:8s} β {beta:<4g} | goals dropping {dropped.tolist()} | mean prior {np.round(Q.mean(0),3).tolist()} "
                  f"| states using {np.round(used_states,2).tolist()} | journeys using {np.round(used_journeys,2).tolist()} | coverage {np.round(cov,2).tolist()}", flush=True)

OUT.write_text(json.dumps(results, indent=1) + "\n")
# the verdict
for tag in WORLDS:
    uni = [r for r in results if r["world"] == tag and r["reference"] == "uniform"]
    pooled = [r for r in results if r["world"] == tag and r["reference"] == "pooled"]
    cand = sorted(set(b for r in uni for b in r["sacrificial_by_prior"] + r["sacrificial_by_journeys"]))
    same = [b for b in cand if all(b in r["sacrificial_by_prior"] + r["sacrificial_by_journeys"] for r in pooled)]
    print(f"{tag:22s}: uniform-reference sacrificial candidates {cand or 'none'}; also sacrificial under pooled: {same or 'none'}")
