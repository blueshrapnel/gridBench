# %% [markdown]
# # Goal-geometry series, 23 — a re-scoring bound for the second-leg advantage
#
# Companion to notebook 22. With the second leg re-optimised for its own start the defect is
#     dF_re = slack - A1 - A2,
# A1 the report's first-leg prior advantage and A2 = F_g(s') - F_{g|s'}(s') the second-leg one.
# By the same lemma as the report's bound, A2 <= F^{pi_{g|s'}}(s')[q_g] - F^{pi_{g|s'}}(s')[q_{g|s'}]
# = T2 * KL(q_{g|s'} || q_g) / beta, because q_{g|s'} is the journey's own marginal (ledger identity).
# So  -dF_re <= C1/beta + C2/beta - slack.  This notebook computes C2 for every (s', g) and checks
# the two-leg bound on every triple of the 13x13 four rooms (det 0.97, pooled, beta 1 and 0.3).

# %%
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from gridcore.bridge import build_env_by_id
from gridcore.info.fixed_point import absorb, occupancy, solve_self_consistent, transition_tensor

OUT = Path("/media/merlin/Dropbox/workbench/topics/switching-costs/figure-code/second-leg-reoptimised")
env = build_env_by_id(env_id="four_rooms", shape=(13, 13), goal=0, determinism=0.97, manhattan=True)
T, _ = transition_tensor(env); n = len(T); T_G = None

def _init(T):
    global T_G; T_G = T

def kl(p, q):
    m = p > 0; return float((p[m] * np.log2(p[m] / q[m])).sum())

def _solve(args):
    sp, g, beta, q0, qg = args
    w = np.zeros(n); w[sp] = 1.0
    s = solve_self_consistent(T_G, g, beta, w, q0=q0)
    N = occupancy(np.einsum("sa,saj->sj", s.policy, absorb(T_G, g)), g)
    T2 = N[sp].sum(); phat = N[sp] @ s.policy / T2
    return sp, g, T2, kl(s.prior, qg) * T2 / beta, float(np.abs(phat - s.prior).max()), s.F[sp]

for beta in [float(b) for b in (sys.argv[1:] or ["1", "0.3"])]:
    e19 = np.load(f"/media/merlin/fixed-point-sweep/switching-twisted/untwisted-pooled-b-{beta:g}.npz")   # [sp, s, g]
    d = np.load(f"/media/merlin/fixed-point-sweep/four_rooms-13x13-det-0.97-untwisted-pooled-b-{beta:g}.npz")
    Q, F = d["q"], d["F"]
    t0 = time.time(); C2 = np.zeros((n, n)); T2 = np.zeros((n, n)); selfc = np.zeros((n, n)); F_re = np.full((n, n), np.nan)
    with Pool(8, initializer=_init, initargs=(T,)) as pool:
        for sp, g, t2, c2, sc, fr in pool.imap_unordered(_solve, [(sp, g, beta, Q[g], Q[g]) for g in range(n) for sp in range(n) if sp != g], chunksize=16):
            C2[sp, g] = c2; T2[sp, g] = t2; selfc[sp, g] = sc; F_re[sp, g] = fr
    off = ~np.eye(n, dtype=bool)
    A2 = F - F_re
    slack, A1, C1, dF = e19["slack"], e19["A"], e19["C"], e19["dF"]
    valid = np.isfinite(slack) & np.isfinite(A1) & np.isfinite(C1)
    for i in range(n): valid[i, i, :] = False; valid[i, :, i] = False; valid[:, i, :][:, i] = False
    A2b = np.broadcast_to(A2[:, None, :], (n, n, n)); C2b = np.broadcast_to(C2[:, None, :], (n, n, n))
    dF_re = dF - A2b
    holds_old = (-dF_re <= C1 + 1e-6); holds_new = (-dF_re <= C1 + C2b - 0 + 1e-6)   # C arrays in e19 are already /beta
    holds_new_minus_slack = (-dF_re <= C1 + C2b - slack + 1e-6)
    viol = (dF_re < -1e-9) & valid
    print(f"beta {beta}: {off.sum()} re-solves in {time.time()-t0:.0f}s; self-consistency max|journey marginal - prior| {selfc[off].max():.1e}")
    print(f"  A2 <= C2 on {(A2[off] <= C2[off] + 1e-6).mean():.6f} of pairs; C2/A2 median {np.median(C2[off]/np.maximum(A2[off],1e-9)):.2f}; C1 median over triples {np.median(C1[valid]):.3f}, C2 median {np.median(C2b[valid]):.3f}")
    print(f"  re-optimised violation covered by C1 alone (the report's bound): {holds_old[valid].mean():.6f}; by C1 + C2: {holds_new[valid].mean():.6f}; by C1 + C2 - slack: {holds_new_minus_slack[valid].mean():.6f}")
    ratio = (C1 + C2b - slack)[viol] / (-dF_re[viol])
    print(f"  on the {viol.sum()} re-optimised violators, (C1 + C2 - slack) / violation: median {np.median(ratio):.2f}, 10th pct {np.percentile(ratio, 10):.2f}, min {ratio.min():.3f}")
    np.savez_compressed(OUT / f"second-leg-bound-b-{beta:g}.npz", C2=C2, T2=T2, A2=A2)
    json.dump(dict(beta=beta, A2_le_C2=float((A2[off] <= C2[off] + 1e-6).mean()), C2_over_A2_median=float(np.median(C2[off]/np.maximum(A2[off],1e-9))),
                   covered_C1=float(holds_old[valid].mean()), covered_C1_C2=float(holds_new[valid].mean()), covered_C1_C2_minus_slack=float(holds_new_minus_slack[valid].mean()),
                   violators=int(viol.sum()), ratio_median=float(np.median(ratio)), ratio_min=float(ratio.min())), open(OUT / f"second-leg-bound-b-{beta:g}.json", "w"), indent=1)
