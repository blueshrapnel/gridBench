# %% [markdown]
# # Goal-geometry series, 22 — is the second leg's policy the direct policy?
#
# Meeting of 7 Oct 2026: the report costs the second leg from the interim goal s' to g
# with the direct goal policy pi_g and its prior q_g (pooled over uniform starts). Daniel
# and Nicola suspect that a policy optimised for starts at s' alone, with the live
# reference of that journey, is a different policy with a lower free energy, so the
# defect's second-leg term is not fixed. This notebook re-solves the self-consistent
# problem for every (s', g) with the start distribution a point mass on s', and compares.
#
# Recorded per (s', g): the re-optimised prior q_{g|s'}, the re-optimised cost
# F_{g|s'}(s'), the direct cost F_g(s'), the saving F_g(s') - F_{g|s'}(s') >= 0, and the
# visitation-weighted L1 distance between pi_g and pi_{g|s'} along the journey from s'.
# Then over all triples (s, s', g): the defect with the fixed second leg (the report's)
# and with the re-optimised one, how many violations each has, and whether the report's
# re-scoring bound still covers the re-optimised violation (it is only proved for the
# fixed one).
#
# World: untwisted four rooms 13x13, det 0.97, pooled reference, beta 1 and 0.3, using the
# stored direct solutions in /media/merlin/fixed-point-sweep/.

# %%
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

from gridcore.bridge import build_env_by_id
from gridcore.info.fixed_point import absorb, occupancy, solve_self_consistent, transition_tensor, free_energy_of_policy

SWEEP = Path("/media/merlin/fixed-point-sweep")
OUT = Path("/media/merlin/Dropbox/workbench/topics/04c-switching-costs/figure-code/second-leg-reoptimised")
BETAS = [float(b) for b in (sys.argv[1:] or ["1", "0.3"])]
DET = float(os.environ.get("DET", "0.97"))          # 0.97 is the twists setting; the report's own four rooms are deterministic (DET=1)
TAG = f"four_rooms-13x13-det-{DET:g}"

env = build_env_by_id(env_id="four_rooms", shape=(13, 13), goal=0, determinism=DET, manhattan=True)
T, states = transition_tensor(env); n = len(states)
T_G = None

def _init(T):
    global T_G; T_G = T

def _solve(args):
    sp, g, beta, q0 = args
    w = np.zeros(n); w[sp] = 1.0
    s = solve_self_consistent(T_G, g, beta, w, q0=q0)
    return sp, g, s.prior, s.F[sp], s.policy, s.converged, s.certified()

for beta in BETAS:
    d = np.load(SWEEP / (f"{TAG}-untwisted-pooled-b-{beta:g}.npz" if DET != 1 else f"{TAG}-pooled-b-{beta:g}.npz"))
    pol, Q, F = d["policies"], d["q"], d["F"]            # pol[g, s, a], Q[g, a], F[s, g]
    N = np.array([occupancy(np.einsum("sa,saj->sj", pol[g], absorb(T, g)), g) for g in range(n)])   # N[g, s, x]
    t0 = time.time()
    tasks = [(sp, g, beta, Q[g]) for g in range(n) for sp in range(n) if sp != g]
    F_re = np.full((n, n), np.nan); q_re = np.full((n, n, 4), np.nan); pol_dist = np.full((n, n), np.nan)
    conv = np.zeros((n, n), bool); cert = np.zeros((n, n), bool)
    with Pool(8, initializer=_init, initargs=(T,)) as pool:
        for k, (sp, g, q, Fsp, pi, c, ce) in enumerate(pool.imap_unordered(_solve, tasks, chunksize=16)):
            F_re[sp, g] = Fsp; q_re[sp, g] = q; conv[sp, g] = c; cert[sp, g] = ce
            v = N[g, sp] / N[g, sp].sum()                         # visitation of the journey from s'
            pol_dist[sp, g] = float(v @ np.abs(pi - pol[g]).sum(1))
            if k % 4000 == 0: print(f"  beta {beta}: {k}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    off = ~np.eye(n, dtype=bool)
    saving = F - F_re                                              # F_g(s') - F_{g|s'}(s'), indexed [s', g]
    print(f"beta {beta}: {len(tasks)} re-solves in {time.time()-t0:.0f}s; converged {conv[off].mean():.4f} certified {cert[off].mean():.4f}")
    print(f"  saving F_g(s') - F_g|s'(s'): min {saving[off].min():.2e} median {np.median(saving[off]):.4f} mean {saving[off].mean():.4f} max {saving[off].max():.4f}; "
          f"relative to F_g(s'): median {np.median(saving[off]/F[off]):.4f} max {(saving[off]/F[off]).max():.4f}")
    print(f"  pairs with saving > 1e-6: {(saving[off] > 1e-6).mean():.4f}; > 0.01: {(saving[off] > 0.01).mean():.4f}; > 0.1: {(saving[off] > 0.1).mean():.4f}")
    print(f"  visitation-weighted L1 policy distance along the journey: median {np.median(pol_dist[off]):.4f} max {pol_dist[off].max():.4f}; > 0.01 in {(pol_dist[off] > 0.01).mean():.4f} of pairs")
    dq = np.abs(q_re - Q[None]).max(2)
    print(f"  prior change max|q_g|s' - q_g|: median {np.median(dq[off]):.4f} max {dq[off].max():.4f}")
    D_fixed = F[:, :, None] + F[None, :, :] - F[:, None, :]       # [s, s', g]
    D_re = F[:, :, None] + F_re[None, :, :] - F[:, None, :]
    valid = np.ones((n, n, n), bool)
    for i in range(n): valid[i, i, :] = False; valid[i, :, i] = False; valid[:, i, i] = False
    viol_fixed = (D_fixed < -1e-9) & valid; viol_re = (D_re < -1e-9) & valid
    print(f"  triples {valid.sum()}: violations fixed {viol_fixed.sum()} ({viol_fixed.sum()/valid.sum():.4%}), re-optimised {viol_re.sum()} ({viol_re.sum()/valid.sum():.4%}); "
          f"new violations {(viol_re & ~viol_fixed).sum()}; worst fixed {D_fixed[valid].min():.3f}, worst re-optimised {D_re[valid].min():.3f}")
    B = np.full((n, n, n), np.inf)                                 # the report's re-scoring bound on the first leg, [s, s', g]
    for sp in range(n):
        own = free_energy_of_policy(N[sp], pol[sp], Q[sp], beta)
        for g in range(n):
            if g == sp: continue
            B[:, sp, g] = free_energy_of_policy(N[sp], pol[sp], Q[g], beta) - own
    finite = np.isfinite(B) & valid
    ok_fixed = (-D_fixed <= B + 1e-6) | ~finite
    ok_re = (-D_re <= B + 1e-6) | ~finite
    print(f"  bound finite on {finite.sum()/valid.sum():.4f} of triples; holds for fixed second leg on {ok_fixed[valid].mean():.6f}, for re-optimised on {ok_re[valid].mean():.6f} "
          f"(fails on {(~ok_re & valid).sum()} triples; worst excess {np.max(np.where(finite, -D_re - B, -np.inf)):.4f})")
    np.savez_compressed(OUT / f"second-leg-reoptimised{"" if DET != 1 else "-det-1"}-b-{beta:g}.npz", F_direct=F, F_re=F_re, q_direct=Q, q_re=q_re, pol_dist=pol_dist, conv=conv, cert=cert)
    summary = dict(beta=beta, pairs=int(off.sum()), converged=float(conv[off].mean()), certified=float(cert[off].mean()),
                   saving=dict(min=float(saving[off].min()), median=float(np.median(saving[off])), mean=float(saving[off].mean()), max=float(saving[off].max()),
                               frac_gt_1e6=float((saving[off] > 1e-6).mean()), frac_gt_0_01=float((saving[off] > 0.01).mean()), frac_gt_0_1=float((saving[off] > 0.1).mean()),
                               rel_median=float(np.median(saving[off]/F[off])), rel_max=float((saving[off]/F[off]).max())),
                   policy_distance=dict(median=float(np.median(pol_dist[off])), max=float(pol_dist[off].max()), frac_gt_0_01=float((pol_dist[off] > 0.01).mean())),
                   prior_change=dict(median=float(np.median(dq[off])), max=float(dq[off].max())),
                   triples=int(valid.sum()), violations_fixed=int(viol_fixed.sum()), violations_reopt=int(viol_re.sum()), new_violations=int((viol_re & ~viol_fixed).sum()),
                   worst_fixed=float(D_fixed[valid].min()), worst_reopt=float(D_re[valid].min()),
                   bound_finite_frac=float(finite.sum()/valid.sum()), bound_holds_fixed=float(ok_fixed[valid].mean()), bound_holds_reopt=float(ok_re[valid].mean()), bound_fails_reopt=int((~ok_re & valid).sum()))
    with open(OUT / f"second-leg-reoptimised{"" if DET != 1 else "-det-1"}-b-{beta:g}.json", "w") as fh: json.dump(summary, fh, indent=1)
