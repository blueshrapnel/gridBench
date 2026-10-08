# %% [markdown]
# # Goal-geometry series, 24 — the commitment ladder and the two-leg bound, in one pass
#
# For one world and beta, from the stored direct solutions (goal policies at the pooled prior):
#   1. for every (s', g): the fixed-prior optimum of goal s' at g's prior q_g, giving the
#      shared-prior slack and the first-leg prior advantage A1 per triple, and the
#      re-scoring bound C1 = F^{pi_{s'}}(s)[q_g] - F^{pi_{s'}}(s)[q_{s'}] (report §4–5);
#   2. for every (s', g): the goal policy re-solved for starts at s' alone (point-mass
#      reference), giving the second-leg advantage A2 = F_g(s') - F_{g|s'}(s'), the prior and
#      policy change, and the second-leg re-scoring bound C2 = T2 KL(q_{g|s'} || q_g)/beta;
#   3. per triple: defects under the commitment ladder (shared prior; pooled per goal; second
#      leg re-planned; a prior per journey), violation counts, and the tightness of C1,
#      C1 - slack, C1 + C2 and C1 + C2 - slack against the matching violation.
# Writes <out>/ladder-<tag>-b-<beta>.json and .npz.  DET=1 is the report's own four rooms.
#
# Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DET=1 PYTHONPATH=<gridCore fixed-point>/src python 24-ladder-and-bounds.py 1 0.3

# %%
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from gridcore.bridge import build_env_by_id
from gridcore.info.fixed_point import (absorb, occupancy, solve_self_consistent, solve_fixed_prior, transition_tensor,
                                       free_energy_of_policy, is_deterministic)

OUT = Path(os.environ.get("OUT", "/media/merlin/Dropbox/workbench/topics/switching-costs/figure-code/second-leg-reoptimised"))
DET = float(os.environ.get("DET", "1")); SHAPE = int(os.environ.get("SHAPE", "13"))
TAG = f"four_rooms-{SHAPE}x{SHAPE}-det-{DET:g}"
SWEEP = Path("/media/merlin/fixed-point-sweep")
BETAS = [float(b) for b in (sys.argv[1:] or ["1", "0.3"])]

env = build_env_by_id(env_id="four_rooms", shape=(SHAPE, SHAPE), goal=0, determinism=DET, manhattan=True)
T, states = transition_tensor(env); n = len(T); DETERMINISTIC = is_deterministic(T)
T_G = POL = Q = None

def _init(T, pol, Q):
    global T_G, POL, Q_G; T_G, POL, Q_G = T, pol, Q

def kl(p, q):
    m = p > 0; return float((p[m] * np.log2(p[m] / q[m])).sum())

def _pair(args):
    sp, g, beta = args
    # (1) goal s' at g's prior: shared-prior first leg
    fq = solve_fixed_prior(absorb(T_G, sp), sp, Q_G[g], beta, warm_policy=POL[sp], deterministic=DETERMINISTIC)
    Fq = fq.F                                                        # [s]: min_pi F_{s'}^pi(s)[q_g]
    # (2) goal g re-solved for starts at s'
    w = np.zeros(n); w[sp] = 1.0
    s = solve_self_consistent(T_G, g, beta, w, q0=Q_G[g])
    P = np.einsum("sa,saj->sj", s.policy, absorb(T_G, g)); P[~np.isfinite(s.F)] = 0.0
    N = occupancy(P, g); T2 = N[sp].sum()
    v = N[sp] / T2
    return (sp, g, Fq, s.F[sp], s.prior, kl(s.prior, Q_G[g]) * T2 / beta, T2, float(v @ np.abs(s.policy - POL[g]).sum(1)),
            bool(s.converged), bool(s.certified()))

for beta in BETAS:
    f = SWEEP / (f"{TAG}-pooled-b-{beta:g}.npz" if DET == 1 else f"{TAG}-untwisted-pooled-b-{beta:g}.npz")
    d = np.load(f); pol, Q, F = d["policies"], d["q"], d["F"]            # pol[g, s, a], Q[g, a], F[s, g]
    Ndir = np.array([occupancy(np.einsum("sa,saj->sj", pol[g], absorb(T, g)), g) for g in range(n)])
    t0 = time.time()
    Fq = np.full((n, n, n), np.nan)        # [sp, s, g]
    F_re = np.full((n, n), np.nan); q_re = np.full((n, n, 4), np.nan); C2 = np.full((n, n), np.nan); T2 = np.full((n, n), np.nan)
    pol_dist = np.full((n, n), np.nan); conv = np.zeros((n, n), bool); cert = np.zeros((n, n), bool)
    with Pool(8, initializer=_init, initargs=(T, pol, Q)) as pool:
        for k, (sp, g, fq, fr, qr, c2, t2, pd, cv, ce) in enumerate(pool.imap_unordered(_pair, [(sp, g, beta) for g in range(n) for sp in range(n) if sp != g], chunksize=16)):
            Fq[sp, :, g] = fq; F_re[sp, g] = fr; q_re[sp, g] = qr; C2[sp, g] = c2; T2[sp, g] = t2; pol_dist[sp, g] = pd; conv[sp, g] = cv; cert[sp, g] = ce
            if k % 5000 == 0: print(f"  beta {beta}: {k}/{n*(n-1)} ({time.time()-t0:.0f}s)", flush=True)
    off = ~np.eye(n, dtype=bool)
    # per-triple quantities, indexed [sp, s, g]
    Fs_sp = F.T[:, :, None]                       # F[s, sp] -> [sp, s, 1]
    Fsp_g = F[:, None, :]                         # F[sp, g] -> [sp, 1, g]
    Fs_g = F[None, :, :]                          # F[s, g]  -> [1, s, g]
    dF = Fs_sp + Fsp_g - Fs_g                                            # report's defect
    slack = Fq + Fsp_g - Fs_g
    A1 = Fq - Fs_sp
    A2 = np.maximum(F - F_re, 0.0)[:, None, :]                           # [sp, 1, g]
    C1 = np.full((n, n, n), np.inf)
    for sp in range(n):
        own = free_energy_of_policy(Ndir[sp], pol[sp], Q[sp], beta)
        for g in range(n):
            if g != sp: C1[sp, :, g] = free_energy_of_policy(Ndir[sp], pol[sp], Q[g], beta) - own
    C2b = C2[:, None, :]
    valid = np.isfinite(slack) & np.isfinite(dF)
    for i in range(n): valid[i, i, :] = False; valid[i, :, i] = False; valid[:, i, i] = False
    A1p = A1 + (F - F_re).T[:, :, None]                                 # first leg re-planned for its start
    d_leg2 = dF - A2
    Fr_T = F_re.T[:, :, None]; Fr_spg = F_re[:, None, :]; Fr_sg = F_re[None, :, :]
    d_all = Fr_T + Fr_spg - Fr_sg
    def rate(D): return float((D[valid] < -1e-9).mean())
    def pairs(D):
        Dm = np.where(valid, D, np.nan); best = np.nanmin(Dm, 0); return float((best[off] < -1e-9).mean())
    def tight(B, D):
        v = (D < -1e-9) & valid; r = B[v] / (-D[v]); fin = np.isfinite(r)
        big = -D[v] > np.percentile(-D[v], 90)
        return dict(violators=int(v.sum()), finite_bound=float(fin.mean()), covered=float(((B[v] >= -D[v] - 1e-6) | ~fin).mean()),
                    ratio_pct=[float(x) for x in np.percentile(r[fin], [10, 50, 90])], within_10pct=float((r[fin] <= 1.1).mean()), within_2x=float((r[fin] <= 2).mean()),
                    ratio_median_largest_10pct=float(np.median(r[fin & big])) if (fin & big).any() else None)
    sav = (F - F_re)[off]
    out = dict(tag=TAG, beta=beta, triples=int(valid.sum()), pairs=int(off.sum()), converged=float(conv[off].mean()), certified=float(cert[off].mean()),
               identity_residual=float(np.abs(dF - (slack - A1))[valid].max()),
               slack_min=float(slack[valid].min()), A1_negative_frac=float((A1[valid] < -1e-9).mean()), A1_median=float(np.median(A1[valid])),
               A1p_negative_frac=float((A1p[valid] < -1e-6).mean()),
               A2=dict(median=float(np.median(sav)), mean=float(sav.mean()), max=float(sav.max()), min=float(sav.min()), rel_median=float(np.median(sav / F[off])), rel_max=float((sav / F[off]).max()),
                       frac_gt_0_1=float((sav > 0.1).mean()), negative_pairs=int((sav < -1e-6).sum())),
               A2_gt_A1_frac=float((np.broadcast_to(A2, (n, n, n)) > A1)[valid].mean()),
               policy_distance=dict(median=float(np.median(pol_dist[off])), max=float(pol_dist[off].max()), frac_gt_0_01=float((pol_dist[off] > 0.01).mean())),
               prior_change_median=float(np.median(np.abs(q_re - Q[None]).max(2)[off])),
               A2_le_C2_frac=float((((F - F_re) <= C2 + 1e-6)[off]).mean()), C2_over_A2_median=float(np.median((C2 / np.maximum(F - F_re, 1e-9))[off])),
               ladder=dict(shared_prior=dict(triples=rate(slack), pairs=pairs(slack)), pooled_per_goal=dict(triples=rate(dF), pairs=pairs(dF)),
                           second_leg_replanned=dict(triples=rate(d_leg2), pairs=pairs(d_leg2)), prior_per_journey=dict(triples=rate(d_all), pairs=pairs(d_all))),
               worst=dict(pooled=float(dF[valid].min()), second_leg=float(d_leg2[valid].min()), per_journey=float(d_all[valid].min())),
               tightness=dict(C1_vs_fixed=tight(C1, dF), C1_minus_slack_vs_fixed=tight(C1 - slack, dF),
                              C1C2_vs_replanned=tight(C1 + C2b, d_leg2), C1C2_minus_slack_vs_replanned=tight(C1 + C2b - slack, d_leg2)),
               C1_over_A1_median_on_violators=float(np.median((C1 / A1)[(dF < -1e-9) & valid & (A1 > 0)])),
               seconds=round(time.time() - t0))
    json.dump(out, open(OUT / f"ladder-{TAG}-b-{beta:g}.json", "w"), indent=1)
    np.savez_compressed(OUT / f"ladder-{TAG}-b-{beta:g}.npz", F=F, F_re=F_re, q=Q, q_re=q_re, C2=C2, T2=T2, pol_dist=pol_dist, conv=conv, cert=cert, Fq=Fq)
    print(json.dumps({k: out[k] for k in ("beta", "converged", "certified", "ladder", "A2", "tightness")}, indent=None)[:3000], flush=True)
