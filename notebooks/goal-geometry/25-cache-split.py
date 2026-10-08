# %% [markdown]
# # Goal-geometry series, 25 — the working-memory split of a violation
#
# For every violating triple (s, s', g) of the four rooms (deterministic, pooled reference), the
# three-term account of the saving from holding two policies instead of one merged one:
#     -dF = T [I(L;A) - I(L;A|S)] / beta  -  G  +  conv / beta
# with T I(L;A) the saving from two caches over one merged marginal (report eq. saving),
# T I(L;A|S) the disambiguation cost of the leg label (eq. disambiguation), G the averaged
# policy's optimality gap F^{pibar}_g(s)[q_g] - F_g(s) (from eq. disagree-split), and
# conv = T KL(pbar || q_g) - sum_i T_i KL(p_i || q_{g_i}) the conversion between journey
# marginals and goal priors (eq. cswitch-saving).  The identity assumes no early arrival; the
# residual against the actual defect is reported.  Writes cache-split-b-<beta>.json/npz.

# %%
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, sys, time
from pathlib import Path
import numpy as np
from gridcore.bridge import build_env_by_id
from gridcore.info.fixed_point import absorb, occupancy, transition_tensor, kl_rows

OUT = Path("/media/merlin/Dropbox/workbench/topics/switching-costs/figure-code/second-leg-reoptimised")
env = build_env_by_id(env_id="four_rooms", shape=(13, 13), goal=0, determinism=1.0, manhattan=True)
T, _ = transition_tensor(env); n = len(T)

def kl(p, q):
    m = p > 0; return float((p[m] * np.log2(p[m] / q[m])).sum())

for beta in [float(b) for b in (sys.argv[1:] or ["1", "0.3"])]:
    d = np.load(f"/media/merlin/fixed-point-sweep/four_rooms-13x13-det-1-pooled-b-{beta:g}.npz")
    pol, Q, F = d["policies"], d["q"], d["F"]                       # pol[g, s, a], Q[g], F[s, g]
    t0 = time.time()
    N = np.array([occupancy(np.einsum("sa,saj->sj", pol[g], absorb(T, g)), g) for g in range(n)])     # N[g, s, x]
    KLg = np.array([kl_rows(pol[g], Q[g]) for g in range(n)])         # per-state KL to own prior, [g, x]
    dF = F[:, :, None] + F[None, :, :] - F[:, None, :]                 # [s, sp, g]
    valid = np.ones((n, n, n), bool)
    for i in range(n): valid[i, i, :] = False; valid[i, :, i] = False; valid[:, i, i] = False
    viol = np.argwhere((dF < -1e-9) & valid)
    rows = []
    for (s, sp, g) in viol:
        N1 = N[sp, s]; N2 = N[g, sp]; T1, T2 = N1.sum(), N2.sum(); Tt = T1 + T2
        p1 = N1 @ pol[sp] / T1; p2 = N2 @ pol[g] / T2; pbar = (T1 * p1 + T2 * p2) / Tt
        I_LA = (T1 * kl(p1, pbar) + T2 * kl(p2, pbar)) / Tt                                   # I(L;A)
        Ns = N1 + N2; vis = Ns > 0
        pibar = np.where(vis[:, None], (N1[:, None] * pol[sp] + N2[:, None] * pol[g]) / np.maximum(Ns, 1e-300)[:, None], pol[g])
        I_LAS = (np.where(vis, N1 * kl_rows(pol[sp], pibar) + N2 * kl_rows(pol[g], pibar), 0.0)).sum() / Tt   # I(L;A|S)
        C_dis = (N1 * kl_rows(pol[sp], pol[g])).sum()                                           # C_disagree
        G = (C_dis - Tt * I_LAS) / beta                                                         # averaged policy's gap, by eq. disagree-split
        conv = Tt * kl(pbar, Q[g]) - T1 * kl(p1, Q[sp]) - T2 * kl(p2, Q[g])
        C_res = (N1 * (kl_rows(pol[sp], Q[g]) - KLg[sp])).sum()                                 # C_rescoring, direct
        rows.append((s, sp, g, -dF[s, sp, g], Tt * I_LA / beta, Tt * I_LAS / beta, G, conv / beta, C_res / beta, (C_res - C_dis) / beta, T1, T2))
    R = np.array(rows); V, S1, S2, G, CV, CR, EX, T1, T2 = R[:, 3], R[:, 4], R[:, 5], R[:, 6], R[:, 7], R[:, 8], R[:, 9], R[:, 10], R[:, 11]
    pred = S1 - S2 - G + CV
    q = lambda x: [float(v) for v in np.percentile(x, [10, 50, 90])]
    out = dict(beta=beta, violators=int(len(R)), seconds=round(time.time() - t0),
               identity_vs_defect=dict(max_abs=float(np.abs(pred - V).max()), median_abs=float(np.median(np.abs(pred - V))),
                                       exact_defect_identity_max_abs=float(np.abs(EX - V).max())),
               cswitch_saving_identity_max_abs=float(np.abs(CR - (Tt_dummy := 0) - (S1 - 0) * beta / 1.0 - 0).max()) if False else None,
               terms_pct=dict(two_caches_saving=q(S1), disambiguation=q(S2), co_information=q(S1 - S2), averaged_policy_gap=q(G), conversion=q(CV), violation=q(V)),
               shares_of_violation_median=dict(co_information=float(np.median((S1 - S2) / V)), minus_gap=float(np.median(-G / V)), conversion=float(np.median(CV / V))),
               co_information_exceeds_violation=float(np.mean(S1 - S2 >= V - 1e-9)),
               co_information_positive=float(np.mean(S1 - S2 > 0)),
               journey_length_median=float(np.median(T1 + T2)),
               break_even=dict(net_per_decision_median=float(np.median((S1 - S2 - G) / (T1 + T2))), net_per_decision_pct=q((S1 - S2 - G) / (T1 + T2))))
    json.dump(out, open(OUT / f"cache-split-b-{beta:g}.json", "w"), indent=1)
    np.savez_compressed(OUT / f"cache-split-b-{beta:g}.npz", rows=R)
    print(json.dumps(out), flush=True)
