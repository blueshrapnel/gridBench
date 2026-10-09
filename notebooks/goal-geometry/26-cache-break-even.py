# %% [markdown]
# # Goal-geometry series, 26 — the first test of the working-memory model
#
# Report §8 (8 Oct 2026): treat a stored policy as a cache line.  A two-leg journey loads
# one more policy than the direct route (at s') and must keep the leg label resident
# between decisions; it earns the co-information of leg, action and state less the
# averaged policy's gap, T[I(L;A) - I(L;A|S)]/beta - G, which grows with the journey's
# length while the load does not.  Prediction: a break-even length, about 6L decisions
# at beta 1 for a load of L bits (net earning 0.16 bits per decision at the median).
#
# For EVERY ordered triple (s, s', g) of the deterministic four rooms this computes the
# earnings in both currencies (the cache earnings above, and the report's own saving -dF),
# the journey length T = T1 + T2, the residency charge T H(L|S)/beta (perfect recall of the
# label; the piggyback bound of §8.1), and then, for a range of load costs L, the share of
# journeys still worth splitting against journey length, the break-even length at which half
# the positive-earning journeys stop paying, and the share of start-goal pairs with some
# interim goal still worth it.  Writes cache-break-even-b-<beta>.{json,npz} and the figure.

# %%
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, sys, time
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from gridcore.bridge import build_env_by_id
from gridcore.info.fixed_point import absorb, occupancy, transition_tensor

OUT = Path("/media/merlin/Dropbox/workbench/topics/04c-switching-costs/figure-code/second-leg-reoptimised")
FIG = Path("/media/merlin/Dropbox/workbench/topics/04c-switching-costs/figures")
env = build_env_by_id(env_id="four_rooms", shape=(13, 13), goal=0, determinism=1.0, manhattan=True)
T, _ = transition_tensor(env); n = len(T); idx = np.arange(n)
LOADS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
EPS = 1e-300

def klrows(p, q):                      # KL(p_s || q_s) over the last axis, rows of p and q
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p > 0, p * np.log2(np.maximum(p, EPS) / np.maximum(q, EPS)), 0.0).sum(-1)

results = {}
for beta in [float(b) for b in (sys.argv[1:] or ["1", "0.3"])]:
    d = np.load(f"/media/merlin/fixed-point-sweep/four_rooms-13x13-det-1-pooled-b-{beta:g}.npz")
    pol, Q, F = d["policies"], d["q"], d["F"]
    t0 = time.time()
    N = np.array([occupancy(np.einsum("sa,saj->sj", pol[g], absorb(T, g)), g) for g in range(n)])   # N[g, s, x]
    dF = F[:, :, None] + F[None, :, :] - F[:, None, :]                                                # [s, sp, g]
    E = np.full((n, n, n), np.nan); Tj = np.full((n, n, n), np.nan); H = np.full((n, n, n), np.nan); CI = np.full((n, n, n), np.nan)
    for sp in range(n):
        N1 = N[sp]; T1 = N1.sum(1); keep1 = T1 > 0                                                # rows s
        p1 = N1 @ pol[sp] / np.maximum(T1, EPS)[:, None]                                          # [s, a]
        for g in range(n):
            if g == sp: continue
            N2 = N[g, sp]; T2 = N2.sum(); p2 = N2 @ pol[g] / T2
            Tt = T1 + T2
            pbar = (T1[:, None] * p1 + T2 * p2[None]) / Tt[:, None]
            I_LA = (T1 * klrows(p1, pbar) + T2 * klrows(p2[None], pbar)) / Tt
            Ns = N1 + N2[None]                                                                     # [s, x]
            share1 = N1 / np.maximum(Ns, EPS)
            pibar = share1[:, :, None] * pol[sp][None] + (1 - share1)[:, :, None] * pol[g][None]   # [s, x, a]
            I_LAS = (N1 * klrows(pol[sp][None], pibar) + N2[None] * klrows(pol[g][None], pibar)).sum(1) / Tt
            with np.errstate(divide="ignore", invalid="ignore"):
                h = -(share1 * np.log2(np.maximum(share1, EPS)) + (1 - share1) * np.log2(np.maximum(1 - share1, EPS)))
            H_LS = (Ns * np.where(Ns > 0, h, 0.0)).sum(1) / Tt                                    # H(L|S) per decision
            C_dis = (N1 * klrows(pol[sp][None], pol[g][None])).sum(1)
            G = (C_dis - Tt * I_LAS) / beta
            E[:, sp, g] = Tt * (I_LA - I_LAS) / beta - G
            CI[:, sp, g] = Tt * (I_LA - I_LAS) / beta
            Tj[:, sp, g] = Tt; H[:, sp, g] = Tt * H_LS / beta
    valid = np.isfinite(E)
    for i in range(n): valid[i, i, :] = False; valid[i, :, i] = False; valid[:, i, i] = False
    S = -dF                                                                                          # the report's saving
    res = dict(beta=beta, seconds=round(time.time() - t0), triples=int(valid.sum()),
               earnings_vs_saving=dict(median_abs_diff=float(np.median(np.abs((E - S)[valid]))), corr=float(np.corrcoef(E[valid], S[valid])[0, 1])),
               positive_earnings_share=float((E[valid] > 1e-9).mean()), positive_saving_share=float((S[valid] > 1e-9).mean()),
               residency_median_on_positive=float(np.median(H[valid & (E > 1e-9)])), journey_length_median=float(np.median(Tj[valid])))
    bins = np.array([0, 4, 8, 12, 16, 20, 25, 30, 40, 60, 100])
    per_L = {}
    for cur_name, cur in (("cache", E), ("report", S)):
        for L in LOADS:
            row = {}
            for rho_name, rho in (("load_only", 0.0), ("load_and_residency", 1.0)):
                net = cur - L - rho * H
                worth = (net > 1e-9) & valid
                pos = (cur > 1e-9) & valid
                frac_len = []
                for a, b in zip(bins[:-1], bins[1:]):
                    m = pos & (Tj >= a) & (Tj < b)
                    frac_len.append(float(worth[m].mean()) if m.sum() >= 50 else None)
                # break-even length: the journey length at which half the positive-earning journeys still pay
                fl = [(np.sqrt(a * b), f) for (a, b), f in zip(zip(bins[:-1], bins[1:]), frac_len) if f is not None]
                be = None
                for (x0, f0), (x1, f1) in zip(fl[:-1], fl[1:]):
                    if f0 < 0.5 <= f1: be = float(x0 + (0.5 - f0) / (f1 - f0) * (x1 - x0)); break
                if be is None and fl and fl[0][1] >= 0.5: be = 0.0
                pairs = np.where(valid, net, -np.inf).max(1) > 1e-9                                 # [s, g]: some s' worth it
                off = ~np.eye(n, dtype=bool)
                row[rho_name] = dict(worth_share_of_triples=float(worth[valid].mean()), worth_share_of_positive=float(worth[pos].mean()) if pos.any() else None,
                                     pairs_with_some_interim=float(pairs[off].mean()), frac_by_length=frac_len, break_even_length=be)
            per_L[f"{L:g}"] = row
        res[f"by_load_{cur_name}"] = per_L; per_L = {}
    results[beta] = res
    np.savez_compressed(OUT / f"cache-break-even-b-{beta:g}.npz", earnings=E, saving=S, length=Tj, residency=H, coinfo=CI)
    json.dump(res, open(OUT / f"cache-break-even-b-{beta:g}.json", "w"), indent=1)
    print(f"beta {beta}: {res['seconds']}s; earnings vs saving corr {res['earnings_vs_saving']['corr']:.3f}; positive earnings {res['positive_earnings_share']:.3f} of triples, positive saving {res['positive_saving_share']:.4f}; residency median {res['residency_median_on_positive']:.2f}", flush=True)
    for L in LOADS:
        r = res["by_load_cache"][f"{L:g}"]; r2 = res["by_load_report"][f"{L:g}"]
        print(f"  L={L:<4g} cache: load-only worth {r['load_only']['worth_share_of_triples']:.4f} of triples, break-even {r['load_only']['break_even_length']}, pairs {r['load_only']['pairs_with_some_interim']:.3f} | +residency worth {r['load_and_residency']['worth_share_of_triples']:.4f}, break-even {r['load_and_residency']['break_even_length']}, pairs {r['load_and_residency']['pairs_with_some_interim']:.3f} || report: load-only worth {r2['load_only']['worth_share_of_triples']:.4f}, break-even {r2['load_only']['break_even_length']}, pairs {r2['load_only']['pairs_with_some_interim']:.3f}", flush=True)

# figure: share of positive-earning journeys still worth splitting against journey length, one line per load, beta 1 and 0.3
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(LOADS)))
for ax, (beta, res) in zip(axes, sorted(results.items(), reverse=True)):
    mids = np.sqrt(bins[:-1] * bins[1:])
    for L, c in zip(LOADS, cols):
        f = res["by_load_cache"][f"{L:g}"]["load_and_residency"]["frac_by_length"]
        xs = [m for m, v in zip(mids, f) if v is not None]; ys = [v for v in f if v is not None]
        ax.plot(xs, ys, "o-", color=c, lw=1.6, ms=3.5, label=f"load {L:g} bits")
        rate = float(np.median((np.load(OUT / f"cache-break-even-b-{beta:g}.npz")["earnings"] / np.load(OUT / f"cache-break-even-b-{beta:g}.npz")["length"])[
            np.isfinite(np.load(OUT / f"cache-break-even-b-{beta:g}.npz")["earnings"]) & (np.load(OUT / f"cache-break-even-b-{beta:g}.npz")["earnings"] > 1e-9)]))
        if L > 0: ax.axvline(L / rate, color=c, lw=0.8, ls=":")      # the ledger's prediction: L divided by the median earning per decision
    ax.set_xscale("log"); ax.set_xlim(2.5, 100); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("journey length T, expected decisions"); ax.set_title(f"β = {beta:g}: share of journeys still worth splitting", fontsize=10, loc="left")
    ax.grid(True, color="#e8e8e8", lw=0.6); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
axes[0].set_ylabel("share of positive-earning journeys"); axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")
fig.text(0.01, 0.005, "earnings: co-information less the averaged policy's gap; charges: one extra load plus residency T H(Λ|S)/β; dotted: the ledger's break-even prediction, the load divided by the median earning per decision of positive-earning journeys (0.07 at β = 1, 0.10 at β = 0.3)", fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(FIG / "cache-break-even.pdf"); fig.savefig(FIG / "cache-break-even.png", dpi=170); print("figure written")
