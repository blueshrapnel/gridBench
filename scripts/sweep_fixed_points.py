#!/usr/bin/env python3
"""Fixed-point sweep with gridCore's nested solver, in the fieldInfo sweep.py file format.

For each (reference, beta) solves every available state as a goal with
gridcore.info.fixed_point.solve_self_consistent and writes

    <out>/<tag>-<reference>-b-<beta>.npz

with policies[g, s, a], q[g, a], F[s, g], steps[s, g], info[s, g] (bits), the three
certificate residuals and the iteration count per goal, plus converged[g] and
certified[g].  An existing file is moved to <file>.bak-<date> first.

Examples
    python sweep_fixed_points.py --env four_rooms --shape 13 --det 1.0 \
        --refs pooled uniform --betas 0.05 0.03 0.02 0.01 0.005 0.003 0.002 0.001 --procs 8
    python sweep_fixed_points.py --env four_rooms --shape 13 --det 0.97 --sigma path/to/x.sigma.npy \
        --tag four_rooms-13x13-det-0.97-evolved_highest_chi --refs pooled --betas 0.1 0.05

Set OMP_NUM_THREADS=1 / OPENBLAS_NUM_THREADS=1 (done below if unset): the solves are
small and BLAS threads only add overhead.  Needs gridcore on the path (the branch
with gridcore.info.fixed_point) and gridcore.bridge for the environments.
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, shutil, time
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from gridcore.bridge import build_env_by_id
from gridcore.info.fixed_point import solve_self_consistent, transition_tensor

T_GLOBAL = None


def _init(T):
    global T_GLOBAL
    T_GLOBAL = T


def _solve(args):
    g, beta, ref = args
    t = time.time()
    s = solve_self_consistent(T_GLOBAL, g, beta, ref)
    return g, s, time.time() - t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="four_rooms"); ap.add_argument("--shape", type=int, default=13)
    ap.add_argument("--det", type=float, default=1.0); ap.add_argument("--sigma", default=None, help=".npy twist (nS x nA), applied with set_sigma + twist_dynamics")
    ap.add_argument("--tag", default=None, help="file prefix; default <env>-<n>x<n>-det-<det>")
    ap.add_argument("--refs", nargs="+", default=["pooled"]); ap.add_argument("--betas", nargs="+", type=float, required=True)
    ap.add_argument("--out", default="/media/merlin/fixed-point-sweep"); ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()

    env = build_env_by_id(env_id=a.env, shape=(a.shape, a.shape), goal=0, determinism=a.det, manhattan=True)
    if a.sigma:
        env.set_sigma(np.load(a.sigma)); env.twist_dynamics()
    T, states = transition_tensor(env); n = len(states)
    tag = a.tag or f"{a.env}-{a.shape}x{a.shape}-det-{a.det:g}"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = out / f"{tag}-sweep-log.jsonl"
    print(f"{tag}: {n} states, refs {a.refs}, betas {a.betas}, {a.procs} processes", flush=True)
    with Pool(a.procs, initializer=_init, initargs=(T,)) as pool:
        for beta in a.betas:
            for ref in a.refs:
                t0 = time.time(); sols = [None] * n; secs = np.zeros(n)
                for g, s, dt in pool.imap_unordered(_solve, [(g, beta, ref) for g in range(n)], chunksize=1):
                    sols[g] = s; secs[g] = dt
                f = out / f"{tag}-{ref}-b-{beta:g}.npz"
                if f.exists():
                    shutil.move(f, f.with_name(f.name + f".bak-{date.today().isoformat()}"))
                np.savez_compressed(
                    f, states=states, policies=np.array([s.policy for s in sols]), q=np.array([s.prior for s in sols]),
                    F=np.column_stack([s.F for s in sols]), steps=np.column_stack([s.steps for s in sols]),
                    info=np.column_stack([s.info for s in sols]),
                    marginal_residual=np.array([s.marginal_residual for s in sols]), bellman_residual=np.array([s.bellman_residual for s in sols]),
                    ledger_residual=np.array([s.ledger_residual for s in sols]), iterations=np.array([s.iterations for s in sols]),
                    converged=np.array([s.converged for s in sols]), certified=np.array([s.certified() for s in sols]))
                rec = dict(tag=tag, reference=ref, beta=beta, goals=n, converged=int(sum(s.converged for s in sols)),
                           certified=int(sum(s.certified() for s in sols)), dropped_label_goals=int(sum((s.prior < 1e-6).any() for s in sols)),
                           mean_F=float(np.mean([s.F.mean() for s in sols])), max_iterations=int(max(s.iterations for s in sols)),
                           seconds=round(time.time() - t0, 1), slowest_goal_seconds=round(float(secs.max()), 1), file=str(f))
                with open(log, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(f"  {ref:9s} beta {beta:<7g} certified {rec['certified']}/{n} converged {rec['converged']}/{n} dropped-label goals {rec['dropped_label_goals']:3d} "
                      f"mean F {rec['mean_F']:8.2f} max it {rec['max_iterations']:6d}  {rec['seconds']:6.0f}s (slowest goal {rec['slowest_goal_seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
