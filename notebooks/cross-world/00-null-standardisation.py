# %% [markdown]
# # cross-world 00 — Null-ensemble standardisation constants
#
# Plan: daily note 2026-07-17, "Null-ensemble standardisation for the
# cross-world twist".  Draw 2,016 twists per ensemble ONCE on the full
# 49-cell lattice (perm_balanced as 21 complete 96-member populations
# preserving the epsilon schedule + rotation balance; shuffle as
# independent uniform rows), project the same matrices into every world
# (wall rows inert), and per (twist, world) compute the per-pair mean
# free energy over all goals, plus the basic fingerprint for validation
# against the twist-generation v2 cache.
#
# Outputs (figs/ sibling "data" dir):
#   draws_{ensemble}_n{N}_seed{SEED}.npz        one stack per ensemble
#   null_standardisation_{tag}.parquet          per (ensemble, draw, env)
#   constants_{tag}.json                        frozen mu/s per env (full run)
#
# Usage:  --sample 24 --workers 8   (test)      | full: no --sample,
#         --workers 10 (desktop etiquette: not all 20)

# %%
import argparse
import json
import random
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/media/merlin/phd-marlyn/gridTwist/src")


def _nb_dir(default: str) -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        cwd = Path.cwd().resolve()
        return cwd if (cwd / Path(default).name).exists() \
            or cwd.name == Path(default).parent.name else Path(default).parent


NB_DIR = _nb_dir("/media/merlin/phd-marlyn/gridBench/notebooks/cross-world"
                 "/00-null-standardisation.py")
DATA_DIR = NB_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ENVS = ["wrap_grid", "open_grid", "helical", "pinwheel", "four_rooms", "pillar_3"]
SHAPE, DET, BETA = (7, 7), 0.97, 1.0
NS = SHAPE[0] * SHAPE[1]
DEFAULT_SEED = 20260718
N_FULL = 2016  # 21 x 96


# %% Draws: once, on the full lattice
def draw_ensembles(n: int, master_seed: int):
    from evolution_core.initial_population import (
        build_individual_genes, build_population_perm_balanced,
    )
    state_order = list(range(NS))

    def genes_to_sigma(genes):
        return np.asarray(genes, dtype=int).reshape(NS, 4)

    rng = random.Random(master_seed)
    pb = []
    while len(pb) < n:
        pop, _ = build_population_perm_balanced(
            state_order=state_order, n_actions=4, population_size=96,
            base_init_mode="hybrid_schedule", init_schedule="uniform",
            init_derangement_prob=0.5, init_derangement_power=1.0,
            dedupe=True, rng=rng)
        pb.extend(pop)
    pb = np.stack([genes_to_sigma(g) for g in pb[:n]])

    rng2 = random.Random(master_seed + 1)
    sh = np.stack([genes_to_sigma(build_individual_genes(
        state_order=state_order, n_actions=4, init_mode="shuffle",
        target_epsilon=None, init_derangement_prob=0.5,
        init_derangement_power=1.0, rng=rng2)) for _ in range(n)])
    return {"perm_balanced": pb, "shuffle": sh}


# %% Per-(twist, world) evaluation
_worker_env = {}


def _env_pack(env_id):
    if env_id not in _worker_env:
        from gridbench.functional_graph.decomposition import (
            decompose, deterministic_successor, per_label_stats,
        )
        from gridbench.functional_graph.probe_env import build_goal_free_probe_env
        env = build_goal_free_probe_env(env_id, SHAPE, DET)
        wf = getattr(env, "walls_flat", None)
        walls = set(int(w) for w in np.ravel(wf)) if wf is not None else set()
        walk = [s for s in range(NS) if s not in walls]
        base = np.stack([deterministic_successor(env, a) for a in range(4)],
                        axis=0)
        _worker_env[env_id] = (walls, walk, base,
                               decompose, per_label_stats)
    return _worker_env[env_id]


def eval_one(args):
    ensemble, draw_id, env_id, sigma_bytes = args
    from gridcore.bridge import (EvalConfig, _state_dist_class,
                                 build_twisted_env_from_sigma)
    from gridcore.info import DecisionInformation as GC_DI
    walls, walk, base, decompose, per_label_stats = _env_pack(env_id)
    sigma = np.frombuffer(sigma_bytes, dtype=np.int64).reshape(NS, 4)
    tot = 0.0
    max_iters = 0
    max_residual = 0.0
    for g in walk:
        cfg = EvalConfig(env_id=env_id, shape=SHAPE, goal=int(g), beta=BETA,
                         determinism=DET, manhattan=True, theta=1e-5,
                         state_dist="uniform")
        e = build_twisted_env_from_sigma(sigma, cfg)
        di = GC_DI(e, _state_dist_class("uniform")(e), 1e-5,
                   max_iterations=200_000, max_info_iterations=10_000)
        _, _, F = di.get_opt_policy_Z_free_vector(1.0)
        # 2026-08-02 review: a null constant built on a silently unconverged
        # solve would poison every downstream z.  Fail loudly instead.
        if not bool(getattr(di, "converged", True)):
            raise RuntimeError(
                f"free-energy solve did not converge: ensemble={ensemble} "
                f"draw={draw_id} env={env_id} goal={g} "
                f"iterations={getattr(di, 'iteration_count', -1)}"
            )
        max_iters = max(max_iters, int(getattr(di, "iteration_count", 0)))
        max_residual = max(
            max_residual, float(getattr(di, "last_blahut_residual", 0.0))
        )
        tot += float(np.asarray(F, dtype=float)[walk].mean())
    sigma_inv = np.argsort(sigma, axis=1)
    idx = np.arange(NS)
    nb, cbr, cov = [], [], []
    for l in range(4):
        fg = decompose(base[sigma_inv[:, l], idx], walls=walls)
        st = per_label_stats(fg)
        nb.append(st["n_basins"])
        cbr.append(st["cycle_basin_ratio"])
        sizes = np.asarray(fg.basin_sizes, dtype=int)
        cov.append(sizes.max() / len(walk) if sizes.size else 0.0)
    return {"ensemble": ensemble, "draw_id": draw_id, "env_id": env_id,
            "mean_free": tot / len(walk),
            "max_blahut_iterations": int(max_iters),
            "max_blahut_residual": float(max_residual),
            "fp_n_basins": float(np.mean(nb)),
            "fp_cycle_basin_ratio": float(np.mean(cbr)),
            "fp_largest_basin_fraction": float(max(cov))}


# %% Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None,
                    help="evaluate only the first N draws per ensemble")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = ap.parse_args()
    n = a.sample or N_FULL
    tag = f"sample{n}" if a.sample else f"full{n}"
    out_dir = NB_DIR / "results" / f"seed{a.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    stacks = draw_ensembles(N_FULL, a.seed)
    for name, arr in stacks.items():
        f = out_dir / f"draws_{name}_n{N_FULL}_seed{a.seed}.npz"
        if not f.exists():
            np.savez_compressed(f, sigmas=arr)
    cart = np.tile(np.arange(4), (NS, 1))

    tasks = [("cartesian", -1, e, cart.astype(np.int64).tobytes())
             for e in ENVS]
    for name, arr in stacks.items():
        for i in range(n):
            for e in ENVS:
                tasks.append((name, i, e, arr[i].astype(np.int64).tobytes()))
    print(f"{len(tasks)} (twist, world) evaluations, {a.workers} workers",
          flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, r in enumerate(ex.map(eval_one, tasks, chunksize=4)):
            rows.append(r)
            if (k + 1) % 120 == 0:
                print(f"  {k + 1}/{len(tasks)}", flush=True)
    df = pd.DataFrame(rows)
    out = out_dir / f"null_standardisation_{tag}.parquet"
    df.to_parquet(out)
    print(f"wrote {out}")

    # constants + Cartesian z preview
    con = {}
    for e in ENVS:
        d = df[(df.env_id == e) & (df.ensemble == "shuffle")].mean_free
        mu = float(d.median())
        s = float(1.4826 * (d - d.median()).abs().median())
        fc = float(df[(df.env_id == e) & (df.ensemble == "cartesian")]
                   .mean_free.iloc[0])
        pb = df[(df.env_id == e) & (df.ensemble == "perm_balanced")].mean_free
        con[e] = {"mu": mu, "s": s, "n": int(len(d)),
                  "cartesian_mean_free": fc,
                  "cartesian_z": (fc - mu) / s if s else float("nan"),
                  "perm_balanced_median": float(pb.median()),
                  "ensemble": "shuffle", "seed": a.seed,
                  "method": "median/1.4826*MAD"}
        print(f"{e:12} mu={mu:8.3f}  s={s:6.3f}  "
              f"Cart={fc:8.3f}  z(Cart)={con[e]['cartesian_z']:+7.2f}  "
              f"pbal med={con[e]['perm_balanced_median']:8.3f}", flush=True)
    cf = out_dir / f"constants_{tag}.json"
    cf.write_text(json.dumps(con, indent=1))
    print(f"wrote {cf}" + ("" if a.sample else "  (FROZEN)"))


if __name__ == "__main__":
    main()
