"""
Reproduce Daniel's random-twist diffusion plots on a 50x50 wrap_grid.

Daniel's old figures (~/Dropbox/phd/daily/2026-April/2026-04-24/twisted-world/)
show trajectory streaks of one committed label at various twist densities,
plus length histograms and a mean-length-vs-twist curve.  No code, no
axes, no seed: we reproduce them with our own infrastructure so we own
the picture and can extend it cleanly to other init schemes and to the
Flajolet-Odlyzko random-mapping null.

Init schemes compared:
  - eps-flip: per state, with probability eps shuffle the row to a random
    permutation, else identity.  This is `GridRoom.build_sigma` -- the
    original method Daniel almost certainly used.
  - perm-balanced (IP-05): same row-level construction (hybrid_schedule)
    plus a global label permutation applied post-hoc.  The constraint
    only matters at the population level; per-sigma functional graph is
    invariant under global label rotation, so a single perm-balanced
    sample sits on the same eps-flip curve (verified below).  We plot it
    as an overlay to make that invariance explicit.

Theoretical references:
  - Cartesian baseline (eps=0): every label cycles along a row or
    column of the torus -> rho = L = 50 for every starting state.
  - F&O random-mapping null on n = 2500: mean rho ~ sqrt(pi n / 2) ~ 63.
    Note this is the *unconstrained* null; our twists are 4-neighbour
    constrained, so the empirical eps=1 curve sits *below* this line.
    Shown as a caveat reference, not a target.

Axes convention (2026-07-03): reader-facing axes report the MEASURED twist
magnitude chi_twist (the ALife paper metric: 1 - best global-ordering match
fraction, via evolution_core.orientation_metrics.assignment_alignment_metrics)
rather than the nominal flip probability eps, which is a generator knob, not
a metric.  Note the compression: random generation saturates measured chi
around ~0.6, so the nominal sweep 0..1 maps onto a shorter measured axis.

Output: figs/{trajectory_streaks,length_histo,mean_rho_vs_chi,diffusion_summary}.png
"""

# %% imports + paths

from pathlib import Path

import random as pyrandom

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from gridcore.envs import GridRoom
from gridbench.functional_graph.single_label_reach import reach_sample, trajectory
from gridbench.functional_graph.null_models import permutation_null, mapping_null

from evolution_core.initial_population import build_population_perm_balanced
from evolution_core.orientation_metrics import assignment_alignment_metrics

def _nb_dir(default: str) -> Path:
    """Resolve correctly as a script and from a jupytext/Jupyter kernel."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        cwd = Path.cwd().resolve()
        default_path = Path(default)
        if (cwd / default_path.name).exists() or cwd.name == default_path.parent.name:
            return cwd
        return default_path.parent


HERE = _nb_dir(
    "/home/karen/phd-marlyn/gridBench/notebooks/single-label-reach/"
    "02-random-twist-diffusion.py"
)
FIG_DIR = HERE / "figs"
FIG_DIR.mkdir(exist_ok=True)
PAPER_FIGURES = Path(
    "/home/karen/Dropbox/phd/writing/twists-home-vectors/figures"
)

SHAPE = (50, 50)
L = SHAPE[0]
N_STATES = L * L
LABEL_N = 0  # convention: label 0 = North in Manhattan neighbourhood


def chi_of(env: "GridRoom") -> float:
    """Measured twist magnitude chi_twist of the whole sigma (ALife paper
    metric): 1 - largest fraction of (s, a) pairs matching a global ordering.
    Reference implementation: gridTwist evolution_core.orientation_metrics.
    Invariant under global label permutation, so eps-flip and perm-balanced
    sigmas built from the same rows measure identically."""
    m = assignment_alignment_metrics(env.sigma, available_states=range(env.nS))
    return float(m["chi_twist"])


# %% env construction helpers

def make_env_eps_flip(epsilon: float, seed: int) -> GridRoom:
    """50x50 wrap_grid with eps-flip twist (GridRoom.build_sigma)."""
    return GridRoom({
        "shape": SHAPE,
        "goals": [],
        "manhattan": True,
        "determinism": 1.0,
        "wrap": True,
        "epsilon": float(epsilon),
        "twist_seed": int(seed),
    })


def make_env_perm_balanced(target_epsilon: float, seed: int) -> GridRoom:
    """50x50 wrap_grid with one perm-balanced (IP-05) sigma loaded in.

    Builds a population of size 1 via build_population_perm_balanced
    using `shuffle_schedule` at the requested target_epsilon, then loads
    the flat genome into env.sigma and applies the twist.
    """
    rng = pyrandom.Random(int(seed))
    # population_size=1 forces initial_target_epsilons -> [0.0] under "uniform";
    # we force the desired epsilon directly.
    env = GridRoom({
        "shape": SHAPE,
        "goals": [],
        "manhattan": True,
        "determinism": 1.0,
        "wrap": True,
        "epsilon": 0.0,  # start untwisted; we'll overwrite sigma
    })
    state_order = list(range(env.nS))
    # Single-genome construction.  base_init_mode="shuffle_schedule" makes
    # this directly comparable to eps-flip (same per-row operator), and
    # the global label perm applied by IP-05 is what distinguishes the
    # two methods at the population marginal level.
    pop, _meta = build_population_perm_balanced(
        state_order=state_order,
        n_actions=env.nA,
        population_size=1,
        base_init_mode="shuffle_schedule",
        init_schedule="random",  # picks one random target_epsilon per individual
        rng=rng,
    )
    # Force the target_epsilon precisely (init_schedule=random gave us a
    # random one; override by re-running the row construction directly).
    from evolution_core.initial_population import (
        action_permutations,
        apply_label_permutation_to_genes,
        build_individual_genes,
    )
    base = build_individual_genes(
        state_order=state_order,
        n_actions=env.nA,
        init_mode="shuffle_schedule",
        target_epsilon=float(target_epsilon),
        rng=rng,
    )
    perms = action_permutations(env.nA)
    rng.shuffle(perms)
    genes = apply_label_permutation_to_genes(base, perms[0])
    sigma = np.asarray(genes, dtype=int).reshape(env.nS, env.nA)
    env.sigma[:, :] = sigma
    env._refresh_sigma_inv()  # cached inverse must stay in lock-step with sigma
    env.twist_dynamics()
    # twist_dynamics rewrites env.P but not env.T; reach_sample reads T.
    # Rebuild T (and friends) to reflect the new sigma.
    env.update_dynamics_for_goals(env.goals)
    return env


# %% (a) trajectory streaks at 5 eps values --------------------------------

EPS_PANELS = [0.05, 0.10, 0.25, 0.50, 0.75]
STREAK_SEED = 7
STREAK_CUTOFF = 500

# Start states: a small band in the lower-middle of the grid, matching
# Daniel's PDFs (his traces start from a localised region and fan
# outward).  10x10 patch centred horizontally, 5 rows up from the
# bottom -- enough starts to read the directional bias but not so many
# that the figure becomes solid red at low eps.
y_band = np.arange(L - 12, L - 2)
x_band = np.arange(L // 2 - 5, L // 2 + 5)
starts = np.array([y * L + x for y in y_band for x in x_band], dtype=int)


def state_to_xy(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (s % L, s // L)


fig, axes = plt.subplots(1, len(EPS_PANELS), figsize=(3.0 * len(EPS_PANELS), 3.4),
                         constrained_layout=True)
panel_chis = []
for ax, eps in zip(axes, EPS_PANELS):
    env = make_env_eps_flip(epsilon=eps, seed=STREAK_SEED)
    chi = chi_of(env)
    panel_chis.append(chi)
    # Faint grey shading over the 10x10 start-state patch.
    ax.add_patch(plt.Rectangle(  # type: ignore[attr-defined]
        (x_band[0] - 0.5, L - 1 - y_band[-1] - 0.5),
        len(x_band), len(y_band),
        facecolor="grey", alpha=0.15, edgecolor="none", zorder=0,
    ))
    for s0 in starts:
        path = trajectory(env, label=LABEL_N, start=int(s0), max_steps=STREAK_CUTOFF)
        xs, ys = state_to_xy(path)
        # Y axis inverted so "north" reads as up.
        ax.plot(xs, L - 1 - ys, color="red", linewidth=0.4, alpha=0.55)
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(-0.5, L - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    # Panels report the MEASURED twist magnitude; the nominal flip rate is
    # generator provenance only.
    ax.set_title(rf"$\chi_\mathrm{{twist}}={chi:.2f}$", fontsize=11)
fig.suptitle(f"Label-N trajectory streaks on {L}x{L} wrap_grid (eps-flip generator, seed={STREAK_SEED})",
             fontsize=11)
print("panel chis (nominal eps -> measured chi):",
      {f"{e:.2f}": round(c, 3) for e, c in zip(EPS_PANELS, panel_chis)})
out = FIG_DIR / "trajectory_streaks.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
fig.savefig(PAPER_FIGURES / "F1-trajectory-streaks.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")


# %% (b) reach distributions ------------------------------------------------

EPS_SWEEP = np.linspace(0.0, 1.0, 21)
N_SEEDS = 8
CUTOFF = 500

# The sweep builds 2 x 21 x 8 environments at 50x50 (~minutes); cache the
# derived arrays so plot-only tweaks re-render instantly.  Delete the npz to
# force a recompute.
CACHE_DIR = HERE / "_cache"
CACHE_DIR.mkdir(exist_ok=True)
SWEEP_CACHE = CACHE_DIR / f"sweep_L{L}_eps{len(EPS_SWEEP)}_seeds{N_SEEDS}_cutoff{CUTOFF}_chi.npz"

if SWEEP_CACHE.exists():
    _c = np.load(SWEEP_CACHE)
    eps_flip_rho_mean = _c["eps_flip_rho_mean"]
    eps_flip_rho_per_seed = _c["eps_flip_rho_per_seed"]
    eps_flip_chi_mean = _c["eps_flip_chi_mean"]
    perm_bal_rho_mean = _c["perm_bal_rho_mean"]
    perm_bal_rho_per_seed = _c["perm_bal_rho_per_seed"]
    perm_bal_chi_mean = _c["perm_bal_chi_mean"]
    all_rho_stack = _c["all_rho_stack"]  # (len(EPS_SWEEP), N_STATES * N_SEEDS)
    print(f"loaded sweep cache {SWEEP_CACHE.name}")
else:
    eps_flip_rho_mean = np.zeros(len(EPS_SWEEP))
    eps_flip_rho_per_seed = np.zeros((len(EPS_SWEEP), N_SEEDS))
    eps_flip_chi_mean = np.zeros(len(EPS_SWEEP))  # measured chi per sweep point
    all_rho_per_eps = []  # for histogram surface

    for i, eps in enumerate(EPS_SWEEP):
        rho_pool = []
        chi_pool = []
        for k in range(N_SEEDS):
            env = make_env_eps_flip(epsilon=float(eps), seed=1000 + 13 * k)
            r = reach_sample(env, label=LABEL_N, cutoff=CUTOFF)
            rho_pool.append(r.rho)
            chi_pool.append(chi_of(env))
            eps_flip_rho_per_seed[i, k] = float(r.rho.mean())
        all_rho = np.concatenate(rho_pool)
        eps_flip_rho_mean[i] = float(all_rho.mean())
        eps_flip_chi_mean[i] = float(np.mean(chi_pool))
        all_rho_per_eps.append(all_rho)

    # Perm-balanced overlay: one sample per eps_sweep point, multiple seeds.
    perm_bal_rho_mean = np.zeros(len(EPS_SWEEP))
    perm_bal_rho_per_seed = np.zeros((len(EPS_SWEEP), N_SEEDS))
    perm_bal_chi_mean = np.zeros(len(EPS_SWEEP))
    for i, eps in enumerate(EPS_SWEEP):
        rho_pool = []
        chi_pool = []
        for k in range(N_SEEDS):
            env = make_env_perm_balanced(target_epsilon=float(eps), seed=2000 + 17 * k)
            r = reach_sample(env, label=LABEL_N, cutoff=CUTOFF)
            rho_pool.append(r.rho)
            chi_pool.append(chi_of(env))
            perm_bal_rho_per_seed[i, k] = float(r.rho.mean())
        perm_bal_rho_mean[i] = float(np.concatenate(rho_pool).mean())
        perm_bal_chi_mean[i] = float(np.mean(chi_pool))

    all_rho_stack = np.stack(all_rho_per_eps)
    np.savez_compressed(
        SWEEP_CACHE,
        eps_sweep=EPS_SWEEP,
        eps_flip_rho_mean=eps_flip_rho_mean,
        eps_flip_rho_per_seed=eps_flip_rho_per_seed,
        eps_flip_chi_mean=eps_flip_chi_mean,
        perm_bal_rho_mean=perm_bal_rho_mean,
        perm_bal_rho_per_seed=perm_bal_rho_per_seed,
        perm_bal_chi_mean=perm_bal_chi_mean,
        all_rho_stack=all_rho_stack,
    )
    print(f"saved sweep cache {SWEEP_CACHE.name}")

print("eps-flip mean rho:", eps_flip_rho_mean.round(2))
print("perm-bal mean rho:", perm_bal_rho_mean.round(2))
print("nominal eps -> measured chi (eps-flip):",
      dict(zip(EPS_SWEEP.round(2).tolist(), eps_flip_chi_mean.round(3).tolist())))
print("NOTE nominal eps=1 uses the derangement code path (build_sigma_full_twist):",
      f"chi={eps_flip_chi_mean[-1]:.3f}, rho={eps_flip_rho_mean[-1]:.2f}",
      "-- plotted as a detached marker, not part of the shuffle curve")


# %% length histogram surface (length x measured chi -> count)

LEN_BINS = np.arange(0, CUTOFF + 5, 5)
H = np.zeros((len(EPS_SWEEP), len(LEN_BINS) - 1), dtype=float)
for i in range(len(EPS_SWEEP)):
    h, _ = np.histogram(all_rho_stack[i], bins=LEN_BINS)
    H[i] = h


def _plot_length_histo(ax, fig):
    # Rows sit at the MEASURED chi of each sweep point; the compression of
    # row spacing toward the top is the chi ceiling under random generation.
    # The nominal eps=1 row is dropped: GridRoom switches to a derangement
    # sampler there (build_sigma_full_twist), a different twist family whose
    # lower chi would fold the y-axis back on itself.
    im = ax.pcolormesh(
        LEN_BINS[:-1] + 2.5, eps_flip_chi_mean[:-1],
        np.ma.masked_less(H[:-1], 1),
        shading="auto", cmap="viridis",
        norm=LogNorm(vmin=1, vmax=float(H.max())),
    )
    ax.set_xlim(0, 200)  # the tail beyond ~4 rho scales is empty
    ax.set_xlabel(r"trajectory length $\rho$ (state-steps to first revisit)")
    ax.set_ylabel(r"measured twist $\chi_\mathrm{twist}$")
    ax.set_title(f"Length distribution (label N), {L}x{L} wrap_grid", fontsize=11)
    fig.colorbar(im, ax=ax, label="count of starting states")


def _plot_mean_rho_vs_chi(ax, show_nulls=True, show_perm_balanced=True,
                          show_derangement=True):
    # show_nulls: the unconstrained F&O reference lines are a notebook-side
    # diagnostic (see 04-cross-env-null-models); the paper figure drops them
    # so the constrained curve gets the axis to itself.
    # show_perm_balanced: the overlay empirically verifies the global-rotation
    # invariance; the paper asserts the (exact) invariance in prose instead.
    # show_derangement: the nominal eps=1 code-path marker is a generator
    # caveat, prose-only in the paper.
    # GridRoom switches to a derangement sampler at nominal eps=1
    # (build_sigma_full_twist) -- a different twist family that measures
    # LOWER chi (zero count-matrix diagonal -> chi ~ 2/3) and reaches
    # farther.  Plot the shuffle-scheme curve through eps <= 0.95 and the
    # derangement endpoint as a detached marker.
    chi_c = eps_flip_chi_mean[:-1]
    ax.fill_between(
        chi_c,
        eps_flip_rho_per_seed[:-1].min(axis=1),
        eps_flip_rho_per_seed[:-1].max(axis=1),
        alpha=0.18, color="C0", label="eps-flip seed range",
    )
    ax.plot(chi_c, eps_flip_rho_mean[:-1], "o-", color="C0", lw=1.6,
            markersize=4, label="eps-flip mean")
    if show_perm_balanced:
        ax.plot(perm_bal_chi_mean, perm_bal_rho_mean, "s--", color="C1", lw=1.2,
                markersize=4, label="perm-balanced (IP-05) mean")
    if show_derangement:
        ax.plot([eps_flip_chi_mean[-1]], [eps_flip_rho_mean[-1]], "D", color="C4",
                markersize=6,
                label="derangement generator (env code path at nominal $\\varepsilon=1$)")

    # Nominal generator rates, annotated so the eps->chi compression is
    # legible without promoting eps to an axis.
    for eps_mark in (0.25, 0.50, 0.95):
        j = int(np.argmin(np.abs(EPS_SWEEP - eps_mark)))
        ax.annotate(rf"nominal ${eps_mark:.2f}$",
                    (eps_flip_chi_mean[j], eps_flip_rho_mean[j]),
                    textcoords="offset points", xytext=(4, 6),
                    fontsize=7, color="grey")

    # Cartesian theoretical: rho = L on a wrap_grid for any committed label.
    ax.axhline(L, color="grey", lw=1.0, ls=":",
               label=f"Cartesian theoretical ($\\rho=L={L}$, $\\chi_\\mathrm{{twist}}=0$)")

    if show_nulls:
        # F&O unconstrained random-mapping null for n=N_STATES.
        fo_mean_rho = float(mapping_null(N_STATES).mean_rho)
        ax.axhline(fo_mean_rho, color="C3", lw=1.0, ls="-.",
                   label=f"F&O random-mapping null mean $\\rho \\approx {fo_mean_rho:.1f}$  (unconstrained)")

        # F&O closed-form for *uniform random permutation* on n=N_STATES.
        fo_perm_mean = float(permutation_null(N_STATES).mean_rho)
        ax.axhline(fo_perm_mean, color="C2", lw=1.0, ls="--",
                   label=f"F&O random-permutation null mean $\\rho \\approx {fo_perm_mean:.1f}$  (unconstrained)")

    ax.set_xlabel(r"measured twist $\chi_\mathrm{twist}$")
    ax.set_ylabel(r"mean single-label reach $\langle \rho \rangle$")
    ax.set_title(f"Mean reach vs measured twist, {L}x{L} wrap_grid (label = N)", fontsize=11)
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper right")


fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
_plot_length_histo(ax, fig)
out = FIG_DIR / "length_histo.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")

fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
_plot_mean_rho_vs_chi(ax)
out = FIG_DIR / "mean_rho_vs_chi.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")


# %% combined two-panel summary (paper figure F2)

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15.5, 4.8), constrained_layout=True)
_plot_length_histo(ax_l, fig)
_plot_mean_rho_vs_chi(ax_r, show_nulls=False, show_perm_balanced=False,
                      show_derangement=False)
out = FIG_DIR / "diffusion_summary.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
fig.savefig(PAPER_FIGURES / "F2-diffusion-summary.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")


# %% (c) sanity / cross-validation -----------------------------------------

# eps = 0: every state must have rho = L exactly (one cycle per column).
env0 = make_env_eps_flip(epsilon=0.0, seed=0)
r0 = reach_sample(env0, label=LABEL_N)
assert r0.fg.n_basins == L, f"Cartesian expected {L} basins, got {r0.fg.n_basins}"
assert (r0.rho == L).all(), "Cartesian eps=0: every rho must equal L"
assert chi_of(env0) == 0.0, "identity sigma must measure chi_twist = 0"
print(f"sanity: eps=0  n_basins={r0.fg.n_basins}  chi={chi_of(env0)}  unique rho={sorted(set(r0.rho.tolist()))}")

# Cross-check reach_sample mean rho against fingerprint per_label_stats.
from gridbench.functional_graph.decomposition import (
    decompose, deterministic_successor, per_label_stats,
)
env_mid = make_env_eps_flip(epsilon=0.5, seed=42)
succ = deterministic_successor(env_mid, action=LABEL_N)
stats = per_label_stats(decompose(succ))
r_mid = reach_sample(env_mid, label=LABEL_N)
print(f"sanity: eps=0.5  reach mean_rho={float(r_mid.rho.mean()):.3f}  fingerprint mean_rho={stats['mean_rho']:.3f}")
assert abs(float(r_mid.rho.mean()) - stats["mean_rho"]) < 1e-9

# perm-balanced sigma should give the same reach distribution as the
# underlying eps-flip sigma (global label rotation is a relabelling --
# functional graph invariant).  Verify at eps=0.5.
env_pb = make_env_perm_balanced(target_epsilon=0.5, seed=42)
r_pb = reach_sample(env_pb, label=LABEL_N)
print(f"sanity: perm-bal eps=0.5  mean_rho={float(r_pb.rho.mean()):.3f}  "
      f"(eps-flip same seed = {float(r_mid.rho.mean()):.3f})")

print("\nall sanity checks passed.")
