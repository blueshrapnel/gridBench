"""00-habit-cycle-intro.py

Emit F0 of the attractor-reach paper: a row of four habit-cycle panels
for one action label across four (env, sigma) combinations:

    [four_rooms 7x7 Cartesian, four_rooms 7x7 GA-best,
     wrap_grid 7x7 Cartesian, wrap_grid 7x7 GA-best]

Each panel shows the deterministic functional graph induced by repeated
single-label dynamics: states coloured by their terminal cycle (basin),
arrows along the deterministic successor map, hatched cycle cells.

The renderer is borrowed from the conference paper notebook
`gridFour/notebooks/alife-relabelled-actions/figure-6.ipynb` cell 8, so
panels are visually consistent with the conference paper figures.

Run with
    PYENV_VERSION=py-3.12-grid python notebooks/basin-typology/01-habit-cycle-intro.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.patches import Rectangle as _Rect

from gridbench import store
from gridbench.functional_graph.probe_env import build_goal_free_probe_env
from gridbench.papers.home_vectors import WALL_COLOUR
from gridvis import display
from gridvis.display_twist import (
    _action_color_indices,
    _action_labels,
    _select_action_colours,
)

mpl.rcParams['mathtext.fontset'] = 'cm'

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


NB_DIR = _nb_dir(
    "/home/karen/phd-marlyn/gridBench/notebooks/basin-typology/"
    "01-habit-cycle-intro.py"
)
FIG_DIR = NB_DIR / 'figs'
FIG_DIR.mkdir(exist_ok=True)
PAPER_FIGURE = Path(
    "/home/karen/Dropbox/phd/writing/twists-home-vectors/figures/"
    "F0-habit-cycle-intro.png"
)
PAPER_REFS_PATH = NB_DIR.parent / "relabelling-actions" / "paper-refs.json"

# Match the rest of the paper's label convention.
LABEL_NAME = 'N'  # 'N', 'E', 'S', 'W' — change to pick a different label.


# ---------------------------------------------------------------------------
# Env construction (mirrors figure-6.ipynb cell 4's build_twisted_env_no_goal)
# ---------------------------------------------------------------------------

def build_env(env_id: str, sigma: np.ndarray | None):
    """Unconditioned env (goals=[]) with optional sigma overlay.

    sigma=None leaves the env at the identity twist (Cartesian baseline).
    """
    env = build_goal_free_probe_env(env_id, (7, 7), 0.97)
    if sigma is not None:
        available = np.asarray(env.available_states, dtype=int)
        env.sigma[available] = sigma[available]
    env.twist_dynamics()
    env.update_dynamics_for_goals(env.goals)
    return env


def load_ga_best_sigma(env_id: str) -> np.ndarray:
    """Load the GA-best sigma for `env_id` from paper-refs.json."""
    refs = json.loads(PAPER_REFS_PATH.read_text())
    spec = refs[env_id]
    keys = (
        "init_method", "fitness_objective", "env_id", "shape", "beta", "det",
        "run_name",
    )
    run_dir = store.run_dir(**{key: spec[key] for key in keys})
    summary_path = next(run_dir.glob("*-multi-all.summary.json"))
    summary = json.loads(summary_path.read_text())
    sigma_hash = summary["best_sigma_hash"]
    sigma_path = next(run_dir.glob(f"hive_sigma/**/sigma_id={sigma_hash}/sigma.npy"))
    return np.load(sigma_path)


# ---------------------------------------------------------------------------
# Functional graph decomposition (lifted from figure-6.ipynb cell 8)
# ---------------------------------------------------------------------------

def _deterministic_successor(env, action: int) -> np.ndarray:
    T = np.asarray(env.T)
    succ = np.zeros(env.nS, dtype=int)
    for s in range(env.nS):
        row = T[s * env.nA + action]
        succ[s] = int(np.argmax(row))
    return succ


def _find_basins(succ: np.ndarray) -> dict:
    n = len(succ)
    visited = np.full(n, -1, dtype=int)
    basin_id = np.full(n, -1, dtype=int)
    cycles: list[list[int]] = []
    current_basin = 0
    for start in range(n):
        if visited[start] >= 0:
            continue
        path: list[int] = []
        s = start
        while visited[s] < 0:
            visited[s] = start
            path.append(s)
            s = succ[s]
        if visited[s] == start:
            cycle_start_idx = path.index(s)
            cycle = path[cycle_start_idx:]
            cycles.append(cycle)
            bid = current_basin
            current_basin += 1
            for c in cycle:
                basin_id[c] = bid
            for c in path[:cycle_start_idx]:
                basin_id[c] = bid
        else:
            bid = basin_id[s]
            for c in path:
                basin_id[c] = bid
    basin_sizes = [int(np.sum(basin_id == b)) for b in range(current_basin)]
    return {'cycles': cycles, 'basin_id': basin_id, 'basin_sizes': basin_sizes}


# ---------------------------------------------------------------------------
# Single-panel renderer
# ---------------------------------------------------------------------------

def render_panel(ax, env, label_idx: int, title: str) -> None:
    action_labels = _action_labels(env)
    cids, ps = _action_color_indices(action_labels)
    action_colors = _select_action_colours(ps)

    walls = set(getattr(env, 'walls_flat', []))
    available = [s for s in range(env.nS) if s not in walls]

    succ = _deterministic_successor(env, label_idx)
    result = _find_basins(succ)
    # Cycles / basins that contain only wall states are display artefacts;
    # filter them out for the basin count and the cycle hatching.
    real_cycles = [c for c in result['cycles']
                   if any(int(s) not in walls for s in c)]
    n_basins = len(real_cycles)
    cycle_states = {int(s) for c in real_cycles for s in c if int(s) not in walls}
    label_colour = action_colors[cids[label_idx]]
    h_grid, w_grid = env.shape

    # Mask walls so they don't take a basin colour.
    basin_grid = result['basin_id'].astype(float).reshape(env.shape)
    if walls:
        wall_mask = np.isin(np.arange(env.nS), sorted(walls)).reshape(env.shape)
        basin_grid = np.ma.array(basin_grid, mask=wall_mask)
    cmap = mpl.colormaps.get_cmap('Set2').copy()
    cmap.set_bad(color=WALL_COLOUR)
    ax.imshow(basin_grid, cmap=cmap,
              vmin=-0.5, vmax=max(n_basins - 0.5, 0.5),
              origin='upper', interpolation='nearest')

    # Draw arrows and cycle hatching only on non-wall cells.
    for s in available:
        r, c = divmod(s, w_grid)
        s_next = int(succ[s])
        if s_next in walls:
            continue
        r2, c2 = divmod(s_next, w_grid)
        dr = r2 - r
        dc = c2 - c
        if abs(dr) > 1:
            dr = -int(np.sign(dr))
        if abs(dc) > 1:
            dc = -int(np.sign(dc))
        if s in cycle_states:
            ax.add_patch(_Rect(
                (c - 0.5, r - 0.5), 1.0, 1.0,
                facecolor='none', edgecolor=label_colour,
                linewidth=0.0, hatch='///', alpha=0.7, zorder=2,
            ))
        if dr == 0 and dc == 0:
            ax.plot(c, r, 'o', color='red', markersize=3, zorder=4)
        else:
            ax.annotate('', xy=(c + dc * 0.4, r + dr * 0.4),
                        xytext=(c - dc * 0.1, r - dr * 0.1),
                        arrowprops=dict(arrowstyle='->', color='black',
                                        lw=0.9, mutation_scale=7),
                        zorder=3)

    # Overlay an opaque, shared wall fill so walls remain environmental
    # features rather than blending with the basin map.
    for state in walls:
        row, column = divmod(int(state), w_grid)
        ax.add_patch(_Rect(
            (column - 0.5, row - 0.5), 1.0, 1.0,
            facecolor=WALL_COLOUR, edgecolor='none', zorder=2.5,
        ))

    state_ids = np.arange(env.nS).reshape(env.shape)
    display.add_label_to_plot(
        ax, state_ids,
        shift=-0.35, size='5', env=env, skip_goals=False,
        format_fn=lambda v: str(int(v)),
    )

    ax.set_title(f'{title}\nlabel {action_labels[label_idx]} · {n_basins} basins',
                 fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, w_grid - 0.5)
    ax.set_ylim(h_grid - 0.5, -0.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    panels = [
        ('four_rooms', None,                              'four-rooms 7×7 · Cartesian'),
        ('four_rooms', load_ga_best_sigma('four_rooms'),  'four-rooms 7×7 · GA-best'),
        ('wrap_grid',  None,                              'wrap-grid 7×7 · Cartesian'),
        ('wrap_grid',  load_ga_best_sigma('wrap_grid'),   'wrap-grid 7×7 · GA-best'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)

    ref_env = build_env('four_rooms', sigma=None)
    action_labels = _action_labels(ref_env)
    label_idx = list(action_labels).index(LABEL_NAME)

    for ax, (env_id, sigma, title) in zip(axes, panels):
        env = build_env(env_id, sigma=sigma)
        render_panel(ax, env, label_idx=label_idx, title=title)

    fig.suptitle(
        f'Habit cycles for label {LABEL_NAME} on two worlds, Cartesian vs GA-best',
        fontsize=11,
    )
    out = FIG_DIR / '00_habit_cycle_intro.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(PAPER_FIGURE, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}')
    print(f'saved {PAPER_FIGURE}')


if __name__ == '__main__':
    main()
