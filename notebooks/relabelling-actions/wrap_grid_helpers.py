"""
Shared helpers for wrap_grid analysis notebooks.

Loads config, builds twisted environments, computes per-action occupancy
distributions, and solves per-goal optimal policies.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gridcore.envs import GridRoom
from gridcore.info.decision_information import DecisionInformation
from gridcore.planning.policy import Policy
from gridcore.planning.state_distribution import (
    StateDistribution,
    UniformStateDistribution,
    get_stationary_distribution_eigen_decomposition,
)
from figure_support import build_env
from gridbench import store


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path = "wrap_grid_config.json") -> dict:
    """Load the wrap_grid run config and return as dict."""
    with open(config_path) as f:
        return json.load(f)


def load_sigma(cfg: dict) -> np.ndarray:
    """Load the best sigma array from the run directory."""
    run_dir = Path(cfg["run_dir"])
    sigma_path = run_dir / cfg["best_sigma_path"]
    return np.load(sigma_path)


def footer_text(cfg: dict) -> str:
    """Return a standard footer annotation string."""
    return f"$\\sigma$ = {cfg['best_sigma_hash']}    run = {cfg['run_name']}"


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------

def build_twisted_env(
    cfg: dict,
    sigma: np.ndarray,
    goal: int = 0,
) -> GridRoom:
    """Build a wrap_grid with the GA-optimised sigma applied."""
    env = build_env(
        cfg["env_id"],
        shape=tuple(cfg["shape"]),
        goal=goal,
        determinism=cfg["determinism"],
    )
    available = np.asarray(env.available_states, dtype=int)
    env.sigma[available] = sigma[available]
    env.twist_dynamics()
    env.update_dynamics_for_goals(env.goals)
    return env


def build_untwisted_env(
    cfg: dict,
    goal: int = 0,
) -> GridRoom:
    """Build an untwisted wrap_grid (Cartesian baseline)."""
    return build_env(
        cfg["env_id"],
        shape=tuple(cfg["shape"]),
        goal=goal,
        determinism=cfg["determinism"],
    )


def build_twisted_env_no_goal(
    cfg: dict,
    sigma: np.ndarray,
) -> GridRoom:
    """Build a wrap_grid with the twist applied but no absorbing goal.

    Use this for analyses of the per-label dynamics that should not depend
    on which goal is conditioned on — e.g. functional-graph decomposition,
    per-action stationary distributions, basin analysis.

    Implementation: passes ``goals=[]`` to GridRoom so no state is made
    absorbing. ``check_goals([])`` is vacuously true; the for-loop in
    ``update_dynamics_for_goals`` over ``self.goals`` is empty, so the
    transition kernel is built without any goal-state modifications.
    """
    from gridcore.envs import GridRoom

    walls = _walls_for_wrap_grid(cfg)
    options = {
        "shape": tuple(cfg["shape"]),
        "goals": [],
        "manhattan": True,
        "determinism": float(cfg["determinism"]),
        "epsilon": 0.0,
        "twist_seed": 0,
        "wrap": True,
    }
    if walls is not None:
        options["walls"] = walls
    env = GridRoom(options)

    available = np.asarray(env.available_states, dtype=int)
    env.sigma[available] = sigma[available]
    env.twist_dynamics()
    env.update_dynamics_for_goals(env.goals)  # goals=[] so no absorption
    return env


def _walls_for_wrap_grid(cfg: dict):
    """wrap_grid has no internal walls, so return None."""
    return None


# ---------------------------------------------------------------------------
# Per-action occupancy d^a(s)
# ---------------------------------------------------------------------------

# DATA_ROOT-relative (schema-11 moved derived caches under <store>/cache/),
# so this resolves against the live store or a frozen bundle's local data.
STATIONARY_CACHE_DIR = store.data_root() / "cache" / "per_action_stationary"


def compute_per_action_stationary(
    env: GridRoom,
    sigma_hash: str | None = None,
) -> dict[int, np.ndarray]:
    """Compute or load cached stationary distribution for each single-action policy.

    The cache key includes the sigma hash AND a goal-condition tag derived
    from ``env.goals``. This prevents silent contamination if the same sigma
    is evaluated on both goal-conditioned and unconditioned environments —
    the per-action stationary distribution depends on both. For per-label
    dynamics analysis that should be goal-independent, pass an unconditioned
    env (built via ``build_twisted_env_no_goal``).

    Returns dict mapping action index to d^a(s) array.
    """
    goal_tag = _goal_cache_tag(env)
    if sigma_hash is not None:
        cache_path = STATIONARY_CACHE_DIR / f"stationary-{sigma_hash[:16]}-{goal_tag}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            return {a: cached[f"action_{a}"] for a in range(env.nA)}

    stationary = {}
    for a in range(env.nA):
        pi = np.zeros((env.nS, env.nA))
        pi[:, a] = 1.0
        stationary[a] = get_stationary_distribution_eigen_decomposition(env, pi)

    if sigma_hash is not None:
        STATIONARY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            STATIONARY_CACHE_DIR / f"stationary-{sigma_hash[:16]}-{goal_tag}.npz",
            **{f"action_{a}": stationary[a] for a in range(env.nA)},
        )

    return stationary


def _goal_cache_tag(env: GridRoom) -> str:
    """Return a short tag describing the env's goal configuration.

    Used as part of the cache key for per-action dynamics so that
    goal-conditioned and unconditioned envs don't share cache entries.
    """
    goals = list(getattr(env, "goals", []) or [])
    if not goals:
        return "nogoal"
    # Stable tag: sorted goal list joined with underscores.
    return "goal" + "_".join(str(int(g)) for g in sorted(goals))


def compute_per_action_transition_matrices(env: GridRoom) -> dict[int, np.ndarray]:
    """Build per-action transition matrices (with teleporting) for convergence plots."""
    matrices = {}
    for a in range(env.nA):
        pi = np.zeros((env.nS, env.nA))
        pi[:, a] = 1.0
        P = StateDistribution.build_transition_matrix(env, pi)
        P = StateDistribution.add_teleporting_to_transition_matrix(env, P)
        matrices[a] = P
    return matrices


# ---------------------------------------------------------------------------
# Functional graph decomposition (basins & cycles)
#
# The canonical implementation lives in
# ``analysis.functional_graph.decomposition``.  The functions below are
# thin shims that preserve the legacy dict-returning signature used by
# the existing notebooks; new call sites should prefer ``decompose()``
# directly, which returns a richer ``FunctionalGraph`` dataclass.
# ---------------------------------------------------------------------------

from gridbench.functional_graph.decomposition import (
    decompose as _decompose,
    deterministic_successor as _deterministic_successor,
)


def deterministic_successor(env: GridRoom, action: int) -> np.ndarray:
    """For each state, return the most-probable successor under the given action."""
    return _deterministic_successor(env, action)


def find_basins(succ: np.ndarray) -> dict:
    """Decompose the functional graph into basins (legacy dict API).

    Returns dict with:
      'cycles': list of lists (each cycle is a list of states)
      'basin_id': array mapping each state to its basin index
      'basin_sizes': list of basin sizes

    For tail-length, rho, terminal nodes, in-degree, or diameter use
    ``analysis.functional_graph.decomposition.decompose`` directly.
    """
    fg = _decompose(succ)
    return {
        "cycles": fg.cycles,
        "basin_id": fg.basin_id,
        "basin_sizes": fg.basin_sizes,
    }


# ---------------------------------------------------------------------------
# Hive cache: load pre-computed per-goal data from goal-X.npz
# ---------------------------------------------------------------------------

def _best_sigma_hive_dir(cfg: dict) -> Path:
    """Return the hive directory for the best sigma."""
    return Path(cfg["run_dir"]) / cfg["best_sigma_path"].rsplit("/sigma.npy", 1)[0]


def load_goal_from_hive(cfg: dict, goal: int) -> dict:
    """Load pre-computed policy/info/free/value for one goal from the hive.

    The hive stores goal-X.npz with keys: policy, info, free, value, state_dist.
    """
    hive_dir = _best_sigma_hive_dir(cfg)
    npz_path = hive_dir / f"goal-{goal}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Hive cache not found: {npz_path}")
    data = np.load(npz_path)
    return {
        "policy": data["policy"],
        "information": data["info"],
        "free_energy": data["free"],
        "value": data["value"],
        "state_dist": data["state_dist"],
    }


def load_goal(cfg: dict, sigma: np.ndarray, goal: int) -> dict:
    """Load per-goal data from hive cache, building the env for plotting.

    Returns dict with keys: policy, information, free_energy, value, env.
    """
    cached = load_goal_from_hive(cfg, goal)
    env = build_twisted_env(cfg, sigma, goal=goal)
    cached["env"] = env
    return cached


def load_all_goals(cfg: dict) -> dict[str, np.ndarray]:
    """Load all per-goal data from the hive into (nS, nS) matrices.

    Returns dict with keys: frees, infos, values, policies.
    """
    shape = tuple(cfg["shape"])
    nS = shape[0] * shape[1]
    nA = 4  # Manhattan

    frees = np.zeros((nS, nS), dtype=float)
    infos = np.zeros((nS, nS), dtype=float)
    values = np.zeros((nS, nS), dtype=float)
    policies = np.zeros((nS, nS, nA), dtype=float)

    hive_dir = _best_sigma_hive_dir(cfg)
    for g in range(nS):
        data = np.load(hive_dir / f"goal-{g}.npz")
        frees[g, :] = data["free"]
        infos[g, :] = data["info"]
        values[g, :] = data["value"]
        policies[g, :, :] = data["policy"]

    return {"frees": frees, "infos": infos, "values": values, "policies": policies}


# ---------------------------------------------------------------------------
# Per-goal optimal policy (recompute fallback for untwisted baseline)
# ---------------------------------------------------------------------------

BASELINE_GOALS_CACHE_DIR = store.data_root() / "cache" / "baseline_goals"


def _baseline_goal_cache_path(cfg: dict, goal: int, beta: float = 1.0) -> Path:
    """Deterministic cache path for an untwisted per-goal solve.

    Key: (env_id, shape, determinism, beta, goal).
    Stored alongside the existing frontier_baseline caches.
    """
    import hashlib
    shape = tuple(cfg["shape"])
    key = f"{cfg['env_id']}-{shape[0]}x{shape[1]}-det{cfg['determinism']}-beta{beta}-goal{goal}"
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    return BASELINE_GOALS_CACHE_DIR / f"baseline_goal-{cfg['env_id']}-{shape[0]}x{shape[1]}-{h}.npz"


def solve_goal_untwisted(
    cfg: dict,
    goal: int,
    beta: float = 1.0,
    theta: float = 1e-5,
    max_iterations: int = 100_000,
    max_info_iterations: int = 20_000,
) -> dict:
    """Load or compute the DI-optimal policy for a single goal without twist.

    Caches per-goal results as npz files under the schema-10 reports cache
    so that any run with the same (env_id, shape, det, beta) can reuse them.
    """
    cache_path = _baseline_goal_cache_path(cfg, goal, beta)

    if cache_path.exists():
        data = np.load(cache_path)
        env = build_untwisted_env(cfg, goal=goal)
        return {
            "policy": data["policy"],
            "information": data["info"],
            "free_energy": data["free"],
            "value": data["value"],
            "env": env,
        }

    # recompute
    env = build_untwisted_env(cfg, goal=goal)
    state_dist = UniformStateDistribution(env)
    di = DecisionInformation(
        env, state_dist,
        theta=float(theta),
        max_iterations=int(max_iterations),
        max_info_iterations=int(max_info_iterations),
    )
    policy, _, free = di.get_opt_policy_Z_free_vector(beta=float(beta))
    information = di.get_decision_information_given_policy(policy).astype(float)
    value, _ = Policy.evaluate_policy(env, policy)

    result = {
        "policy": np.asarray(policy, dtype=float),
        "information": information.astype(float),
        "free_energy": np.asarray(free, dtype=float),
        "value": value.astype(float),
        "env": env,
    }

    # cache for future use
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        policy=result["policy"],
        info=result["information"],
        free=result["free_energy"],
        value=result["value"],
    )
    return result


def dominant_action_grid(policy: np.ndarray, env: GridRoom) -> np.ndarray:
    """Return (h, w) grid of dominant action index at each state.

    Wall/goal states get -1.
    """
    dom = np.argmax(policy, axis=1)
    walls = set(getattr(env, "walls_flat", []))
    for s in walls:
        dom[s] = -1
    return dom.reshape(env.shape)


# ---------------------------------------------------------------------------
# Free energy matrix (from hive cache) and MDS
# ---------------------------------------------------------------------------

def load_free_energy_matrix(cfg: dict) -> np.ndarray:
    """Load the (nS, nS) free energy matrix from hive cache."""
    return load_all_goals(cfg)["frees"]


def symmetrise(matrix: np.ndarray) -> np.ndarray:
    """Symmetrise a matrix: (M + M.T) / 2."""
    return (matrix + matrix.T) / 2.0


def load_baseline_reference(cfg: dict) -> dict:
    """Load the untwisted (identity sigma) baseline reference from the run.

    Returns the aggregate baseline data from
    frames/analysis/global_alignment_baseline_reference.json, which includes
    expected_di, expected_free, expected_value, sigma_hash, etc.

    For per-goal per-state arrays (policy, info, free, value), use
    solve_goal_untwisted() which recomputes from the identity sigma.
    """
    ref_path = Path(cfg["run_dir"]) / "frames" / "analysis" / "global_alignment_baseline_reference.json"
    if not ref_path.exists():
        raise FileNotFoundError(f"Baseline reference not found: {ref_path}")
    with open(ref_path) as f:
        return json.load(f)


def mds_html_path(cfg: dict) -> Path | None:
    """Return the path to the interactive MDS HTML if it exists.

    Generated by batch_collated_mds.py via the shallow reports pipeline.
    """
    run_dir = Path(cfg["run_dir"])
    sigma_plots = run_dir / "frames" / "analysis"
    # look for the interactive HTML generated by batch_collated_mds
    candidates = list(sigma_plots.glob("*frees*3d*mds*.html"))
    if not candidates:
        # also check sigma-plots/ directory
        candidates = list(run_dir.glob("sigma-plots/*frees*3d*mds*.html"))
    return candidates[0] if candidates else None


def build_interactive_mds(
    frees: np.ndarray,
    env,
    *,
    title: str = "MDS of symmetrised free energy",
    metadata: dict[str, str] | None = None,
    out_html: Path | str | None = None,
):
    """Build a report-style interactive 3D MDS plotly Figure.

    Uses the same rendering code as batch_collated_mds.py (sphere nodes,
    grey edges, orthographic camera, state-map inset).

    Returns the plotly Figure for inline display via fig.show() in notebooks.
    Optionally writes HTML to *out_html*.
    """
    from experiments.MDS_plots.batch_collated_mds import (
        _active_states,
        _build_compact_state_map_traces,
        _build_metadata_text,
        _build_undirected_edges,
        _make_sphere_trace,
        _node_colours_hex,
    )
    from utility.geometry_plots import (
        calculate_symmetric_adjacency_matrix,
        get_matrix_embedding,
    )
    import plotly.graph_objects as go

    sym = calculate_symmetric_adjacency_matrix(frees)
    active_states = _active_states(env, sym.shape[0])
    sym = sym[np.ix_(active_states, active_states)]
    coords = get_matrix_embedding(sym, state=1, components=3)
    active_set = set(active_states)
    state_to_idx = {s: i for i, s in enumerate(active_states)}
    edges = {
        (i, j)
        for s1, s2 in _build_undirected_edges(env)
        if s1 in active_set and s2 in active_set
        for i, j in [(state_to_idx[s1], state_to_idx[s2])]
    }
    env_id = metadata.get("env_id") if metadata else None
    all_node_colours = _node_colours_hex(env, env_id=env_id)
    node_colours = [all_node_colours[s] for s in active_states]

    edge_x, edge_y, edge_z = [], [], []
    for i, j in edges:
        edge_x.extend([coords[i, 0], coords[j, 0], None])
        edge_y.extend([coords[i, 1], coords[j, 1], None])
        edge_z.extend([coords[i, 2], coords[j, 2], None])

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = max(float((maxs - mins).max()), 1e-6)
    fill_radius = 0.028 * span
    highlight_radius = 0.009 * span
    centre = coords.mean(axis=0)

    traces = []
    label_x, label_y, label_z, label_text = [], [], [], []

    for idx, state in enumerate(active_states):
        x, y, z = (float(coords[idx, k]) for k in range(3))
        traces.append(_make_sphere_trace(x, y, z, fill_radius, fill=node_colours[idx], opacity=1.0))
        traces.append(_make_sphere_trace(
            x - 0.010 * span, y + 0.010 * span, z + 0.010 * span,
            highlight_radius, fill="#ffffff", opacity=0.95,
        ))
        direction = np.asarray([x, y, z], dtype=float) - centre
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            direction = np.asarray([0.0, 0.0, 1.0])
            norm = 1.0
        direction /= norm
        offset = 1.45 * fill_radius
        label_x.append(x + offset * float(direction[0]))
        label_y.append(y + offset * float(direction[1]))
        label_z.append(z + offset * float(direction[2]))
        label_text.append(str(state))

    traces.append(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, mode="lines",
        line=dict(color="rgba(100,100,100,0.82)", width=7),
        hoverinfo="skip", showlegend=False,
    ))

    scene_annotations = [
        dict(x=label_x[i], y=label_y[i], z=label_z[i], text=label_text[i],
             showarrow=False,
             font=dict(color="white", size=13, family="Arial Black, DejaVu Sans, sans-serif"),
             xanchor="center", yanchor="middle", bgcolor="rgba(0,0,0,0.35)", borderpad=1)
        for i in range(len(label_text))
    ]

    inset_edge, inset_node, inset_text = _build_compact_state_map_traces(env, env_id=env_id)
    fig = go.Figure(data=[*traces, inset_edge, inset_node, inset_text])
    # enlarged inset: wider and taller so state IDs don't overlap
    inset_w = max(0.22, min(0.36, 0.28 * (env.shape[1] / max(1, env.shape[0]))))
    inset_h_top = 0.98
    inset_h_bot = inset_h_top - inset_w * (env.shape[0] / max(1, env.shape[1]))
    inset_h_bot = max(0.55, inset_h_bot)  # don't let it shrink too much
    fig.update_layout(
        xaxis2=dict(visible=False, domain=[0.02, 0.02 + inset_w],
                     range=[-0.8, ((env.shape[1] - 1) * 1.15) + 0.8],
                     constrain="domain", anchor="y2"),
        yaxis2=dict(visible=False, domain=[inset_h_bot, inset_h_top],
                     range=[((env.shape[0] - 1) * 1.15) + 0.8, -0.8],
                     scaleanchor="x2", scaleratio=1, constrain="domain", anchor="x2"),
    )
    fig.add_annotation(x=0.02, y=inset_h_bot - 0.01, xref="paper", yref="paper",
                       text="<b>State Colour Map</b>", showarrow=False,
                       xanchor="left", yanchor="top", font=dict(size=14))
    fig.add_annotation(x=0.02, y=inset_h_bot - 0.04, xref="paper", yref="paper",
                       text=_build_metadata_text(env, metadata), showarrow=False,
                       xanchor="left", yanchor="top", align="left", font=dict(size=12),
                       bgcolor="rgba(255,255,255,0.8)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1)

    fig.update_layout(
        title=title,
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                   aspectmode="data", camera=dict(projection=dict(type="orthographic")),
                   annotations=scene_annotations),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.75)", bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1, font=dict(size=10)),
    )

    if out_html is not None:
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_html), include_plotlyjs="cdn")

    return fig


# ---------------------------------------------------------------------------
# Footer annotation helper
# ---------------------------------------------------------------------------

def add_footer(fig, text: str, fontsize: float = 5.5, color: str = "#888888"):
    """Add a footer annotation below a matplotlib figure.

    Works for both Figure and Axes objects.
    """
    if hasattr(fig, 'figure'):
        fig = fig.figure
    fig.text(0.5, 0.005, text, ha="center", va="bottom",
             fontsize=fontsize, color=color)
