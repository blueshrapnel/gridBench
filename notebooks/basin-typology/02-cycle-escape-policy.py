# %% [markdown]
# # 51 — Cycle-escape: exemplar-plus-table figure and role-summary heatmap
#
# Prototype replacements for the paper's 12-grid cycle-escape figure
# (twists-home-vectors, fig:cycle-escape-four-rooms), per Karen's review:
# the 12-panel version is evidence, not explanation.
#
# Two outputs:
#   F-cycle-escape-exemplar-table.png — ONE exemplar twist (the clearest
#     asymmetric four_rooms 7x7 run-best), four label panels stripped of
#     stat titles, with a compact role table beneath:
#     label | role | stay % | escape -> | reading.
#   F-cycle-escape-role-heatmap.png — rows = runs (7x7, 9x9, 13x13
#     four_rooms cohorts), columns = labels ordered DOM->SLNC, colour =
#     stay-in-home-cycle rate, cell text = dominant escape destination.
#     Robustness across runs and SCALE without parsing arrows.
#
# Cohort safety per feedback_fepm_sigmas_not_for_analytics: exclude any
# run_name containing "fepm" or "random".  Data: schema10-export collated
# run-best pickles (same source as script 30, which fed the paper).

# %%
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gridbench.functional_graph.probe_env import build_goal_free_probe_env
from gridbench.functional_graph.label_graphs import label_graphs, walls_and_nonwalls
from gridbench.papers.home_vectors import WALL_COLOUR


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


_here = _nb_dir(
    "/home/karen/phd-marlyn/gridBench/notebooks/basin-typology/"
    "02-cycle-escape-policy.py"
)

SCHEMA_EXPORT = Path("/media/merlin/grid-twist/data-schema-10/multi/schema10-export")
FIG_DIR = _here / "figs"
FIG_DIR.mkdir(exist_ok=True)
PAPER_FIGURES = Path(
    "/home/karen/Dropbox/phd/writing/twists-home-vectors/figures"
)
PAPER_FIGURES.mkdir(parents=True, exist_ok=True)

ENV_ID = "four_rooms"
DET = 0.97
SHAPES = [(7, 7), (9, 9), (13, 13)]
ACTION_NAMES = "NESW"  # display only; roles are what matter

# %% [markdown]
# ## Per-sigma cycle-escape metrics

# %%
def quadrant(s: int, H: int, W: int) -> str:
    r, c = divmod(int(s), W)
    return f"R{(0 if r < H // 2 else 2) + (0 if c < W // 2 else 1)}"


def cycle_escape_stats(env, sigma: np.ndarray) -> dict:
    """Coverage ranking + one-step stay/escape roles at the dominant cycle."""
    H, W = env.shape
    walls, nonwall = walls_and_nonwalls(env)
    graphs = label_graphs(env, sigma)
    n_nonwall = len(nonwall)

    cov = []
    for g in graphs:
        bs = np.asarray(g.fg.basin_sizes, dtype=int)
        cov.append((int(bs.max()) / n_nonwall) if bs.size else 0.0)
    order = list(np.argsort(cov)[::-1])  # DOM ... SLNC
    dom = order[0]
    # The home cycle is the attractor of the dominant label's LARGEST
    # basin (cycles[b] aligns with basin_sizes[b]) -- not the longest
    # cycle: wall-pinned self-loops can out-length-tie a real orbit.
    fg_dom = graphs[dom].fg
    b_star = int(np.argmax(np.asarray(fg_dom.basin_sizes, dtype=int)))
    cycle_dom = set(int(s) for s in fg_dom.cycles[b_star]) & set(nonwall)
    home_room = None
    if cycle_dom:
        rooms = [quadrant(s, H, W) for s in cycle_dom]
        home_room = max(set(rooms), key=rooms.count)

    labels = []
    for rank, li in enumerate(order):
        succ = graphs[li].succ
        stay = esc_rooms = None
        if cycle_dom:
            stays, escapes = 0, []
            for s in cycle_dom:
                t = int(succ[s])
                if t in cycle_dom:
                    stays += 1
                else:
                    escapes.append(quadrant(t, H, W))
            stay = stays / len(cycle_dom)
            esc_rooms = escapes
        focus_room, focus = "—", 0.0
        if esc_rooms:
            focus_room = max(set(esc_rooms), key=esc_rooms.count)
            focus = esc_rooms.count(focus_room) / len(esc_rooms)
        labels.append({
            "label": li, "rank": rank, "coverage": cov[li], "stay": stay,
            "focus_room": focus_room if esc_rooms else "—",
            "focus": focus if esc_rooms else 0.0,
            "n_escape": len(esc_rooms) if esc_rooms else 0,
        })
    return {"labels": labels, "cycle_dom": cycle_dom, "home_room": home_room,
            "graphs": graphs, "order": order}


def load_cohort(shape):
    env = build_goal_free_probe_env(ENV_ID, shape, DET)
    root = SCHEMA_EXPORT / f"shape={shape[0]}x{shape[1]}" / f"env_id={ENV_ID}"
    pickles = sorted(root.rglob("*beta=1/*/*.pickle")) or sorted(root.rglob("*.pickle"))
    rows = []
    for p in pickles:
        blob = pickle.load(open(p, "rb"))
        prov = blob.get("provenance", {}) if isinstance(blob.get("provenance"), dict) else {}
        run_name = (prov.get("run_name") or "").lower()
        if "fepm" in run_name or "random" in run_name or "rand-" in run_name:
            continue
        sigma = np.asarray(blob["sigma"], dtype=int)
        stats = cycle_escape_stats(env, sigma)
        rows.append({"path": p, "run_name": prov.get("run_name", "") or p.parent.name,
                     "sigma": sigma, "stats": stats, "shape": shape})
    return env, rows

# %%
cohorts = {}
envs = {}
for shape in SHAPES:
    try:
        env, rows = load_cohort(shape)
    except Exception as e:  # noqa: BLE001
        print(f"{shape}: skipped ({e})")
        continue
    envs[shape], cohorts[shape] = env, rows
    print(f"{shape}: {len(rows)} runs in cohort")

# %% [markdown]
# ## Select the exemplar: clearest asymmetric 7x7 run-best
# Asymmetric = dominant coverage high, silenced coverage low.  Print the
# ranking to verify against the paper figure (expects 0.85 / 0.82 / 0.72).

# %%
def is_asymmetric(row):
    ls = row["stats"]["labels"]
    return (ls[-1]["coverage"] <= 0.25
            and ls[-1]["stay"] is not None and ls[-1]["stay"] >= 0.65)

def asym_key(row):
    ls = row["stats"]["labels"]
    return (is_asymmetric(row), ls[0]["coverage"])

seven = sorted(cohorts[(7, 7)], key=asym_key, reverse=True)
for r in seven[:5]:
    ls = r["stats"]["labels"]
    print(f"  cov={ls[0]['coverage']:.2f} .. {ls[-1]['coverage']:.2f}  "
          f"stay(slnc)={ls[-1]['stay']:.2f}  {r['run_name'][:60]}")
EXEMPLAR = seven[0]

# %% [markdown]
# ## Figure A: exemplar four label panels + role table

# %%
ROOM_TINTS = {"R0": "#f5e9d4", "R1": "#dce8f5", "R2": "#e3f0dd", "R3": "#f0e0ec"}
ROLE_NAMES = {0: "dominant — homing", 1: "intermediate", 2: "intermediate", 3: "silenced"}


def draw_panel(ax, env, stats, rank):
    H, W = env.shape
    walls, nonwall = walls_and_nonwalls(env)
    info = stats["labels"][rank]
    li = info["label"]
    succ = stats["graphs"][li].succ
    cyc = stats["cycle_dom"]
    for s in range(H * W):
        r, c = divmod(s, W)
        color = WALL_COLOUR if s in walls else ROOM_TINTS[quadrant(s, H, W)]
        ax.add_patch(plt.Rectangle((c, H - 1 - r), 1, 1, facecolor=color,
                                   edgecolor="white", linewidth=0.8))
    for s in cyc:
        r, c = divmod(s, W)
        ax.add_patch(plt.Rectangle((c + 0.04, H - 1 - r + 0.04), 0.92, 0.92,
                                   fill=False, edgecolor="#d4a017", linewidth=2.2))
    for s in cyc:
        r, c = divmod(s, W)
        t = int(succ[s])
        if t == s:
            ax.plot(c + 0.5, H - 1 - r + 0.5, "o", color="black", ms=4)
            continue
        tr, tc = divmod(t, W)
        dr, dc = tr - r, tc - c
        if abs(dr) > 1: dr = -np.sign(dr)      # wrap (not expected in four_rooms)
        if abs(dc) > 1: dc = -np.sign(dc)
        colr = "#2166ac" if t in cyc else "#d73027"
        ax.annotate("", xy=(c + 0.5 + 0.38 * dc, H - 1 - r + 0.5 - 0.38 * dr),
                    xytext=(c + 0.5, H - 1 - r + 0.5),
                    arrowprops=dict(arrowstyle="-|>", color=colr, lw=2.4))
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    role = ROLE_NAMES.get(rank, "intermediate")
    ax.set_title(f"label {ACTION_NAMES[li]}\n{role}", fontsize=11)


def interpretation(info, rank, n_ranks, home_room):
    if rank == 0:
        return "holds the home cycle"
    if info["stay"] >= 0.65:
        return "cycle-respecting at home"
    if info["focus"] >= 0.5:
        if info["focus_room"] == home_room:
            return "spills within the home room"
        return f"directed escape to {info['focus_room']}"
    return "diffuse escape"


fig = plt.figure(figsize=(13.2, 5.4), dpi=170)
gs = fig.add_gridspec(2, 4, height_ratios=[3.1, 1.15], hspace=0.16)
env7 = envs[(7, 7)]
st = EXEMPLAR["stats"]
for rank in range(4):
    ax = fig.add_subplot(gs[0, rank])
    draw_panel(ax, env7, st, rank)

ax_t = fig.add_subplot(gs[1, :]); ax_t.axis("off")
home = st["home_room"]
col_labels = ["label", "role", "coverage", "stay in home cycle",
              "escape →", "reading"]
cell_rows = []
for rank, info in enumerate(st["labels"]):
    esc = "—" if info["n_escape"] == 0 else (
        f"{info['focus_room']}{' (home)' if info['focus_room'] == home else ''}"
        f" ({info['focus']:.0%} of escapes)")
    cell_rows.append([
        ACTION_NAMES[info["label"]],
        ROLE_NAMES.get(rank, "intermediate"),
        f"{info['coverage']:.2f}",
        f"{info['stay']:.0%}",
        esc,
        interpretation(info, rank, 4, home),
    ])
table = ax_t.table(cellText=cell_rows, colLabels=col_labels,
                   loc="center", cellLoc="center",
                   colWidths=[0.06, 0.20, 0.09, 0.16, 0.20, 0.27])
table.auto_set_font_size(False); table.set_fontsize(9.5)
table.scale(1.0, 1.45)
for j in range(len(col_labels)):
    table[0, j].set_text_props(weight="bold")
    table[0, j].set_facecolor("#e8e8e8")

fig.suptitle(
    f"Cycle-escape roles, one exemplar (four_rooms 7×7, clearest asymmetric run-best; "
    f"home cycle gold, home room {home}).\n"
    "Arrows: one step from each home-cycle state under that label — "
    "blue stays on the cycle, red escapes, dot = self-loop.",
    fontsize=10.5)
out_a = FIG_DIR / "F-cycle-escape-exemplar-table.png"
fig.savefig(out_a, bbox_inches="tight")
plt.close(fig)
print("saved", out_a)

# %% [markdown]
# ## Figure B: role-summary heatmap across runs and scales

# %%
rows, row_labels, row_shapes = [], [], []
for shape in SHAPES:
    if shape not in cohorts:
        continue
    for r in sorted(cohorts[shape], key=asym_key, reverse=True):
        ls = r["stats"]["labels"]
        rows.append([info["stay"] if info["stay"] is not None else np.nan
                     for info in ls])
        ann = []
        for rank, info in enumerate(ls):
            if rank == 0 or info["n_escape"] == 0:
                ann.append("—")
            elif info["focus"] >= 0.5:
                ann.append(f"→{info['focus_room']}")
            else:
                ann.append("·")
        sig = ""
        try:
            blob = pickle.load(open(r["path"], "rb"))
            prov = blob.get("provenance", {})
            sig = str(prov.get("sigma_hash", ""))[:8]
        except Exception:
            sig = r["path"].parent.parent.name[:8]
        row_labels.append(f"{shape[0]}×{shape[1]}  {sig}  cov {ls[0]['coverage']:.2f}")
        row_shapes.append((shape, ann))

M = np.array(rows, dtype=float)
n = len(rows)
fig, ax = plt.subplots(figsize=(7.4, 0.42 * n + 1.8), dpi=170)
im = ax.imshow(M, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
ax.set_xticks(range(4), ["dominant", "2nd", "3rd", "silenced"], fontsize=10)
ax.set_yticks(range(n), row_labels, fontsize=7.5)
for i, (shape, ann) in enumerate(row_shapes):
    for j, a in enumerate(ann):
        v = M[i, j]
        ax.text(j, i, a, ha="center", va="center", fontsize=8,
                color="white" if v > 0.6 else "#222222")
# separators between shape groups
prev = None
for i, (shape, _) in enumerate(row_shapes):
    if prev is not None and shape != prev:
        ax.axhline(i - 0.5, color="black", lw=1.4)
    prev = shape
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("stay-in-home-cycle rate", fontsize=9)
ax.set_title("Cycle-escape roles across runs and scales (four_rooms)\n"
             "colour = stay rate at the dominant home cycle; "
             "→Rk = dominant escape room (· diffuse, — none)",
             fontsize=10)
out_b = FIG_DIR / "F-cycle-escape-role-heatmap.png"
fig.savefig(out_b, bbox_inches="tight")
plt.close(fig)
print("saved", out_b)

# %% [markdown]
# ## Policy view: what each label is FOR (build-up to cycle-escape)
#
# Karen's reframe: infer label roles from the POLICY, not from one-step
# successor tests.  For the same exemplar sigma, solve the goal-optimal
# policy for every goal (gridcore kernel, beta=1), colour each state by
# the greedy label, overlay a greedy trajectory, and add each label's
# policy usage share to the role table.  The figure should stand alone:
# labels have jobs, and the silenced label has (almost) none.

# %%
from gridcore.bridge import EvalConfig, _state_dist_class, build_twisted_env_from_sigma
from gridcore.info import DecisionInformation as GC_DI

BETA, THETA = 1.0, 1e-5
ex_env = envs[(7, 7)]
ex_sigma = EXEMPLAR["sigma"]
ex_stats = EXEMPLAR["stats"]
H7, W7 = ex_env.shape
_, nonwall7 = walls_and_nonwalls(ex_env)
walls7 = set(range(H7 * W7)) - set(nonwall7)
goals7 = list(nonwall7)

policies = {}
for g in goals7:
    cfg = EvalConfig(env_id="four_rooms", shape=(H7, W7), goal=int(g), beta=BETA,
                     determinism=DET, manhattan=True, theta=THETA,
                     state_dist="uniform")
    env_g = build_twisted_env_from_sigma(ex_sigma, cfg)
    di = GC_DI(env_g, _state_dist_class("uniform")(env_g), THETA,
               max_iterations=200_000, max_info_iterations=10_000)
    pi, _, _ = di.get_opt_policy_Z_free_vector(BETA)
    policies[g] = np.asarray(pi, dtype=float)
print(f"solved {len(policies)} goal policies (gridcore, beta={BETA})")

# usage share: fraction of (state, goal) pairs whose greedy action is label l
greedy = {g: np.argmax(policies[g], axis=1) for g in goals7}
usage = np.zeros(4)
for g in goals7:
    for s in goals7:
        if s == g:
            continue
        usage[greedy[g][s]] += 1
usage /= usage.sum()
print("usage share by label:", {ACTION_NAMES[i]: f"{usage[i]:.1%}" for i in range(4)})

# %%
LABEL_COLOURS = {0: "#3d7fb5", 1: "#f59322", 2: "#4ca64c", 3: "#d64545"}  # N E S W
label_succ = {li: ex_stats["graphs"][li].succ for li in range(4)}


def rollout(goal, start, cap=80):
    path = [int(start)]
    s = int(start)
    for _ in range(cap):
        if s == goal:
            break
        li = int(greedy[goal][s])
        s = int(label_succ[li][s])
        if s == path[-1]:
            break  # self-loop; stuck
        path.append(s)
    return path


def draw_policy_panel(ax, goal, start, title):
    for s in range(H7 * W7):
        r, c = divmod(s, W7)
        is_wall = s in walls7
        colr = WALL_COLOUR if is_wall else LABEL_COLOURS[int(greedy[goal][s])]
        ax.add_patch(plt.Rectangle((c, H7 - 1 - r), 1, 1, facecolor=colr,
                                   edgecolor="white", linewidth=0.7,
                                   alpha=1.0 if is_wall else 0.75))
    for s in ex_stats["cycle_dom"]:
        r, c = divmod(s, W7)
        ax.add_patch(plt.Rectangle((c + 0.05, H7 - 1 - r + 0.05), 0.9, 0.9,
                                   fill=False, edgecolor="#b8860b", linewidth=2.4))
    path = rollout(goal, start)
    xs = [divmod(s, W7)[1] + 0.5 for s in path]
    ys = [H7 - 1 - divmod(s, W7)[0] + 0.5 for s in path]
    ax.plot(xs, ys, color="white", lw=4.5, solid_capstyle="round", zorder=8)
    ax.plot(xs, ys, color="black", lw=2.0, solid_capstyle="round", zorder=9)
    ax.plot(xs[0], ys[0], "o", color="black", ms=8, zorder=10)
    gr, gc = divmod(goal, W7)
    ax.text(gc + 0.5, H7 - 1 - gr + 0.5, "×", ha="center", va="center",
            fontsize=17, fontweight="bold", zorder=11)
    ax.set_xlim(0, W7); ax.set_ylim(0, H7)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10.5)


# story goals: on the home cycle / adjacent room / opposite room
home_cells = sorted(ex_stats["cycle_dom"])
goal_home = home_cells[len(home_cells) // 2]
def _room_goal(room):
    cands = [s for s in goals7 if quadrant(s, H7, W7) == room
             and s not in ex_stats["cycle_dom"]]
    return cands[len(cands) // 2]
goal_adj = _room_goal("R1")
goal_opp = _room_goal("R3")
start_far = max(goals7, key=lambda s: abs(divmod(s, W7)[0] - divmod(goal_home, W7)[0])
                + abs(divmod(s, W7)[1] - divmod(goal_home, W7)[1]))

fig = plt.figure(figsize=(13.2, 6.2), dpi=170)
gs = fig.add_gridspec(2, 3, height_ratios=[3.2, 1.25], hspace=0.18)
panels = [
    (goal_home, start_far, "goal ON the home cycle:\nthe homing label does everything"),
    (goal_adj, start_far, "goal in an adjacent room:\nan escape label takes over locally"),
    (goal_opp, ex_stats and min(goals7), "goal in the opposite room:\nescape labels chain, then hand back"),
]
for k, (g, st, ttl) in enumerate(panels):
    ax = fig.add_subplot(gs[0, k])
    draw_policy_panel(ax, g, st, ttl)

ax_t = fig.add_subplot(gs[1, :]); ax_t.axis("off")
home = ex_stats["home_room"]
col_labels = ["label", "role (goal-free anatomy)", "coverage",
              "stay in home cycle", "policy usage share", "reading"]
cell_rows = []
for rank, info in enumerate(ex_stats["labels"]):
    li = info["label"]
    cell_rows.append([
        ACTION_NAMES[li],
        ROLE_NAMES.get(rank, "intermediate"),
        f"{info['coverage']:.2f}",
        f"{info['stay']:.0%}",
        f"{usage[li]:.1%}",
        interpretation(info, rank, 4, home),
    ])
table = ax_t.table(cellText=cell_rows, colLabels=col_labels,
                   loc="center", cellLoc="center",
                   colWidths=[0.06, 0.22, 0.09, 0.15, 0.15, 0.25])
table.auto_set_font_size(False); table.set_fontsize(9.5)
table.scale(1.0, 1.42)
for j in range(len(col_labels)):
    table[0, j].set_text_props(weight="bold")
    table[0, j].set_facecolor("#e8e8e8")
for i in range(len(cell_rows)):
    li = ex_stats["labels"][i]["label"]
    table[i + 1, 0].set_facecolor(LABEL_COLOURS[li])
    table[i + 1, 0].set_text_props(color="white", weight="bold")

fig.suptitle(
    "What each label is for: greedy label of the goal-optimal policy (colour), same exemplar twist.\n"
    "Gold outline: the dominant label's home cycle.  Black path: greedy trajectory (dot = start, × = goal).",
    fontsize=10.5)
out_c = FIG_DIR / "F-cycle-escape-policy-view.png"
fig.savefig(out_c, bbox_inches="tight")
plt.close(fig)
print("saved", out_c)

# %% [markdown]
# ## Policy view v2 — per-panel usage bars + route label strips
#
# Fixes from Karen's review of v1: the home-goal panel DOES use E at two
# funnel cells, so the headline must carry honest counts; and the
# goal-free table under goal-specific panels was ambiguous.  v2 gives
# every panel its own stacked usage bar (this goal only), draws the
# trajectory's label sequence as a colour strip under the panel, and
# puts the all-goals aggregate in a clearly-labelled fourth column.

# %%
def greedy_counts(goal):
    counts = np.zeros(4)
    for s in goals7:
        if s != goal:
            counts[int(greedy[goal][s])] += 1
    return counts / counts.sum()


def rollout_labels(goal, start, cap=80):
    s, labels_seq, path = int(start), [], [int(start)]
    for _ in range(cap):
        if s == goal:
            break
        li = int(greedy[goal][s])
        t = int(label_succ[li][s])
        if t == path[-1] and t != goal:
            break
        labels_seq.append(li)
        path.append(t)
        s = t
    return path, labels_seq


def stacked_bar(ax, shares, y=0.5, h=0.55):
    x = 0.0
    for li in (3, 2, 1, 0):        # W S E N display order
        w = shares[li]
        if w <= 0:
            continue
        ax.barh(y, w, left=x, height=h, color=LABEL_COLOURS[li],
                edgecolor="white", linewidth=0.6)
        if w >= 0.06:
            ax.text(x + w / 2, y, f"{ACTION_NAMES[li]} {w:.0%}",
                    ha="center", va="center", fontsize=8.2,
                    color="white", fontweight="bold")
        x += w
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")


def label_strip(ax, labels_seq):
    n = len(labels_seq)
    for k, li in enumerate(labels_seq):
        ax.add_patch(plt.Rectangle((k, 0), 1, 1, facecolor=LABEL_COLOURS[li],
                                   edgecolor="white", linewidth=0.5))
    ax.text(-0.6, 0.5, "route:", ha="right", va="center", fontsize=8.5)
    ax.set_xlim(-6, max(n, 22)); ax.set_ylim(-0.1, 1.1)
    ax.axis("off")


PANELS = [
    (goal_home, start_far, "goal on the home cycle"),
    (goal_adj, start_far, "goal in an adjacent room"),
    (goal_opp, min(goals7), "goal in the opposite room"),
]

fig = plt.figure(figsize=(14.4, 5.9), dpi=170)
gs = fig.add_gridspec(3, 4, height_ratios=[3.4, 0.42, 0.42],
                      width_ratios=[1, 1, 1, 0.55], hspace=0.14, wspace=0.12)

for k, (g, st, ttl) in enumerate(PANELS):
    shares = greedy_counts(g)
    lead = int(np.argmax(shares))
    ax = fig.add_subplot(gs[0, k])
    draw_policy_panel(ax, g, st, "")
    n_states = len(goals7) - 1
    ax.set_title(f"{ttl}\n{ACTION_NAMES[lead]} steers "
                 f"{shares[lead] * n_states:.0f} of {n_states} states",
                 fontsize=10.5)
    ax_b = fig.add_subplot(gs[1, k])
    stacked_bar(ax_b, shares)
    path, seq = rollout_labels(g, st)
    ax_s = fig.add_subplot(gs[2, k])
    label_strip(ax_s, seq)

# aggregate column
ax_txt = fig.add_subplot(gs[0, 3]); ax_txt.axis("off")
order_use = np.argsort(usage)[::-1]
lines = ["across ALL 40 goals,", "share of greedy decisions:", ""]
ax_txt.text(0.02, 0.98, "\n".join(lines), fontsize=10, va="top")
for j, li in enumerate(order_use):
    y = 0.72 - j * 0.16
    ax_txt.add_patch(plt.Rectangle((0.05, y - 0.05), 0.12, 0.11,
                                   facecolor=LABEL_COLOURS[int(li)],
                                   transform=ax_txt.transAxes))
    role = {0: "homing", 1: "travel", 2: "travel", 3: ""}
    ax_txt.text(0.22, y, f"{ACTION_NAMES[int(li)]}  {usage[int(li)]:.0%}",
                fontsize=11.5, va="center", fontweight="bold",
                transform=ax_txt.transAxes)
ax_txt.text(0.22, 0.72 - 3 * 0.16 - 0.10,
            "the silenced label:\n1% --- the policy has\nretired it",
            fontsize=9.5, va="top", transform=ax_txt.transAxes)

fig.suptitle(
    "The twist gives each label a job.  Colour = greedy label of the goal-optimal policy at each state\n"
    "(gold outline: the homing label's home cycle; black path: greedy route, dot $\\to$ $\\times$; strip: labels issued along the route)",
    fontsize=11)
out_c2 = FIG_DIR / "F-cycle-escape-policy-view-v2.png"
fig.savefig(out_c2, bbox_inches="tight")
paper_c2 = PAPER_FIGURES / "F-cycle-escape-policy-view-v2.png"
fig.savefig(paper_c2, bbox_inches="tight")
plt.close(fig)
print("saved", out_c2)
print("saved", paper_c2)

# %% [markdown]
# ## Seam-correspondence test: Louvain communities vs label-switch seams
#
# Hypothesis (Karen): the twist's basin co-membership communities tile the
# world into the same regions the policy's label switching reveals.  Build
# the co-membership graph for the exemplar sigma (edge weight = number of
# labels under which two states share a basin), Louvain-partition it, then
# (a) quantify: mean label-switch frequency across community-boundary
# edges vs within-community edges (switch frequency = fraction of the 40
# goals whose greedy labels differ across that adjacency), and
# (b) redraw the v2 policy panels with community boundaries overlaid.

# %%
import networkx as nx

nonwall_set7 = set(nonwall7)
basin_ids = {li: ex_stats["graphs"][li].fg.basin_id for li in range(4)}

G = nx.Graph()
G.add_nodes_from(nonwall7)
nw = list(nonwall7)
for i, s in enumerate(nw):
    for t in nw[i + 1:]:
        w = sum(1 for li in range(4) if basin_ids[li][s] == basin_ids[li][t])
        if w > 0:
            G.add_edge(s, t, weight=w)

rng_seeds = range(10)
best = None
for sd in rng_seeds:
    comms = nx.community.louvain_communities(G, weight="weight", seed=sd)
    q = nx.community.modularity(G, comms, weight="weight")
    if best is None or q > best[0]:
        best = (q, comms)
Q_ex, comms = best
comm_of = {}
for k, cset in enumerate(comms):
    for s in cset:
        comm_of[s] = k
print(f"exemplar sigma: Q={Q_ex:.3f}, {len(comms)} communities, "
      f"sizes={sorted((len(c) for c in comms), reverse=True)}")

# grid adjacencies among nonwall states
adj_pairs = []
for s in nonwall7:
    r, c = divmod(s, W7)
    for dr, dc in ((0, 1), (1, 0)):
        rr, cc = r + dr, c + dc
        if rr < H7 and cc < W7:
            t = rr * W7 + cc
            if t in nonwall_set7:
                adj_pairs.append((s, t))

switch_freq = {}
for (s, t) in adj_pairs:
    n_sw = sum(1 for g in goals7 if greedy[g][s] != greedy[g][t])
    switch_freq[(s, t)] = n_sw / len(goals7)

boundary = [(s, t) for (s, t) in adj_pairs if comm_of[s] != comm_of[t]]
within = [(s, t) for (s, t) in adj_pairs if comm_of[s] == comm_of[t]]
sf_b = np.mean([switch_freq[e] for e in boundary])
sf_w = np.mean([switch_freq[e] for e in within])
print(f"adjacencies: {len(adj_pairs)} total, {len(boundary)} on community boundaries")
print(f"mean label-switch frequency: boundary={sf_b:.2f}  within={sf_w:.2f}  "
      f"ratio={sf_b / sf_w:.2f}x")

# top-quartile seam agreement: of the strongest-switching edges, how many
# lie on a community boundary?
thresh = np.quantile(list(switch_freq.values()), 0.75)
hot = [e for e in adj_pairs if switch_freq[e] >= thresh]
hit = sum(1 for e in hot if comm_of[e[0]] != comm_of[e[1]])
print(f"hot seams (top quartile, switch>={thresh:.2f}): {len(hot)}; "
      f"{hit} on community boundaries ({hit / len(hot):.0%})")

# %%
def draw_comm_boundaries(ax):
    for (s, t) in boundary:
        r1, c1 = divmod(s, W7)
        r2, c2 = divmod(t, W7)
        if r1 == r2:   # vertical shared edge
            x = max(c1, c2)
            y0 = H7 - 1 - r1
            ax.plot([x, x], [y0, y0 + 1], color="#2d004b", lw=3.4,
                    solid_capstyle="butt", zorder=12)
        else:          # horizontal shared edge
            y = H7 - 1 - max(r1, r2) + 1
            x0 = c1
            ax.plot([x0, x0 + 1], [y, y], color="#2d004b", lw=3.4,
                    solid_capstyle="butt", zorder=12)


fig = plt.figure(figsize=(14.4, 6.1), dpi=170)
gs = fig.add_gridspec(3, 4, height_ratios=[3.4, 0.42, 0.42],
                      width_ratios=[1, 1, 1, 0.55], hspace=0.14, wspace=0.12)
for k, (g, st, ttl) in enumerate(PANELS):
    shares = greedy_counts(g)
    lead = int(np.argmax(shares))
    ax = fig.add_subplot(gs[0, k])
    draw_policy_panel(ax, g, st, "")
    draw_comm_boundaries(ax)
    n_states = len(goals7) - 1
    ax.set_title(f"{ttl}\n{ACTION_NAMES[lead]} steers "
                 f"{shares[lead] * n_states:.0f} of {n_states} states",
                 fontsize=10.5)
    ax_b = fig.add_subplot(gs[1, k])
    stacked_bar(ax_b, shares)
    path, seq = rollout_labels(g, st)
    ax_s = fig.add_subplot(gs[2, k])
    label_strip(ax_s, seq)

ax_txt = fig.add_subplot(gs[0, 3]); ax_txt.axis("off")
ax_txt.text(0.02, 0.98,
            f"dark seams: Louvain\ncommunities of the twist's\nbasin co-membership graph\n"
            f"(Q={Q_ex:.2f}, {len(comms)} communities)\n\n"
            f"label-switch frequency\nacross a seam: {sf_b:.0%}\n"
            f"within a community: {sf_w:.0%}\n\n"
            f"{hit} of {len(hot)} hottest switching\nedges lie on a seam",
            fontsize=9.5, va="top", transform=ax_txt.transAxes)

fig.suptitle(
    "Policy view with the twist's own map overlaid: label-switch seams vs basin-community boundaries",
    fontsize=11.5)
out_v3 = FIG_DIR / "F-cycle-escape-policy-view-v3-louvain.png"
fig.savefig(out_v3, bbox_inches="tight")
plt.close(fig)
print("saved", out_v3)
