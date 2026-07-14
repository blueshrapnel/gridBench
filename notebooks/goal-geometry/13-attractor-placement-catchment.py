# %% [markdown]
# # 13 — Attractor placement: wall and doorway catchment
#
# Question (2026-07-12, from the animation review): the run-best home
# cycles keep turning up against walls and beside doorways.  Is the
# placement an accident of seed, or does the wall geometry privilege
# territory?  Two statistics per run-best twist, against base rates:
#
# 1. **Wall adjacency**: fraction of dominant-cycle cells adjacent to a
#    wall or the boundary (base rate = same fraction over all walkable
#    cells — at 7x7 the rooms are 3x3, so the base rate is 82% and the
#    statistic is nearly vacuous; it becomes informative at 13x13+).
# 2. **Doorway placement**: dominant-cycle cells on the door corridors
#    (the wall-cross row/column), plus Manhattan distance from the
#    cycle to the nearest door cell.  Base rate 12% (7x7) down to 4%
#    (15x15).
#
# Cohorts: four_rooms K=1 beta=1 free-energy run-bests --
# 13x perm_balanced g500 (core-silence-hunt-10-07),
# 14x shuffle g500 (core-silence-hunt2-shuffle-10-07),
# 7x shuffle g500 9x9 (core-silence-scale-env-warm-11-07),
# and the three large anecdotes (04-07 13x13 warm g1000, 15x15 g200).
#
# Promoted from gridFour attractor-fingerprint-probe/54 (2026-07-14);
# produces tab:catchment + the "Where the cycles sit" subsection of twists-home-vectors
# sec:anatomy.

# %%
import json
import glob
from collections import Counter

import numpy as np

from gridcore.bridge import EvalConfig, build_twisted_env_from_sigma

BASE = '/media/merlin/grid-twist/gridtwist-outputs'
DELTA = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # N E S W


def four_rooms_walls(n):
    ident = np.tile(np.arange(4), (n * n, 1))
    env = build_twisted_env_from_sigma(
        ident, EvalConfig(env_id='four_rooms', shape=(n, n), goal=0,
                          beta=1.0))
    return np.asarray(env.walls_flat, int)


class FRTopo:
    def __init__(self, n):
        self.n = n
        self.NS = n * n
        self.walls = four_rooms_walls(n)
        self.wallset = set(self.walls.tolist())
        self.walk = np.array([s for s in range(self.NS)
                              if s not in self.wallset])
        self.nwalk = len(self.walk)
        sp = np.zeros((self.NS, 4), int)
        for s in range(self.NS):
            r, c = divmod(s, n)
            for a, (dr, dc) in DELTA.items():
                rr, cc = r + dr, c + dc
                t = rr * n + cc
                ok = (0 <= rr < n and 0 <= cc < n
                      and t not in self.wallset and s not in self.wallset)
                sp[s, a] = t if ok else s
        self.SP = sp
        rows = [w // n for w in self.walls]
        cols = [w % n for w in self.walls]
        self.cross_r = max(set(rows), key=rows.count)
        self.cross_c = max(set(cols), key=cols.count)
        self.doors = [s for s in self.walk
                      if s // n == self.cross_r or s % n == self.cross_c]

    def wall_adjacent(self, s):
        r, c = divmod(s, self.n)
        for dr, dc in DELTA.values():
            rr, cc = r + dr, c + dc
            if (not (0 <= rr < self.n and 0 <= cc < self.n)
                    or rr * self.n + cc in self.wallset):
                return True
        return False


def dominant_cycle(topo, sigma):
    """Cycle cells of the best label's largest basin (run-best sigma)."""
    sigma = np.asarray(sigma, int)
    sinv = np.argsort(sigma, axis=1)
    succ = np.take_along_axis(topo.SP, sinv, axis=1)
    best = None
    for l in range(4):
        f = succ[:, l]
        cyc = np.full(topo.NS, -1, int)
        on = np.zeros(topo.NS, bool)
        state = np.zeros(topo.NS, int)
        ncyc = 0
        for s0 in topo.walk:
            if state[s0]:
                continue
            path, s = [], s0
            while state[s] == 0:
                state[s] = 1
                path.append(s)
                s = f[s]
            if state[s] == 1:
                k = path.index(s)
                for u in path[k:]:
                    cyc[u] = ncyc
                    on[u] = True
                for u in path[:k]:
                    cyc[u] = ncyc
                ncyc += 1
            else:
                for u in path:
                    cyc[u] = cyc[s]
            for u in path:
                state[u] = 2
        sizes = np.bincount(cyc[topo.walk], minlength=max(ncyc, 1))
        cov = sizes.max() / topo.nwalk
        if best is None or cov > best[0]:
            dom = int(sizes.argmax())
            best = (cov, [s for s in topo.walk if on[s] and cyc[s] == dom])
    return best


def room_of(topo, s):
    r, c = divmod(s, topo.n)
    if r == topo.cross_r or c == topo.cross_c:
        return 'door'
    return ('N' if r < topo.cross_r else 'S') + \
           ('W' if c < topo.cross_c else 'E')


def door_dist(topo, cells):
    ds = [(t // topo.n, t % topo.n) for t in topo.doors]
    return min(abs(s // topo.n - r) + abs(s % topo.n - c)
               for s in cells for r, c in ds)


# %%
GROUPS = [
    ('7x7 perm-bal g500', 7,
     f'{BASE}/core-silence-hunt-10-07/*four-rooms-7x7*/*summary.json'),
    ('7x7 shuffle g500', 7,
     f'{BASE}/core-silence-hunt2-shuffle-10-07/*four-rooms-7x7*/'
     '*summary.json'),
    ('9x9 shuffle g500', 9,
     f'{BASE}/core-silence-scale-env-warm-11-07/*four-rooms-9x9*/'
     '*summary.json'),
    ('13x13 perm-bal', 13,
     f'{BASE}/*/g*04-07-b1-free-ga-four-rooms-13x13*/*summary.json'),
    ('15x15 perm-bal', 15,
     f'{BASE}/*/g*04-07-b1-free-ga-four-rooms-15x15*/*summary.json'),
]

for label, n, pat in GROUPS:
    topo = FRTopo(n)
    base_wadj = np.mean([topo.wall_adjacent(s) for s in topo.walk])
    base_door = len(topo.doors) / topo.nwalk
    rooms, dists, wfracs, covs = [], [], [], []
    for p in sorted(glob.glob(pat)):
        h = json.load(open(p))['history']
        cov, cells = dominant_cycle(topo, h[-1]['best_sigma'])
        rs = [room_of(topo, s) for s in cells]
        rooms.append(max(set(rs), key=rs.count))
        dists.append(door_dist(topo, cells))
        wfracs.append(np.mean([topo.wall_adjacent(s) for s in cells]))
        covs.append(cov)
    print(f'{label}: n={len(rooms)}  rooms={dict(Counter(rooms))}')
    print(f'  wall-adjacent cycle cells {np.mean(wfracs)*100:.0f}% '
          f'(base {base_wadj*100:.0f}%)   '
          f'door-corridor cycles {sum(r == "door" for r in rooms)}'
          f'/{len(rooms)} (base {base_door*100:.0f}%)   '
          f'door-dist med {np.median(dists):.0f}   '
          f'coverage med {np.median(covs):.2f}')
