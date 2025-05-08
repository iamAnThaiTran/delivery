import numpy as np
from collections import deque


def run_bfs(grid, start, goal):
    """
    Perform a BFS from `goal` to compute shortest-path distances
    to all reachable cells, then determine the move at `start`
    that leads one step closer to `goal`.
    Returns (move_action, distance).
    """
    n_rows = len(grid)
    n_cols = len(grid[0]) if n_rows > 0 else 0
    dq = deque([goal])
    dist = {goal: 0}
    # BFS outward from goal
    while dq:
        current = dq.popleft()
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nxt = (current[0] + dx, current[1] + dy)
            if (0 <= nxt[0] < n_rows and 0 <= nxt[1] < n_cols and
                grid[nxt[0]][nxt[1]] == 0 and nxt not in dist):
                dist[nxt] = dist[current] + 1
                dq.append(nxt)
    # If `start` unreachable, stay still
    if start not in dist:
        return 'S', float('inf')
    # From `start`, pick a neighbor one step closer to goal
    r, c = start
    for move, (dx, dy) in zip(['U','D','L','R'], [(-1,0),(1,0),(0,-1),(0,1)]):
        nxt = (r + dx, c + dy)
        if nxt in dist and dist[nxt] == dist[start] - 1:
            return move, dist[start]
    # No better neighbor found
    return 'S', dist[start]


class Agents:
    """
    Improved greedy controller with dynamic reassignment and avoiding duplicate targets.
    """
    def __init__(self):
        self.n_robots = 0
        self.map = None
        self.robots = []            # [(r, c, carrying_id)]
        self.robots_target = []     # [pid or None]
        self.packages = []          # [(pid, sr, sc, tr, tc)]
        self.packages_free = []     # [True if unassigned]
        self.is_init = False

    def init_agents(self, state):
        """Initialize at t=0."""
        self.n_robots = len(state['robots'])
        self.map = state['map']
        # zero-index robots
        self.robots = [(r-1, c-1, carry) for (r, c, carry) in state['robots']]
        self.robots_target = [None] * self.n_robots
        self.packages = []
        self.packages_free = []
        for p in state['packages']:
            pid, sr, sc, tr, tc, _, _ = p
            self.packages.append((pid, sr-1, sc-1, tr-1, tc-1))
            self.packages_free.append(True)
        self.is_init = False

    def update_inner_state(self, state):
        # update robots and clear delivered targets
        for i, robot in enumerate(state['robots']):
            prev = self.robots[i]
            r, c, carry = robot
            self.robots[i] = (r-1, c-1, carry)
            if prev[2] != 0 and carry == 0:
                # delivered, mark target free
                self.robots_target[i] = None
        # register new packages
        for p in state['packages']:
            pid, sr, sc, tr, tc, _, _ = p
            if not any(pkg[0] == pid for pkg in self.packages):
                self.packages.append((pid, sr-1, sc-1, tr-1, tc-1))
                self.packages_free.append(True)

    def get_actions(self, state):
        # skip update on first call
        if not self.is_init:
            self.is_init = True
        else:
            self.update_inner_state(state)

        actions = []
        for i in range(self.n_robots):
            r, c, carry = self.robots[i]
            # if carrying, always deliver
            if carry != 0:
                idx = next(j for j,pkg in enumerate(self.packages) if pkg[0] == carry)
                _, _, _, tr, tc = self.packages[idx]
                move, _ = run_bfs(self.map, (r,c), (tr,tc))
                pkg_act = '2' if (r,c) == (tr,tc) else '0'
                actions.append((move, pkg_act))
                continue

            # dynamic reassignment: check for closer unassigned package
            # compute current target dist
            cur_pid = self.robots_target[i]
            cur_dist = float('inf')
            if cur_pid is not None:
                idx_cur = next(j for j,p in enumerate(self.packages) if p[0] == cur_pid)
                sr, sc = self.packages[idx_cur][1:3]
                cur_dist = abs(sr - r) + abs(sc - c)
            # find nearest unassigned pkg
            best_j, best_d = None, float('inf')
            for j,(pid,sr,sc,_,_) in enumerate(self.packages):
                if not self.packages_free[j]: continue
                d = abs(sr - r) + abs(sc - c)
                if d < best_d:
                    best_j, best_d = j, d
            # if no target or found a significantly closer one, reassign
            if best_j is not None and (cur_pid is None or best_d + 1 < cur_dist):
                # free old target
                if cur_pid is not None:
                    idx_old = next(j for j,p in enumerate(self.packages) if p[0] == cur_pid)
                    self.packages_free[idx_old] = True
                # assign new
                self.packages_free[best_j] = False
                self.robots_target[i] = self.packages[best_j][0]

            # now target = robots_target[i]
            if self.robots_target[i] is not None:
                pid = self.robots_target[i]
                idx = next(j for j,p in enumerate(self.packages) if p[0] == pid)
                sr, sc = self.packages[idx][1:3]
                move, _ = run_bfs(self.map, (r,c), (sr,sc))
                pkg_act = '1' if (r,c) == (sr,sc) else '0'
                actions.append((move, pkg_act))
            else:
                # no pkg available, stay idle
                actions.append(('S','0'))

        return actions