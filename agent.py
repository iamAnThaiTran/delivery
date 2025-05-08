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
    while dq:
        current = dq.popleft()
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nxt = (current[0] + dx, current[1] + dy)
            if (0 <= nxt[0] < n_rows and 0 <= nxt[1] < n_cols
                and grid[nxt[0]][nxt[1]] == 0 and nxt not in dist):
                dist[nxt] = dist[current] + 1
                dq.append(nxt)
    # if unreachable
    if start not in dist:
        return 'S', float('inf')
    # pick neighbor one step closer
    r, c = start
    for move, (dx, dy) in zip(['U','D','L','R'], [(-1,0),(1,0),(0,-1),(0,1)]):
        nxt = (r + dx, c + dy)
        if nxt in dist and dist[nxt] == dist[start] - 1:
            return move, dist[start]
    return 'S', dist[start]

class Agents:
    """
    Greedy + BFS agent with deadline-aware selection.
    Select only packages that can be delivered on time (total_time <= deadline),
    prioritizing smallest slack (deadline - total_time). If none, fallback to nearest.
    """
    def __init__(self):
        self.n_robots = 0
        self.map = None
        self.robots = []            # [(r,c,carry_id)]
        self.robots_target = []     # [pid or None]
        # packages: [(pid, sr, sc, tr, tc, deadline)]
        self.packages = []
        self.packages_free = []     # [True if unassigned]
        self.is_init = False

    def init_agents(self, state):
        """Initialize at t=0."""
        self.n_robots = len(state['robots'])
        self.map = state['map']
        self.robots = [(r-1, c-1, carry) for (r, c, carry) in state['robots']]
        self.robots_target = [None] * self.n_robots
        self.packages = []
        self.packages_free = []
        for p in state['packages']:
            pid, sr, sc, tr, tc, start_time, deadline = p
            self.packages.append((pid, sr-1, sc-1, tr-1, tc-1, deadline))
            self.packages_free.append(True)
        self.is_init = False

    def update_inner_state(self, state):
        # update robots and clear delivered targets
        for i, robot in enumerate(state['robots']):
            pr, pc, pcarry = self.robots[i]
            r, c, carry = robot
            self.robots[i] = (r-1, c-1, carry)
            if pcarry != 0 and carry == 0:
                # delivered, free target
                old_pid = self.robots_target[i]
                if old_pid is not None:
                    idx_old = next(j for j,pkg in enumerate(self.packages) if pkg[0] == old_pid)
                    self.packages_free[idx_old] = True
                self.robots_target[i] = None
        # register new packages
        for p in state['packages']:
            pid, sr, sc, tr, tc, start_time, deadline = p
            if not any(pkg[0] == pid for pkg in self.packages):
                self.packages.append((pid, sr-1, sc-1, tr-1, tc-1, deadline))
                self.packages_free.append(True)

    def get_actions(self, state):
        # initialization
        if not self.is_init:
            self.is_init = True
        else:
            self.update_inner_state(state)

        t = state['time_step']
        actions = []
        for i in range(self.n_robots):
            r, c, carry = self.robots[i]
            # if carrying, deliver via BFS
            if carry != 0:
                idx = next(j for j,pkg in enumerate(self.packages) if pkg[0] == carry)
                _, _, _, tr, tc, _ = self.packages[idx]
                move, _ = run_bfs(self.map, (r, c), (tr, tc))
                pkg_act = '2' if (r, c) == (tr, tc) else '0'
                actions.append((move, pkg_act))
                continue

            # deadline-aware selection
            best_j = None
            best_slack = float('inf')
            # fallback metrics
            fallback_j = None
            fallback_d = float('inf')
            for j,(pid, sr, sc, tr, tc, dl) in enumerate(self.packages):
                if not self.packages_free[j]:
                    continue
                # BFS distances
                _, d1 = run_bfs(self.map, (r, c), (sr, sc))
                _, d2 = run_bfs(self.map, (sr, sc), (tr, tc))
                total_time = t + d1 + d2
                slack = dl - total_time
                # on-time candidate
                if slack >= 0:
                    if slack < best_slack:
                        best_slack = slack
                        best_j = j
                # track nearest if no on-time
                if d1 < fallback_d:
                    fallback_d = d1
                    fallback_j = j
            # assign
            target_j = best_j if best_j is not None else fallback_j
            if target_j is not None:
                self.packages_free[target_j] = False
                self.robots_target[i] = self.packages[target_j][0]
            # move or idle
            pid = self.robots_target[i]
            if pid is not None:
                idx = next(j for j,pkg in enumerate(self.packages) if pkg[0] == pid)
                sr, sc = self.packages[idx][1:3]
                move, _ = run_bfs(self.map, (r, c), (sr, sc))
                pkg_act = '1' if (r, c) == (sr, sc) else '0'
            else:
                move, pkg_act = 'S', '0'
            actions.append((move, pkg_act))
        return actions
