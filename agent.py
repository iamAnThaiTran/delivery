import numpy as np
from collections import deque


def run_bfs(grid, start, goal):
    n_rows = len(grid)
    n_cols = len(grid[0]) if n_rows > 0 else 0
    dq = deque([goal])
    dist = {goal: 0}
    while dq:
        current = dq.popleft()
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nxt = (current[0] + dx, current[1] + dy)
            if (0 <= nxt[0] < n_rows and 0 <= nxt[1] < n_cols and
                grid[nxt[0]][nxt[1]] == 0 and nxt not in dist):
                dist[nxt] = dist[current] + 1
                dq.append(nxt)
    if start not in dist:
        return 'S', float('inf')
    r, c = start
    for move, (dx, dy) in zip(['U','D','L','R'], [(-1,0),(1,0),(0,-1),(0,1)]):
        nxt = (r + dx, c + dy)
        if nxt in dist and dist[nxt] == dist[start] - 1:
            return move, dist[start]
    return 'S', dist[start]


class Agents:
    def __init__(self):
        self.n_robots = 0
        self.map = None
        self.robots = []
        self.robots_target = []
        self.packages = []
        self.packages_free = []
        self.is_init = False

    def init_agents(self, state):
        self.n_robots = len(state['robots'])
        self.map = state['map']
        self.robots = [(r-1, c-1, carry) for (r, c, carry) in state['robots']]
        self.robots_target = [None] * self.n_robots
        self.packages = []
        self.packages_free = []
        for p in state['packages']:
            pid, sr, sc, tr, tc, _, deadline = p
            deadline = int(deadline)  # ép kiểu ở đây
            self.packages.append((pid, sr-1, sc-1, tr-1, tc-1, deadline))
            self.packages_free.append(True)
        self.is_init = False

    def update_inner_state(self, state):
        for i, robot in enumerate(state['robots']):
            pr, pc, pcarry = self.robots[i]
            r, c, carry = robot
            self.robots[i] = (r-1, c-1, carry)
            if pcarry != 0 and carry == 0:
                old_pid = self.robots_target[i]
                if old_pid is not None:
                    idx_old = next(j for j, pkg in enumerate(self.packages) if pkg[0] == old_pid)
                    self.packages_free[idx_old] = True
                self.robots_target[i] = None

        for p in state['packages']:
            pid, sr, sc, tr, tc, _, deadline = p
            deadline = int(deadline)  # ép kiểu lại nếu có gói mới
            if not any(pkg[0] == pid for pkg in self.packages):
                self.packages.append((pid, sr-1, sc-1, tr-1, tc-1, deadline))
                self.packages_free.append(True)

    def get_actions(self, state):
        if not self.is_init:
            self.is_init = True
        else:
            self.update_inner_state(state)

        t = state['time_step']
        actions = []
        planned_positions = set()

        for i in range(self.n_robots):
            r, c, carry = self.robots[i]

            if carry != 0:
                idx = next(j for j, pkg in enumerate(self.packages) if pkg[0] == carry)
                _, _, _, tr, tc, deadline = self.packages[idx]
                if t + run_bfs(self.map, (r, c), (tr, tc))[1] > int(deadline):
                    self.robots[i] = (r, c, 0)
                    self.robots_target[i] = None
                    self.packages_free[idx] = True
                    carry = 0

            if carry != 0:
                idx = next(j for j, pkg in enumerate(self.packages) if pkg[0] == carry)
                _, _, _, tr, tc, _ = self.packages[idx]
                move, _ = run_bfs(self.map, (r, c), (tr, tc))
                pkg_act = '2' if (r, c) == (tr, tc) else '0'
            else:
                cur_pid = self.robots_target[i]
                cur_dist = float('inf')
                if cur_pid is not None:
                    idx_cur = next(j for j, p in enumerate(self.packages) if p[0] == cur_pid)
                    sr, sc = self.packages[idx_cur][1:3]
                    cur_dist = abs(sr - r) + abs(sc - c)

                best_j = None
                best_score = -float('inf')
                for j, (pid, sr, sc, tr, tc, deadline) in enumerate(self.packages):
                    if not self.packages_free[j]:
                        continue
                    move1, d1 = run_bfs(self.map, (r, c), (sr, sc))
                    move2, d2 = run_bfs(self.map, (sr, sc), (tr, tc))
                    total_time = t + d1 + d2
                    if total_time > int(deadline):
                        continue
                    score = -d1 + 0.2 * (1000 - int(deadline))  # thêm int()
                    if score > best_score:
                        best_score = score
                        best_j = j

                if best_j is not None:
                    if cur_pid is not None:
                        idx_old = next(j for j, p in enumerate(self.packages) if p[0] == cur_pid)
                        self.packages_free[idx_old] = True
                    self.packages_free[best_j] = False
                    self.robots_target[i] = self.packages[best_j][0]

                pid = self.robots_target[i]
                if pid is not None:
                    idx = next(j for j, pkg in enumerate(self.packages) if pkg[0] == pid)
                    sr, sc = self.packages[idx][1:3]
                    move, _ = run_bfs(self.map, (r, c), (sr, sc))
                    pkg_act = '1' if (r, c) == (sr, sc) else '0'
                else:
                    move, pkg_act = 'S', '0'

            dr, dc = {'U': (-1,0), 'D': (1,0), 'L': (0,-1), 'R': (0,1), 'S': (0,0)}[move]
            next_pos = (r + dr, c + dc)
            if next_pos in planned_positions:
                move = 'S'
            planned_positions.add(next_pos)

            actions.append((move, pkg_act))

        return actions
