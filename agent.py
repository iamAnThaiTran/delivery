import numpy as np
from heapq import heappush, heappop

# Hàm A* để thay thế BFS: có heuristic (Manhattan distance)
def run_astar(grid, start, goal):
    n_rows, n_cols = len(grid), len(grid[0])
    open_set = [(0 + abs(goal[0]-start[0]) + abs(goal[1]-start[1]), 0, start)]
    came_from = {}
    g_score = {start: 0}
    while open_set:
        _, cost, current = heappop(open_set)
        if current == goal:
            break
        for move, (dx, dy) in zip(['U','D','L','R'], [(-1,0),(1,0),(0,-1),(0,1)]):
            nxt = (current[0]+dx, current[1]+dy)
            if 0 <= nxt[0] < n_rows and 0 <= nxt[1] < n_cols and grid[nxt[0]][nxt[1]] == 0:
                tentative_g = g_score[current] + 1
                if nxt not in g_score or tentative_g < g_score[nxt]:
                    g_score[nxt] = tentative_g
                    priority = tentative_g + abs(goal[0]-nxt[0]) + abs(goal[1]-nxt[1])
                    heappush(open_set, (priority, tentative_g, nxt))
                    came_from[nxt] = current

    if goal not in g_score:
        return 'S', float('inf')

    curr = goal
    while curr != start:
        if curr not in came_from:
            return 'S', float('inf')
        prev = came_from[curr]
        if prev == start:
            dx, dy = curr[0] - start[0], curr[1] - start[1]
            for move, (mx, my) in zip(['U','D','L','R'], [(-1,0),(1,0),(0,-1),(0,1)]):
                if (dx, dy) == (mx, my):
                    return move, g_score[goal]
            break
        curr = prev

    return 'S', g_score[goal]


class Agents:
    def __init__(self):
        self.n_robots = 0
        self.map = None
        self.robots = []
        self.robots_target = []
        self.packages = []
        self.packages_free = []
        self.is_init = False
        self.reserved = set()

    def init_agents(self, state):
        self.n_robots = len(state['robots'])
        self.map = state['map']
        self.robots = [(r-1, c-1, carry) for (r, c, carry) in state['robots']]
        self.robots_target = [None] * self.n_robots
        self.packages = []
        self.packages_free = []
        for p in state['packages']:
            pid, sr, sc, tr, tc, _, deadline = p
            deadline = int(deadline)
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
            deadline = int(deadline)
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
        reserved_positions = set()

        available_robots = [i for i, (_, _, carry) in enumerate(self.robots) if carry == 0]
        for j, (pid, sr, sc, tr, tc, deadline) in enumerate(self.packages):
            if not self.packages_free[j]:
                continue
            best_robot = None
            best_score = -float('inf')
            for i in available_robots:
                r, c, _ = self.robots[i]
                _, d1 = run_astar(self.map, (r, c), (sr, sc))
                _, d2 = run_astar(self.map, (sr, sc), (tr, tc))
                total_time = t + d1 + d2
                margin = int(deadline) - total_time
                if margin < 0:
                    continue
                score = -d1 - d2 + 0.3 * margin
                if score > best_score:
                    best_score = score
                    best_robot = i
            if best_robot is not None:
                old_pid = self.robots_target[best_robot]
                if old_pid is not None:
                    idx_old = next(j for j, pkg in enumerate(self.packages) if pkg[0] == old_pid)
                    self.packages_free[idx_old] = True
                self.robots_target[best_robot] = pid
                self.packages_free[j] = False

        for i in range(self.n_robots):
            r, c, carry = self.robots[i]
            if carry != 0:
                idx = next(j for j, pkg in enumerate(self.packages) if pkg[0] == carry)
                _, _, _, tr, tc, _ = self.packages[idx]
                move, _ = run_astar(self.map, (r, c), (tr, tc))
                pkg_act = '2' if (r, c) == (tr, tc) else '0'
            else:
                pid = self.robots_target[i]
                if pid is not None:
                    idx = next(j for j, pkg in enumerate(self.packages) if pkg[0] == pid)
                    sr, sc = self.packages[idx][1:3]
                    move, _ = run_astar(self.map, (r, c), (sr, sc))
                    pkg_act = '1' if (r, c) == (sr, sc) else '0'
                else:
                    move, pkg_act = 'S', '0'

            dr, dc = {'U': (-1,0), 'D': (1,0), 'L': (0,-1), 'R': (0,1), 'S': (0,0)}[move]
            next_pos = (r + dr, c + dc)
            if next_pos in reserved_positions:
                move = 'S'
                next_pos = (r, c)
            reserved_positions.add(next_pos)
            actions.append((move, pkg_act))

        return actions
