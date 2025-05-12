import numpy as np
from heapq import heappush, heappop
from collections import deque

def run_bfs(grid, start, goal, return_path=False):
    n_rows = len(grid)
    n_cols = len(grid[0]) if n_rows > 0 else 0
    dq = deque([goal])
    dist = {goal: 0}
    came_from = {}
    while dq:
        current = dq.popleft()
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nxt = (current[0] + dx, current[1] + dy)
            if (0 <= nxt[0] < n_rows and 0 <= nxt[1] < n_cols and
                grid[nxt[0]][nxt[1]] == 0 and nxt not in dist):
                dist[nxt] = dist[current] + 1
                came_from[nxt] = current
                dq.append(nxt)
    if start not in dist:
        return ('S', float('inf')) if not return_path else []
    if not return_path:
        r, c = start
        for move, (dx, dy) in zip(['U','D','L','R'], [(-1,0),(1,0),(0,-1),(0,1)]):
            nxt = (r + dx, c + dy)
            if nxt in dist and dist[nxt] == dist[start] - 1:
                return move, dist[start]
        return 'S', dist[start]
    else:
        path = []
        curr = start
        while curr != goal:
            curr = came_from.get(curr, goal)
            path.append(curr)
        return path

def run_astar(grid, start, goal, return_path=False):
    n_rows, n_cols = len(grid), len(grid[0])
    open_set = [(0 + abs(goal[0]-start[0]) + abs(goal[1]-start[1]), 0, start)]
    came_from = {}
    g_score = {start: 0}
    while open_set:
        _, cost, current = heappop(open_set)
        if current == goal:
            break
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nxt = (current[0]+dx, current[1]+dy)
            if 0 <= nxt[0] < n_rows and 0 <= nxt[1] < n_cols and grid[nxt[0]][nxt[1]] == 0:
                tentative_g = g_score[current] + 1
                if nxt not in g_score or tentative_g < g_score[nxt]:
                    g_score[nxt] = tentative_g
                    priority = tentative_g + abs(goal[0]-nxt[0]) + abs(goal[1]-nxt[1])
                    heappush(open_set, (priority, tentative_g, nxt))
                    came_from[nxt] = current
    if goal not in g_score:
        return ('S', float('inf')) if not return_path else []
    if not return_path:
        curr = goal
        while curr != start:
            prev = came_from[curr]
            if prev == start:
                dx, dy = curr[0] - start[0], curr[1] - start[1]
                for move, (mx, my) in zip(['U','D','L','R'], [(-1,0),(1,0),(0,-1),(0,1)]):
                    if (dx, dy) == (mx, my):
                        return move, g_score[goal]
                break
        return 'S', g_score[goal]
    else:
        path = []
        curr = goal
        while curr != start:
            path.append(curr)
            curr = came_from.get(curr, start)
        path.reverse()
        return path

class Agents:
    def __init__(self):
        self.n_robots = 0
        self.map = None
        self.robots = []
        self.robots_target = []
        self.packages = []
        self.packages_free = []
        self.is_init = False
        self.target_age = []
        self.use_astar = True
        self.future_paths = []
        self.blocking_flags = []
        self.next_positions = []
        self.stuck_counter = []

    def init_agents(self, state):
        self.n_robots = len(state['robots'])
        self.map = state['map']
        self.use_astar = len(self.map) > 12
        self.robots = [(r-1, c-1, carry) for (r, c, carry) in state['robots']]
        self.robots_target = [None] * self.n_robots
        self.target_age = [0] * self.n_robots
        self.packages = []
        self.packages_free = []
        self.future_paths = [[] for _ in range(self.n_robots)]
        self.blocking_flags = [False for _ in range(self.n_robots)]
        self.next_positions = [None for _ in range(self.n_robots)]
        self.stuck_counter = [0 for _ in range(self.n_robots)]
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
                self.robots_target[i] = None
        for p in state['packages']:
            pid, sr, sc, tr, tc, _, deadline = p
            deadline = int(deadline)
            if not any(pkg[0] == pid for pkg in self.packages):
                self.packages.append((pid, sr-1, sc-1, tr-1, tc-1, deadline))
                self.packages_free.append(True)

    def run_path(self, start, goal, return_path=False):
        if self.use_astar:
            return run_astar(self.map, start, goal, return_path)
        return run_bfs(self.map, start, goal, return_path)

    def get_actions(self, state):
        if not self.is_init:
            self.is_init = True
        else:
            self.update_inner_state(state)

        t = state['time_step']
        actions = []
        reserved_positions = {}
        self.blocking_flags = [False] * self.n_robots
        self.future_paths = [[] for _ in range(self.n_robots)]
        self.next_positions = [None for _ in range(self.n_robots)]

        available_robots = [i for i, (_, _, carry) in enumerate(self.robots) if carry == 0]
        for j, (pid, sr, sc, tr, tc, deadline) in enumerate(self.packages):
            if not self.packages_free[j]:
                continue
            best_robot = None
            best_score = -float('inf')
            for i in available_robots:
                r, c, _ = self.robots[i]
                _, d1 = self.run_path((r, c), (sr, sc))
                _, d2 = self.run_path((sr, sc), (tr, tc))
                total_time = t + d1 + d2
                margin = int(deadline) - total_time
                if margin < 0:
                    continue
                alpha = 0.5 if not self.use_astar else 0.3
                score = -d1 - d2 + alpha * margin
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
                self.target_age[best_robot] = 0

        for i in range(self.n_robots):
            r, c, carry = self.robots[i]
            self.target_age[i] += 1
            if carry != 0:
                idx = next(j for j, pkg in enumerate(self.packages) if pkg[0] == carry)
                _, _, _, tr, tc, _ = self.packages[idx]
                path = self.run_path((r, c), (tr, tc), return_path=True)
                self.future_paths[i] = path[:3] if path else []
                if path:
                    next_pos = path[0]
                    move = {
                        (-1, 0): 'U', (1, 0): 'D', (0, -1): 'L', (0, 1): 'R'
                    }.get((next_pos[0]-r, next_pos[1]-c), 'S')
                else:
                    next_pos = (r, c)
                    move = 'S'
                pkg_act = '2' if (r, c) == (tr, tc) else '0'
            else:
                pid = self.robots_target[i]
                if pid is not None:
                    idx = next(j for j, pkg in enumerate(self.packages) if pkg[0] == pid)
                    sr, sc = self.packages[idx][1:3]
                    path = self.run_path((r, c), (sr, sc), return_path=True)
                    self.future_paths[i] = path[:3] if path else []
                    if path:
                        next_pos = path[0]
                        move = {
                            (-1, 0): 'U', (1, 0): 'D', (0, -1): 'L', (0, 1): 'R'
                        }.get((next_pos[0]-r, next_pos[1]-c), 'S')
                    else:
                        next_pos = (r, c)
                        move = 'S'
                    pkg_act = '1' if (r, c) == (sr, sc) else '0'
                else:
                    move, pkg_act = 'S', '0'
                    next_pos = (r, c)

            self.next_positions[i] = next_pos

            for j in range(self.n_robots):
                if i != j and self.future_paths[j]:
                    if (r, c) in self.future_paths[j][:2]:
                        self.blocking_flags[i] = True

            if next_pos in reserved_positions:
                blocking = reserved_positions[next_pos]
                blocking_carry = self.robots[blocking][2]
                if self.next_positions[blocking] == (r, c) and carry != 0 and blocking_carry != 0:
                    if i < blocking:
                        reserved_positions[next_pos] = i
                    else:
                        found_alt = False
                        for alt_move in ['L', 'R', 'U', 'D']:
                            dr_alt, dc_alt = {'U': (-1,0), 'D': (1,0), 'L': (0,-1), 'R': (0,1)}[alt_move]
                            alt_pos = (r + dr_alt, c + dc_alt)
                            if (0 <= alt_pos[0] < len(self.map) and
                                0 <= alt_pos[1] < len(self.map[0]) and
                                self.map[alt_pos[0]][alt_pos[1]] == 0 and
                                alt_pos not in reserved_positions):
                                move = alt_move
                                next_pos = alt_pos
                                reserved_positions[next_pos] = i
                                found_alt = True
                                break
                        if not found_alt:
                            move = 'S'
                            next_pos = (r, c)
                        self.stuck_counter[i] += 1
                        actions.append((move, pkg_act))
                        continue
                elif carry != 0 and blocking_carry == 0:
                    reserved_positions[next_pos] = i
                    # Đánh dấu robot blocking cần né chỗ cho robot mang hàng
                    self.blocking_flags[blocking] = True

                elif carry != 0 and blocking_carry != 0:
                    if i < blocking:
                        reserved_positions[next_pos] = i
                    else:
                        found_alt = False
                        for alt_move in ['L', 'R', 'U', 'D']:
                            dr_alt, dc_alt = {'U': (-1,0), 'D': (1,0), 'L': (0,-1), 'R': (0,1)}[alt_move]
                            alt_pos = (r + dr_alt, c + dc_alt)
                            if (0 <= alt_pos[0] < len(self.map) and
                                0 <= alt_pos[1] < len(self.map[0]) and
                                self.map[alt_pos[0]][alt_pos[1]] == 0 and
                                alt_pos not in reserved_positions):
                                move = alt_move
                                next_pos = alt_pos
                                reserved_positions[next_pos] = i
                                found_alt = True
                                break
                        if not found_alt:
                            move = 'S'
                            next_pos = (r, c)
                        self.stuck_counter[i] += 1
                        actions.append((move, pkg_act))
                        continue
                else:
                    move = 'S'
                    next_pos = (r, c)
            else:
                reserved_positions[next_pos] = i

            if self.blocking_flags[i] :
                if self.stuck_counter[i] >= 0:
                    for alt_move in ['L', 'R', 'U', 'D']:
                        dr_alt, dc_alt = {'U': (-1,0), 'D': (1,0), 'L': (0,-1), 'R': (0,1)}[alt_move]
                        alt_pos = (r + dr_alt, c + dc_alt)
                        if (0 <= alt_pos[0] < len(self.map) and
                            0 <= alt_pos[1] < len(self.map[0]) and
                            self.map[alt_pos[0]][alt_pos[1]] == 0 and
                            alt_pos not in reserved_positions):
                            move = alt_move
                            next_pos = alt_pos
                            reserved_positions[next_pos] = i
                            self.stuck_counter[i] = 0
                            break
            if move != 'S':
                self.stuck_counter[i] = 0

            actions.append((move, pkg_act))

        return actions
