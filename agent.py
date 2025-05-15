import numpy as np
from heapq import heappush, heappop
from collections import deque

def run_bfs(grid, start, goal, return_path=False):
    n_rows = len(grid)
    n_cols = len(grid[0]) if n_rows > 0 else 0
    dq = deque([goal])
    dist = {goal: 0}
    came_from = {}
    visited = set([goal])
    
    while dq:
        current = dq.popleft()
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nxt = (current[0] + dx, current[1] + dy)
            if (0 <= nxt[0] < n_rows and 0 <= nxt[1] < n_cols and
                grid[nxt[0]][nxt[1]] == 0 and nxt not in visited):
                dist[nxt] = dist[current] + 1
                came_from[nxt] = current
                visited.add(nxt)
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
        # Tạo path từ start tới goal dựa trên came_from
        path = []
        curr = start
        while curr != goal:
            next_node = came_from.get(curr)
            if next_node is None:
                # Không tìm được đường đi hoàn chỉnh
                return []
            path.append(next_node)
            curr = next_node
        return path

def run_astar(grid, start, goal, return_path=False, max_steps=10000):
    n_rows, n_cols = len(grid), len(grid[0])
    open_set = [(0 + abs(goal[0]-start[0]) + abs(goal[1]-start[1]), 0, start)]
    came_from = {}
    g_score = {start: 0}
    closed_set = set()
    steps = 0

    while open_set:
        steps += 1
        if steps > max_steps:
            # Quá nhiều bước duyệt, trả về không tìm được đường
            if return_path:
                return []
            else:
                return 'S', float('inf')

        _, cost, current = heappop(open_set)
        if current == goal:
            break
        if current in closed_set:
            continue
        closed_set.add(current)

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
        if return_path:
            return []
        else:
            return 'S', float('inf')

    if return_path:
        path = []
        curr = goal
        while curr != start:
            path.append(curr)
            curr = came_from.get(curr, start)
        path.reverse()
        return path
    else:
        curr = goal
        while curr != start:
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
        self.target_age = []
        self.use_astar = True
        self.future_paths = []
        self.blocking_flags = []
        self.next_positions = []
        self.stuck_counter = []
        self.path_cache = {}  # <== Thêm bộ nhớ đệm kết quả tìm đường
        self.current_robot_positions = {}

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
        self.path_cache = {}  # reset cache mỗi khi khởi tạo lại
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
        self.current_robot_positions.clear()  # Xóa dữ liệu cũ
        for i, (r, c, _) in enumerate(self.robots):
            self.current_robot_positions[i] = (r, c)

    # Hàm tìm đường có cache để tránh tính lại nhiều lần
    # def run_path(self, start, goal, return_path=False, max_path_len=50):
    #     max_path_len = 1000 if len(self.map) <= 12 else 50
    #     key = (start, goal)
    #     cached = self.path_cache.get(key)
    #     if cached is not None:
    #         dist, path = cached
    #     else:
    #         if self.use_astar:
    #             path = run_astar(self.map, start, goal, return_path=True)
    #         else:
    #             path = run_bfs(self.map, start, goal, return_path=True)
    #         dist = len(path) if path else float('inf')
    #         self.path_cache[key] = (dist, path)   # Lưu cache bằng dict
        
    #     if not path or dist == float('inf'):
    #         # Không tìm được đường đi
    #         if return_path:
    #             return []
    #         else:
    #             return 'S', dist
        
    #     if return_path:
    #         # Trả về đường đi cắt ngắn tối đa max_path_len (giúp map lớn)
    #         return path[:max_path_len]
        
    #     # return_path=False: trả move kế tiếp + khoảng cách đầy đủ (không cắt)
    #     next_pos = path[0]
    #     move = self.get_move(start, next_pos)
    #     return move, dist
    def run_path(self, start, goal, return_path=False):
        key = (start, goal, return_path)
        if key in self.path_cache:
            return self.path_cache[key]
        if self.use_astar:
            result = run_astar(self.map, start, goal, return_path)
        else:
            result = run_bfs(self.map, start, goal, return_path)
        self.path_cache[key] = result
        return result


    def get_move(self, start, next_pos):
        dx, dy = next_pos[0] - start[0], next_pos[1] - start[1]
        for move, (mx, my) in zip(['U', 'D', 'L', 'R'], [(-1,0),(1,0),(0,-1),(0,1)]):
            if (dx, dy) == (mx, my):
                return move
        # fallback nếu không tìm được move hợp lệ
        return 'S'


    def get_actions(self, state):
    # Hàm chính để lấy hành động cho tất cả robot
        if not self.is_init:
            self.init_agents(state)
            self.is_init = True
        else:
            self.update_inner_state(state)

        t = state['time_step']
        actions = [None] * self.n_robots
        reserved_positions = {}
        self.blocking_flags = [False] * self.n_robots
        self.future_paths = [[] for _ in range(self.n_robots)]
        self.next_positions = [None for _ in range(self.n_robots)]
        blocking_moves = {}
        blocking_next_positions = {}

        carrying_robots = [i for i, (_, _, carry) in enumerate(self.robots) if carry != 0]

        non_carrying_robots = [i for i, (_, _, carry) in enumerate(self.robots) if carry == 0]
        #carrying_robots = [i for i, (_, _, carry) in enumerate(self.robots) if carry != 0]
        # Sắp xếp theo độ ưu tiên
        
        def get_priority(i, t):
            if self.robots[i][2] == 0:
                return float('inf')
            idx = next(j for j, pkg in enumerate(self.packages) if pkg[0] == self.robots[i][2])
            _, _, _, tr, tc, deadline = self.packages[idx]
            r, c, _ = self.robots[i]
            distance = abs(r - tr) + abs(c - tc)
            return deadline - t - distance
        
        # Sắp xếp robot mang hàng theo độ ưu tiên
        carrying_robots.sort(key=lambda i: get_priority(i, t))
        #print(f"Processing carrying_robots in order: {carrying_robots}")

        #xu ly va cham robot mang hang
        def resolve_carrying_conflict(i, blocking, r, c, next_pos, reserved_positions, t):
            priority_i = get_priority(i, t)
            priority_blocking = get_priority(blocking, t)
            #print(f"robot {i} priority: {priority_i}")
           # print(f"robot {blocking} priority: {priority_blocking}")

            br, bc, _ = self.robots[blocking]
            if priority_i < priority_blocking:
                reserved_positions[next_pos] = i
                move = {(-1,0):'U', (1,0):'D', (0,-1):'L', (0,1):'R'}.get((next_pos[0]-r, next_pos[1]-c), 'S')
                self.stuck_counter[i] = 0
                if (br, bc) == next_pos and (br, bc) in reserved_positions:
                    del reserved_positions[(br, bc)]
                found_alt = False
                for mv, (dr, dc) in zip(['L', 'R', 'U', 'D'], [(0,-1), (0,1), (-1,0), (1,0)]):
                    nr, nc = br + dr, bc + dc
                    if (0 <= nr < len(self.map) and 0 <= nc < len(self.map[0]) and
                        self.map[nr][nc] == 0 and (nr, nc) not in reserved_positions and (nr, nc) != next_pos):
                        actions[blocking] = (mv, actions[blocking][1] if actions[blocking] else '0')
                        self.next_positions[blocking] = (nr, nc)
                        reserved_positions[(nr, nc)] = blocking
                        found_alt = True
                        break
                if not found_alt:
                    actions[blocking] = ('S', actions[blocking][1] if actions[blocking] else '0')
                    self.next_positions[blocking] = (br, bc)
                    self.stuck_counter[blocking] += 1
                return move, next_pos
            # Robot i ưu tiên thấp hơn
            found_alt = False
            best_alt_pos = None
            best_alt_mv = None
            max_dist = -1
            for mv, (dr, dc) in zip(['L', 'R', 'U', 'D'], [(0,-1), (0,1), (-1,0), (1,0)]):
                nr, nc = r + dr, c + dc
                if (0 <= nr < len(self.map) and 0 <= nc < len(self.map[0]) and
                    self.map[nr][nc] == 0 and (nr, nc) not in reserved_positions and (nr, nc) != (br, bc)):
                    dist = abs(nr - br) + abs(nc - bc)
                    if dist > max_dist:
                        max_dist = dist
                        best_alt_pos = (nr, nc)
                        best_alt_mv = mv
                        found_alt = True
            if not found_alt:
                from collections import deque
                q = deque([(r, c, 0, '')])
                seen = {(r, c)}
                max_depth = 2
                while q and not found_alt:
                    cr, cc, d, first = q.popleft()
                    if d >= max_depth:
                        continue
                    for mv, (dr, dc) in zip(['U', 'D', 'L', 'R'], [(-1,0), (1,0), (0,-1), (0,1)]):
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < len(self.map) and 0 <= nc < len(self.map[0]) and
                            self.map[nr][nc] == 0 and (nr, nc) not in reserved_positions and (nr, nc) != (br, bc) and (nr, nc) not in seen):
                            seen.add((nr, nc))
                            q.append((nr, nc, d + 1, first or mv))
                            best_alt_pos = (nr, nc)
                            best_alt_mv = first or mv
                            found_alt = True
                            break
            if found_alt:
                actions[i] = (best_alt_mv, actions[i][1] if actions[i] else '0')
                self.next_positions[i] = best_alt_pos
                reserved_positions[best_alt_pos] = i
                self.stuck_counter[i] = 0
            else:
                actions[i] = ('S', actions[i][1] if actions[i] else '0')
                self.next_positions[i] = (r, c)
                self.stuck_counter[i] += 1
            return actions[i][0], self.next_positions[i]

        # Xử lý robot mang hàng
        for i in carrying_robots:
            r, c, carry = self.robots[i]
            self.target_age[i] += 1
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

            self.next_positions[i] = next_pos
            #print(f"Robot {i} at ({r}, {c}), next_pos = {next_pos}, reserved_positions = {reserved_positions}")

            # Xử lý trường hợp robot mang hàng bị chặn
            if next_pos in reserved_positions:
                blocking = reserved_positions[next_pos]
                blocking_carry = self.robots[blocking][2]
                print(f"Conflict detected: Robot {i} vs Robot {blocking}, blocking_carry = {blocking_carry}")
                if blocking_carry == 0:
                    reserved_positions[next_pos] = i
                    self.blocking_flags[blocking] = True
                    br, bc, _ = self.robots[blocking]
                    old_blocking_pos = self.next_positions[blocking] or (br, bc)
                    if old_blocking_pos in reserved_positions and reserved_positions[old_blocking_pos] == blocking:
                        del reserved_positions[old_blocking_pos]
                    found_alt = False
                    for alt_move in ['L', 'R', 'U', 'D']:
                        dr_alt, dc_alt = {'U': (-1,0), 'D': (1,0), 'L': (0,-1), 'R': (0,1)}[alt_move]
                        alt_pos = (br + dr_alt, bc + dc_alt)
                        if (0 <= alt_pos[0] < len(self.map) and
                            0 <= alt_pos[1] < len(self.map[0]) and
                            self.map[alt_pos[0]][alt_pos[1]] == 0 and
                            alt_pos not in reserved_positions):
                            actions[blocking] = (alt_move, actions[blocking][1] if actions[blocking] else '0')
                            self.next_positions[blocking] = alt_pos
                            reserved_positions[alt_pos] = blocking
                            found_alt = True
                            break
                    if not found_alt:
                        actions[blocking] = ('S', actions[blocking][1] if actions[blocking] else '0')
                        self.next_positions[blocking] = (br, bc)
                        reserved_positions[(br, bc)] = blocking
                    print(f"Robot {blocking} stuck at ({br}, {bc})")
                    print(f"robot", i, "stuck at", next_pos)
                else:
                    move, next_pos = resolve_carrying_conflict(i, blocking, r, c, next_pos, reserved_positions, t)
                    actions[i] = (move, actions[i][1] if actions[i] else '0')
                    reserved_positions[next_pos] = i  # Luôn dự trữ next_pos
                    self.next_positions[i] = next_pos
                    self.stuck_counter[i] += 1 if move == 'S' else 0
                    
            else:
                for j, (br, bc, _) in enumerate(self.robots):
                    if j != i and (br, bc) == next_pos and self.robots[j][2] != 0:
                        blocking = j
                        blocking_carry = self.robots[j][2]
                        #print(f"Direct conflict detected: Robot {i} vs Robot {blocking}")
                        move, next_pos = resolve_carrying_conflict(i, blocking, r, c, next_pos, reserved_positions, t)
                        actions[i] = (move, actions[i][1] if actions[i] else '0')
                        reserved_positions[next_pos] = i
                        self.next_positions[i] = next_pos
                        self.stuck_counter[i] += 1 if move == 'S' else 0
                        break
                    if j != i and (br, bc) == next_pos and self.robots[j][2] == 0:
                        blocking = j
                        blocking_carry = 0
                        print(f"Direct conflict detected: Robot {i} vs Robot {blocking}, blocking_carry = {blocking_carry}")
                        reserved_positions[next_pos] = i  # Robot mang hàng được ưu tiên
                        self.blocking_flags[blocking] = True
                        old_blocking_pos = self.next_positions[blocking] or (br, bc)
                        if old_blocking_pos in reserved_positions and reserved_positions[old_blocking_pos] == blocking:
                            del reserved_positions[old_blocking_pos]
                        found_alt = False
                        for alt_move in ['L', 'R', 'U', 'D']:
                            dr_alt, dc_alt = {'U': (-1,0), 'D': (1,0), 'L': (0,-1), 'R': (0,1)}[alt_move]
                            alt_pos = (br + dr_alt, bc + dc_alt)
                            accepted_flag = True
                            for k, (kr, kc, _) in enumerate(self.robots):
                                if k !=j and self.robots[k][2] != 0 and (kr, kc) == alt_pos:
                                    accepted_flag = False
                                    break


                            
                            if (accepted_flag and
                                0 <= alt_pos[0] < len(self.map) and
                                0 <= alt_pos[1] < len(self.map[0]) and
                                self.map[alt_pos[0]][alt_pos[1]] == 0 and

                                
                                alt_pos not in reserved_positions
                            ):
                                actions[blocking] = (alt_move, actions[blocking][1] if actions[blocking] else '0')
                                self.next_positions[blocking] = alt_pos
                                reserved_positions[alt_pos] = blocking
                                found_alt = True
                                print(f"Robot {blocking} need moved to ({alt_pos[0]}, {alt_pos[1]})")
                                break
                        if not found_alt:
                            actions[blocking] = ('S', actions[blocking][1] if actions[blocking] else '0')
                            self.next_positions[blocking] = (br, bc)
                            self.stuck_counter[blocking] += 1  # Theo dõi robot kẹt
                        print(f"Robot {blocking} stuck at ({br}, {bc})")
                        print(f"Robot {i} stuck at {next_pos}")
                        break
                
            reserved_positions[next_pos] = i
            actions[i] = (move, pkg_act)
    

  

        # Gán mục tiêu cho robot không mang hàng
        available_robots = [i for i in non_carrying_robots if actions[i] is None]
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
                alpha = 0.9 if not self.use_astar else 0.7
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

        # Xử lý robot không mang hàng
        for i in non_carrying_robots:
            if actions[i] is not None:  # Đã được gán hành động né tránh
                continue
            r, c, carry = self.robots[i]
            self.target_age[i] += 1
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

            if next_pos in reserved_positions:
                blocking = reserved_positions[next_pos]
                blocking_carry = self.robots[blocking][2]
                if blocking_carry != 0:  # Bị robot mang hàng chặn
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
                            found_alt = True
                            break
                    if not found_alt:
                        move = 'S'
                        next_pos = (r, c)
                    self.stuck_counter[i] += 1 if move == 'S' else 0
                else:  # Hai robot không mang hàng
                    move = 'S'
                    next_pos = (r, c)

            if next_pos not in reserved_positions:
                reserved_positions[next_pos] = i

            actions[i] = (move, pkg_act)

        return actions