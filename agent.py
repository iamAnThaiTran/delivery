import numpy as np
from collections import deque


def run_bfs(grid, start, goal):
    """
    BFS from `goal` to compute distances then pick one-step move from `start`.
    Returns (move_action, distance).
    """
    n_rows, n_cols = len(grid), len(grid[0]) if grid else 0
    dq = deque([goal]); dist = {goal: 0}
    while dq:
        cur = dq.popleft()
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nxt = (cur[0]+dx, cur[1]+dy)
            if 0<=nxt[0]<n_rows and 0<=nxt[1]<n_cols and grid[nxt[0]][nxt[1]]==0 and nxt not in dist:
                dist[nxt] = dist[cur] + 1; dq.append(nxt)
    if start not in dist:
        return 'S', np.inf
    r,c = start
    for move,(dx,dy) in zip(['U','D','L','R'], [(-1,0),(1,0),(0,-1),(0,1)]):
        nxt = (r+dx, c+dy)
        if nxt in dist and dist[nxt] == dist[start] - 1:
            return move, dist[start]
    return 'S', dist[start]


class Agents:
    """
    Ant Colony Optimization for assignment + BFS for routing.
    Handles cases where robots > packages by assigning only min(#robots,#packages).
    """
    def __init__(self, ant_count=10, iters=5, alpha=1.0, beta=2.0, rho=0.1, Q=100.0):
        self.ant_count = ant_count; self.iters = iters
        self.alpha = alpha; self.beta = beta
        self.rho = rho; self.Q = Q
        self.n_robots = 0; self.map = None
        self.robots = []            # [(r,c,carry)]
        self.packages = []          # [(pid,sr,sc,tr,tc)]
        self.packages_free = []     # True if unassigned
        self.assignments = {}       # robot_id -> package_idx
        self.is_init = False

    def init_agents(self, state):
        self.n_robots = len(state['robots']); self.map = state['map']
        self.robots = [(r-1,c-1,carry) for (r,c,carry) in state['robots']]
        self.packages = []; self.packages_free = []
        for p in state['packages']:
            pid,sr,sc,tr,tc,_,_ = p
            self.packages.append((pid,sr-1,sc-1,tr-1,tc-1))
            self.packages_free.append(True)
        self.assignments = {}
        self.is_init = False

    def update_inner_state(self, state):
        for i,robot in enumerate(state['robots']):
            r,c,carry = robot; pr,pc,pcarry = self.robots[i]
            self.robots[i] = (r-1,c-1,carry)
            if pcarry!=0 and carry==0:
                self.assignments.pop(i,None)
        for p in state['packages']:
            pid,sr,sc,tr,tc,_,_ = p
            if not any(pkg[0]==pid for pkg in self.packages):
                self.packages.append((pid,sr-1,sc-1,tr-1,tc-1))
                self.packages_free.append(True)

    def _aco_assign(self, free_robots, free_pkgs):
        R,P = len(free_robots), len(free_pkgs)
        if R==0 or P==0:
            return {}
        M = min(R,P)
        tau = np.ones((R,P))
        eta = np.zeros((R,P)); dist_mat = np.zeros((R,P))
        for i,ri in enumerate(free_robots):
            r,c,_ = self.robots[ri]
            for j,pj in enumerate(free_pkgs):
                _,sr,sc,_,_ = self.packages[pj]
                _,d = run_bfs(self.map,(r,c),(sr,sc))
                dist_mat[i,j] = d; eta[i,j] = 1.0/(d+1e-6)
        best_global, best_score = {}, np.inf
        for _ in range(self.iters):
            best_local, best_lscore = {}, np.inf
            for _ in range(self.ant_count):
                rem = set(range(P)); assign = {}
                for i in range(M):
                    if not rem:
                        break
                    probs = np.array([ (tau[i,j]**self.alpha)*(eta[i,j]**self.beta) for j in rem ])
                    probs = probs / probs.sum()
                    choices = list(rem)
                    chosen = np.random.choice(choices, p=probs)
                    assign[i] = chosen; rem.remove(chosen)
                score = sum(dist_mat[i,assign[i]] for i in assign)
                if score < best_lscore:
                    best_lscore, best_local = score, assign.copy()
            if best_lscore < best_score:
                best_score, best_global = best_lscore, best_local.copy()
            tau *= (1-self.rho)
            for i,j in best_local.items():
                tau[i,j] += self.Q/(best_lscore+1e-6)
        result = {}
        for i,ri in enumerate(free_robots[:M]):
            j = best_global.get(i,None)
            if j is not None:
                result[ri] = free_pkgs[j]
        return result

    def get_actions(self, state):
        if not self.is_init:
            self.is_init = True
        else:
            self.update_inner_state(state)
        actions = []
        free_robots = [i for i,(r,c,carry) in enumerate(self.robots) if carry==0]
        free_pkgs = [j for j,free in enumerate(self.packages_free) if free]
        new_assign = self._aco_assign(free_robots, free_pkgs)
        for i,pj in new_assign.items():
            self.assignments[i] = pj; self.packages_free[pj] = False
        for i in range(self.n_robots):
            r,c,carry = self.robots[i]
            if carry!=0:
                idx = next(j for j,p in enumerate(self.packages) if p[0]==carry)
                tr,tc = self.packages[idx][3:5]
                move,_ = run_bfs(self.map,(r,c),(tr,tc)); pkg_act='2' if (r,c)==(tr,tc) else '0'
            else:
                pj = self.assignments.get(i,None)
                if pj is not None:
                    sr,sc = self.packages[pj][1:3]
                    move,_ = run_bfs(self.map,(r,c),(sr,sc)); pkg_act='1' if (r,c)==(sr,sc) else '0'
                else:
                    move, pkg_act = 'S','0'
            actions.append((move,str(pkg_act)))
        return actions
