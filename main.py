from env import Environment
from agent import Agents as Agents
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def render_env(env):
    obs = np.array(env.grid)
    vis = np.copy(obs)

    # Vẽ điểm pickup duy nhất (KHÔNG vẽ dropoff)
    for p in env.packages:
        if p.status == 'waiting':
            sr, sc = p.start
            vis[sr][sc] = 4  # Pickup point (orange)

    # Vẽ robot sau cùng (đè lên pickup)
    for robot in env.robots:
        r, c = robot.position
        if robot.carrying == 0:
            vis[r][c] = 2  # Robot chưa cầm hàng (blue)
        else:
            vis[r][c] = 3  # Robot đang cầm hàng (green)

    cmap = colors.ListedColormap(['white', 'black', 'blue', 'green', 'orange'])
    bounds = [0,1,2,3,4,5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(vis, cmap=cmap, norm=norm)

    # Ghi số thứ tự robot và target
    for i, robot in enumerate(env.robots):
        r, c = robot.position
        plt.text(c, r, str(i), va='center', ha='center', color='white', fontsize=8, weight='bold')

        if hasattr(robot, 'agent') and robot.agent.robots_target[i] is not None:
            pid = robot.agent.robots_target[i]
            for p in env.packages:
                if p.pid == pid:
                    if robot.carrying == 0:
                        tr, tc = p.start
                    else:
                        tr, tc = p.target
                    plt.text(tc, tr, str(i), va='center', ha='center', color='black', fontsize=8, weight='bold')
                    break

    plt.title(f"Time Step: {env.t}, Total Reward: {env.total_reward:.2f}")
    plt.pause(0.05)
    plt.clf()



if __name__ == "__main__":
    env = Environment(map_file="map1.txt", max_time_steps=1000, n_robots=5, n_packages=100, seed=10)
    state = env.reset()

    from agent import Agents
    agents = Agents()
    agents.init_agents(state)

    done = False
    rewards = []

    plt.figure(figsize=(6, 6))

    while not done:
        render_env(env)
        actions = agents.get_actions(state)
        state, reward, done, infos = env.step(actions)
        agents.update_inner_state(state)
        rewards.append(reward)

    plt.close()
    print("✅ Episode finished")
    print("🏁 Total reward:", infos['total_reward'])
    print("🕒 Total time steps:", infos['total_time_steps'])
