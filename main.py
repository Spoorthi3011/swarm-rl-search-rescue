from environment.grid_env import GridSearchRescueEnv
import random
import time

env = GridSearchRescueEnv(grid_size=8, n_agents=2, n_victims=3, n_obstacles=5)

obs = env.reset()

done = False

while not done:
    actions = [random.randint(0, 4) for _ in range(env.n_agents)]

    obs, rewards, done, _ = env.step(actions)

    env.render()
    print("Rewards:", rewards)
    time.sleep(0.3)
