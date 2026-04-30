
import numpy as np

from env.grid_env import GridSearchRescueEnv
from agents.dqn_agent import DQNAgent
from agents.replay_buffer import ReplayBuffer


# =========================
# Preprocessing
# =========================
def preprocess_obs(obs, grid_size):
    grid = obs["grid"].flatten()
    pheromone = obs["pheromone"].flatten()

    x, y = obs["position"]
    pos = np.array([x / grid_size, y / grid_size])

    return np.concatenate([grid, pheromone, pos])


# =========================
# Config
# =========================
GRID_SIZE = 8
N_AGENTS = 2
EPISODES = 200
MAX_STEPS = 100

# state = grid + pheromone + position
STATE_DIM = (GRID_SIZE * GRID_SIZE) * 2 + 2
ACTION_DIM = 5


# =========================
# Init
# =========================
env = GridSearchRescueEnv(
    grid_size=GRID_SIZE,
    n_agents=N_AGENTS,
    max_steps=MAX_STEPS
)

agents = [DQNAgent(STATE_DIM, ACTION_DIM) for _ in range(N_AGENTS)]
buffers = [ReplayBuffer() for _ in range(N_AGENTS)]


# =========================
# Training Loop
# =========================
for ep in range(EPISODES):
    obs = env.reset()
    states = [preprocess_obs(o, GRID_SIZE) for o in obs]

    done = False
    total_rewards = [0 for _ in range(N_AGENTS)]

    while not done:
        # select actions
        actions = [
            agents[i].select_action(states[i])
            for i in range(N_AGENTS)
        ]

        # environment step
        next_obs, rewards, done, _ = env.step(actions)
        next_states = [preprocess_obs(o, GRID_SIZE) for o in next_obs]

        # store + train
        for i in range(N_AGENTS):
            buffers[i].push(
                states[i],
                actions[i],
                rewards[i],
                next_states[i],
                done
            )

            agents[i].train(buffers[i])
            total_rewards[i] += rewards[i]

        states = next_states

    # update target networks
    if ep % 10 == 0:
        for agent in agents:
            agent.update_target()

    # logging
    print(f"Episode {ep:03d} | Rewards: {total_rewards} | Eps: {[round(a.epsilon, 2) for a in agents]}")
```
