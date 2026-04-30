import numpy as np

from env.grid_env import GridSearchRescueEnv
from agents.dqn_agent import DQNAgent
from agents.replay_buffer import ReplayBuffer


def preprocess_obs(obs, grid_size):
    grid = obs["grid"].flatten()
    pheromone = obs["pheromone"].flatten()

    x, y = obs["position"]
    pos = np.array([x / grid_size, y / grid_size])

    return np.concatenate([grid, pheromone, pos])


GRID_SIZE = 8
N_AGENTS = 2
EPISODES = 200

STATE_DIM = (GRID_SIZE * GRID_SIZE) * 2 + 2
ACTION_DIM = 5

env = GridSearchRescueEnv(grid_size=GRID_SIZE, n_agents=N_AGENTS)

agents = [DQNAgent(STATE_DIM, ACTION_DIM) for _ in range(N_AGENTS)]
buffers = [ReplayBuffer() for _ in range(N_AGENTS)]

global_best_reward = -float("inf")
global_best_state = None


for ep in range(EPISODES):
    obs = env.reset()
    states = [preprocess_obs(o, GRID_SIZE) for o in obs]

    done = False
    total_rewards = [0] * N_AGENTS

    while not done:
        actions = [agents[i].select_action(states[i]) for i in range(N_AGENTS)]

        next_obs, rewards, done, _ = env.step(actions)
        next_states = [preprocess_obs(o, GRID_SIZE) for o in next_obs]

        for i in range(N_AGENTS):
            buffers[i].push(states[i], actions[i], rewards[i], next_states[i], done)
            agents[i].train(buffers[i])
            total_rewards[i] += rewards[i]

        states = next_states

    # personal best
    for i in range(N_AGENTS):
        agents[i].update_personal_best(total_rewards[i])

    # global best
    best_idx = np.argmax(total_rewards)
    if total_rewards[best_idx] > global_best_reward:
        global_best_reward = total_rewards[best_idx]
        global_best_state = {
            k: v.clone() for k, v in agents[best_idx].model.state_dict().items()
        }

    # PSO step
    if ep % 5 == 0:
        for agent in agents:
            agent.pso_update(global_best_state)

    # target update
    if ep % 10 == 0:
        for agent in agents:
            agent.update_target()

    print(f"Episode {ep} | Rewards: {total_rewards} | Global Best: {global_best_reward}")
