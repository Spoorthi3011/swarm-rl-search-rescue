from env.grid_env import GridSearchRescueEnv
from agents.dqn_agent import DQNAgent
from agents.replay_buffer import ReplayBuffer

import numpy as np

env = GridSearchRescueEnv(grid_size=8, n_agents=2)

state_dim = 8*8 + 2
action_dim = 5

agents = [DQNAgent(state_dim, action_dim) for _ in range(env.n_agents)]
buffers = [ReplayBuffer() for _ in range(env.n_agents)]

episodes = 200

for ep in range(episodes):
    obs = env.reset()

    states = [preprocess_obs(o, env.grid_size) for o in obs]

    done = False
    total_rewards = [0] * env.n_agents

    while not done:
        actions = [
            agents[i].select_action(states[i])
            for i in range(env.n_agents)
        ]

        next_obs, rewards, done, _ = env.step(actions)
        next_states = [preprocess_obs(o, env.grid_size) for o in next_obs]

        for i in range(env.n_agents):
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

    # update target networks occasionally
    if ep % 10 == 0:
        for agent in agents:
            agent.update_target()

    print(f"Episode {ep} | Rewards: {total_rewards}")
def preprocess_obs(obs, grid_size):
    grid = obs["grid"].flatten()
    x, y = obs["position"]

    pos = np.array([x / grid_size, y / grid_size])

    return np.concatenate([grid, pos])
