
import numpy as np
import flwr as fl
from environment.grid_env import GridSearchRescueEnv
from agents.dqn_agent import DQNAgent
from agents.replay_buffer import ReplayBuffer
from federated.flwr_client import DQNClient


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
# Federated Helper
# =========================
def run_federated_round(agents, rounds=3):
    clients = [DQNClient(agent) for agent in agents]

    def client_fn(cid: str):
        return clients[int(cid)]

    strategy = fl.server.strategy.FedAvg()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(agents),
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )


# =========================
# Config
# =========================
GRID_SIZE = 8
N_AGENTS = 2
EPISODES = 200
MAX_STEPS = 100

STATE_DIM = (GRID_SIZE * GRID_SIZE) * 2 + 2
ACTION_DIM = 5

FED_INTERVAL = 10   # run federation every N episodes
FED_ROUNDS = 3


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

global_best_reward = -float("inf")
global_best_state = None


# =========================
# Training Loop
# =========================
for ep in range(EPISODES):
    obs = env.reset()
    states = [preprocess_obs(o, GRID_SIZE) for o in obs]

    done = False
    total_rewards = [0] * N_AGENTS

    while not done:
        actions = [
            agents[i].select_action(states[i])
            for i in range(N_AGENTS)
        ]

        next_obs, rewards, done, _ = env.step(actions)
        next_states = [preprocess_obs(o, GRID_SIZE) for o in next_obs]

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

    # =========================
    # PSO: Personal + Global Best
    # =========================
    for i in range(N_AGENTS):
        agents[i].update_personal_best(total_rewards[i])

    best_idx = np.argmax(total_rewards)

    if total_rewards[best_idx] > global_best_reward:
        global_best_reward = total_rewards[best_idx]
        global_best_state = {
            k: v.clone() for k, v in agents[best_idx].model.state_dict().items()
        }

    # Apply PSO update
    if ep % 5 == 0 and global_best_state is not None:
        for agent in agents:
            agent.pso_update(global_best_state)

    # =========================
    # Federated Learning Step
    # =========================
    if ep % FED_INTERVAL == 0 and ep > 0:
        print("🔄 Running Federated Aggregation...")
        run_federated_round(agents, rounds=FED_ROUNDS)

    # =========================
    # Target Network Update
    # =========================
    if ep % 10 == 0:
        for agent in agents:
            agent.update_target()

    # =========================
    # Logging
    # =========================
    print(
        f"Episode {ep:03d} | Rewards: {total_rewards} | "
        f"Global Best: {round(global_best_reward, 2)}"
    )

