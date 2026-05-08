# 🐜 Swarm-Based Multi-Agent Reinforcement Learning for Search & Rescue

A decentralized **Multi-Agent Reinforcement Learning (MARL)** framework for autonomous search-and-rescue operations using **swarm intelligence** and **federated learning**.

This project explores how biologically inspired coordination mechanisms — specifically **Particle Swarm Optimization (PSO)** and **pheromone-based stigmergy** — can improve cooperation, convergence speed, and rescue efficiency in distributed reinforcement learning systems.

---

# 📌 Overview

Traditional centralized MARL systems often suffer from:

- High communication overhead
- Poor scalability
- Single points of failure
- Privacy concerns

This project proposes a **fully decentralized swarm-based MARL architecture** where agents coordinate indirectly using environmental signals and collaborative policy sharing.

The system combines:

- 🧠 Deep Q-Network (DQN) agents
- 🐜 Pheromone-based stigmergy
- 🌐 Federated learning with Flower (FedAvg)
- ⚡ PSO-inspired policy optimization

to create emergent cooperative behavior without requiring centralized control.

---

# 🚀 Features

- ✅ Custom grid-based search & rescue simulation
- ✅ Independent DQN agents implemented in PyTorch
- ✅ Pheromone trail communication (stigmergy)
- ✅ PSO-inspired swarm policy sharing
- ✅ Federated learning using Flower (`flwr`)
- ✅ Decentralized coordination architecture
- ✅ Reward tracking and visualization
- ✅ Emergent cooperative exploration behavior
- ✅ Privacy-preserving distributed learning

---

# 🧠 Research Objective

The primary objective of this research is to evaluate whether swarm-inspired coordination improves:

- 📈 Learning convergence speed
- 🎯 Rescue success rate
- 🤝 Multi-agent coordination
- 🛡️ Robustness in decentralized environments

without requiring:

- centralized control
- direct communication
- raw experience sharing

---

# 🏗️ System Architecture

```plaintext
+-------------------+
|   DQN Agents      |
+-------------------+
          |
          v
+-------------------+
| Environment       |
| Interaction       |
+-------------------+
          |
          v
+-------------------+
| Pheromone Trails  |
| (Stigmergy)       |
+-------------------+
          |
          v
+-------------------+
| PSO Policy        |
| Sharing           |
+-------------------+
          |
          v
+-------------------+
| Federated         |
| Aggregation       |
| (Flower FedAvg)   |
+-------------------+
```

---

# ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Reinforcement Learning | PyTorch |
| Federated Learning | Flower (`flwr`) |
| Environment Simulation | NumPy |
| Visualization | Matplotlib |

---

# 📁 Project Structure

```plaintext
swarm-rl-search-rescue/
│
├── agents/
│   ├── dqn_agent.py
│   └── replay_buffer.py
│
├── environment/
│   └── grid_env.py
│
├── flwr_client.py
├── train.py
├── main.py
├── plot_rewards.py
├── requirements.txt
└── README.md
```

---

# 🧪 Environment Design

The environment is a grid-based disaster simulation containing:

| Symbol | Meaning |
|---|---|
| `A` | Agents |
| `V` | Victims |
| `#` | Obstacles |
| `*` | Pheromone trails |

Agents must:

- Explore unknown regions
- Locate victims
- Avoid obstacles
- Coordinate efficiently
- Maximize rescue rewards

---

# 🐜 Swarm Intelligence Components

## 1️⃣ Pheromone-Based Stigmergy

Agents communicate indirectly through environmental modifications.

### Mechanism

- Agents deposit pheromone trails while exploring
- Trails decay over time
- Other agents observe pheromone intensity
- Coordination emerges without explicit communication

### Benefits

- Distributed coordination
- Scalable communication
- Reduced overhead
- Emergent exploration behavior

---

## 2️⃣ PSO-Inspired Policy Sharing

Each agent maintains:

- 🧠 Personal best policy (`pBest`)
- 🌍 Global best policy (`gBest`)

Policy updates are influenced by swarm-wide performance:

```python
new_policy = (
    w * current_policy
    + c1 * personal_best
    + c2 * global_best
)
```

Where:

- `w` → inertia weight
- `c1` → cognitive coefficient
- `c2` → social coefficient

### Benefits

- Faster convergence
- Improved exploration-exploitation balance
- Knowledge sharing across agents

---

# 🔐 Federated Learning

Federated learning is implemented using **Flower (`flwr`)**.

## Workflow

1. Agents train locally on their environments
2. Only model parameters are shared
3. Raw experiences remain private
4. Flower performs Federated Averaging (FedAvg)
5. Aggregated policies are redistributed

### Advantages

- Privacy-preserving learning
- Reduced communication costs
- Decentralized optimization
- Scalable distributed training

---

# 📊 Experimental Results

Training experiments demonstrated:

- 📈 Increasing cumulative rewards
- 🤝 Emergent cooperative behavior
- 🎯 Successful victim rescues
- ⚡ Faster convergence with swarm coordination

### Example Training Output

```plaintext
Episode 050 | Rewards: [18.2, 21.4] | Global Best: 36.15
```

---

# ▶️ Installation

## Clone Repository

```bash
git clone https://github.com/Spoorthi3011/swarm-rl-search-rescue.git
cd swarm-rl-search-rescue
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch flwr matplotlib numpy
```

---

# ▶️ Running the Project

## Run Environment Simulation

```bash
python main.py
```

## Run Training

```bash
python train.py
```

## Plot Reward Curves

```bash
python plot_rewards.py
```

---

# 📈 Example Output

```plaintext
Episode 050 | Rewards: [18.2, 21.4] | Global Best Reward: 36.15
```

---

# 🎯 Research Contributions

This work contributes:

- 🐜 Hybrid swarm intelligence + MARL architecture
- 🌐 Decentralized coordination without centralized control
- 🔐 Privacy-preserving federated reinforcement learning
- 🤝 Emergent cooperative rescue behavior
- ⚡ Scalable distributed learning framework

---

# 🚀 Future Improvements

Planned enhancements include:

- PPO-based multi-agent architectures
- OpenMP / multiprocessing parallel simulations
- Larger swarm-scale experiments
- Dynamic disaster environments
- Real-time visualization dashboard
- Communication-efficient federated optimization
- Attention-based agent coordination

---

# 👩‍💻 Author

**Spoorthi**  
Swarm-Based MARL Research Project  
2026

---

# 📜 License

This project is licensed under the **MIT License**.

---

# ⭐ Acknowledgements

- PyTorch
- Flower Federated Learning Framework
- OpenAI Reinforcement Learning Research
- Swarm Intelligence & Multi-Agent Systems Research Community

---

# 🌟 If You Like This Project

Give this repository a ⭐ on GitHub and support decentralized AI research!
