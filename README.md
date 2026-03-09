# Swarm-Based Multi-Agent Reinforcement Learning for Search & Rescue

> Decentralised multi-agent RL system with swarm-inspired coordination (PSO, 
> pheromone models) and federated learning via Flower (flwr) for autonomous 
> search-and-rescue scenarios.

![Status](https://img.shields.io/badge/Status-Under%20Development-orange?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Flower](https://img.shields.io/badge/Flower%20flwr-Federated%20Learning-6A1B9A?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

This project investigates whether bio-inspired swarm coordination — specifically 
Particle Swarm Optimisation (PSO) and pheromone-based stigmergy — can improve 
decentralised policy learning in multi-agent reinforcement learning (MARL) for 
search-and-rescue environments.

Agents learn independently without a central controller. Swarm mechanisms 
guide emergent coordination, while Flower (flwr) enables federated model 
aggregation across agents without sharing raw experience data. OpenMP is used 
for parallel environment simulation.

**Research Question:** Does swarm-inspired coordination improve rescue rate, 
convergence speed, and robustness compared to independent learners?

---

## Key Techniques

| Component | Detail |
|---|---|
| MARL Algorithm | QMIX / Independent Q-Learning (IQL) baseline |
| Swarm Coordination | PSO-based policy guidance, pheromone stigmergy |
| Federated Learning | Flower (flwr) — FedAvg aggregation |
| Parallelisation | OpenMP multi-environment simulation |
| Framework | PyTorch |
| Environment | Custom grid-world rescue environment (Gym-compatible) |

---

## Results

| Configuration | Rescue Rate | Mean Steps | Convergence (rounds) |
|---|---|---|---|
| Centralised baseline | [XX]% | [XX] | [XX] |
| Independent learners (IQL) | [XX]% | [XX] | [XX] |
| Swarm-guided MARL | [XX]% | [XX] | [XX] |
| **Swarm MARL + Federated** | **[XX]%** | **[XX]** | **[XX]** |

---

## Folder Structure
```
swarm-marl-search-rescue/
├── src/
│   ├── agent.py         # decentralised agent, local Q-network
│   ├── server.py        # Flower federated server
│   ├── environment.py   # custom rescue grid environment
│   ├── swarm.py         # PSO and pheromone coordination
│   └── utils.py         # reward shaping, logging
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_training.ipynb
│   └── 03_results_analysis.ipynb
├── data/                # environment configs, replay buffers
├── models/              # agent checkpoints
├── results/             # rescue metrics, convergence plots
├── requirements.txt
└── README.md
```

---

## How to Run
```bash
git clone https://github.com/Spoorthi3011/swarm-marl-search-rescue
cd swarm-marl-search-rescue
pip install -r requirements.txt

# Start federated server
python src/server.py --rounds 50 --strategy fedavg

# Launch agents (run in separate terminals or via script)
python src/agent.py --id 0 --env rescue-v1 --swarm pso
python src/agent.py --id 1 --env rescue-v1 --swarm pso
```

---

## Related Work

This project builds on federated and privacy-preserving ML concepts explored in:

> **Meta-Modeling with Drug Discovery Stack Regressor** — Spoorthi Jolakula Suresh (First Author)  
> *Biomedical Research International*, 2024  
> DOI: [10.2174/0115701638405489251006073137](https://doi.org/10.2174/0115701638405489251006073137)

---

## License

MIT License — see [LICENSE](LICENSE)
