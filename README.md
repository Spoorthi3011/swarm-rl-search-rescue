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
├── agents/          # decentralised agent policies, Q-networks
├── environment/     # custom rescue grid environment
├── swarm/           # PSO and pheromone coordination modules
├── training/        # training loops, federated server
├── experiments/     # configs, results, benchmark runs
├── config.py        # global hyperparameters and settings
├── main.py          # entry point
├── requirements.txt
└── README.md
```

---

## How to Run
```bash
git clone https://github.com/Spoorthi3011/swarm-marl-search-rescue
cd swarm-marl-search-rescue
pip install -r requirements.txt

# Run main entry point
python main.py --rounds 50 --strategy fedavg --swarm pso

# Or launch agents individually
python agents/agent.py --id 0 --env rescue-v1
python agents/agent.py --id 1 --env rescue-v1
```

---

## Roadmap

- [x] Repository structure and documentation
- [x] Environment design (custom rescue grid)
- [ ] IQL baseline implementation
- [ ] PSO-based swarm coordination module
- [ ] Pheromone stigmergy integration
- [ ] Flower (flwr) federated server setup
- [ ] Benchmark experiments and results
- [ ] Paper writeup

---

## Related Work

> **Meta-Modeling with Drug Discovery Stack Regressor** — Spoorthi Jolakula Suresh (First Author)
> *Biomedical Research International*, 2024
> DOI: [10.2174/0115701638405489251006073137](https://doi.org/10.2174/0115701638405489251006073137)

---

## License

MIT License — see [LICENSE](LICENSE)
```

---

**Three fixes made:**
1. Removed the stray ` ```docs(readme)... ``` ` commit message that was sitting in the middle of your README
2. Fixed `How to Run` — changed `src/server.py` and `src/agent.py` to match your actual folder structure (`agents/`, `training/`)
3. Added the **Roadmap** section we discussed

Commit message to use:
```
docs(readme): fix run commands, remove stray commit message, add roadmap
