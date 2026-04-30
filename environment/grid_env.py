import numpy as np
import random

class GridSearchRescueEnv:
    def __init__(
        self,
        grid_size=8,
        n_agents=2,
        n_victims=3,
        n_obstacles=5,
        max_steps=100,
        seed=None
    ):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_victims = n_victims
        self.n_obstacles = n_obstacles
        self.max_steps = max_steps

        self.pheromone_decay = 0.95
        self.pheromone_deposit = 1.0

        if seed:
            np.random.seed(seed)
            random.seed(seed)

        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.pheromone = np.zeros((self.grid_size, self.grid_size), dtype=float)

        self._place_items(1, self.n_obstacles)
        self._place_items(2, self.n_victims)

        self.agent_positions = []
        for _ in range(self.n_agents):
            self.agent_positions.append(self._get_empty_cell())

        self.steps = 0
        self.rescued = 0

        return self._get_obs()

    def _place_items(self, item_type, count):
        for _ in range(count):
            x, y = self._get_empty_cell()
            self.grid[x, y] = item_type

    def _get_empty_cell(self):
        while True:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            if self.grid[x, y] == 0:
                return (x, y)

    def step(self, actions):
        rewards = [0] * self.n_agents
        new_positions = []

        for i, action in enumerate(actions):
            x, y = self.agent_positions[i]

            if action == 1: x -= 1
            elif action == 2: x += 1
            elif action == 3: y -= 1
            elif action == 4: y += 1

            x = np.clip(x, 0, self.grid_size - 1)
            y = np.clip(y, 0, self.grid_size - 1)

            if self.grid[x, y] == 1:
                x, y = self.agent_positions[i]

            new_positions.append((x, y))

        self.agent_positions = new_positions

        # pheromone deposit
        for (x, y) in self.agent_positions:
            self.pheromone[x, y] += self.pheromone_deposit

        # decay
        self.pheromone *= self.pheromone_decay

        # rescue + reward
        for i, (x, y) in enumerate(self.agent_positions):
            if self.grid[x, y] == 2:
                self.grid[x, y] = 0
                rewards[i] += 10
                self.rescued += 1

            # pheromone shaping
            rewards[i] += 0.01 * self.pheromone[x, y]

        self.steps += 1

        done = (
            self.rescued == self.n_victims or
            self.steps >= self.max_steps
        )

        return self._get_obs(), rewards, done, {}

    def _get_obs(self):
        obs = []
        for pos in self.agent_positions:
            obs.append({
                "grid": self.grid.copy(),
                "pheromone": self.pheromone.copy(),
                "position": pos
            })
        return obs

    def render(self):
        display = np.array(self.grid, dtype=object)

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.pheromone[x, y] > 0.5:
                    display[x, y] = "*"

        for x, y in self.agent_positions:
            display[x, y] = "A"

        display = display.astype(str)
        display[display == "0"] = "."
        display[display == "1"] = "#"
        display[display == "2"] = "V"

        print("\n".join([" ".join(row) for row in display]))
        print()
