
import numpy as np
import random

class GridSearchRescueEnv:
    def __init__(
        self,
        grid_size=10,
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

        if seed:
            np.random.seed(seed)
            random.seed(seed)

        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        self._place_items(1, self.n_obstacles)  # obstacles
        self._place_items(2, self.n_victims)    # victims

        self.agent_positions = []
        for _ in range(self.n_agents):
            pos = self._get_empty_cell()
            self.agent_positions.append(pos)

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
        """
        actions: list of actions for each agent
        0 = stay, 1 = up, 2 = down, 3 = left, 4 = right
        """
        rewards = [0] * self.n_agents

        new_positions = []

        for i, action in enumerate(actions):
            x, y = self.agent_positions[i]

            if action == 1:   # up
                x -= 1
            elif action == 2: # down
                x += 1
            elif action == 3: # left
                y -= 1
            elif action == 4: # right
                y += 1

            # boundary check
            x = np.clip(x, 0, self.grid_size - 1)
            y = np.clip(y, 0, self.grid_size - 1)

            # obstacle check
            if self.grid[x, y] == 1:
                x, y = self.agent_positions[i]  # revert

            new_positions.append((x, y))

        self.agent_positions = new_positions

        # check for rescues
        for i, (x, y) in enumerate(self.agent_positions):
            if self.grid[x, y] == 2:
                self.grid[x, y] = 0
                rewards[i] += 10
                self.rescued += 1

        self.steps += 1

        done = (
            self.rescued == self.n_victims or
            self.steps >= self.max_steps
        )

        return self._get_obs(), rewards, done, {}

    def _get_obs(self):
        """
        Return observation per agent:
        (grid, agent position)
        """
        obs = []
        for pos in self.agent_positions:
            obs.append({
                "grid": self.grid.copy(),
                "position": pos
            })
        return obs

    def render(self):
        display = np.array(self.grid, dtype=str)

        for x, y in self.agent_positions:
            display[x, y] = "A"

        display[display == "0"] = "."
        display[display == "1"] = "#"
        display[display == "2"] = "V"

        print("\n".join([" ".join(row) for row in display]))
        print()
