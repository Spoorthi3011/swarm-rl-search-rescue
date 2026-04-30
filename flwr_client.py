import flwr as fl
import torch


class DQNClient(fl.client.NumPyClient):
    def __init__(self, agent):
        self.agent = agent

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.agent.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.agent.model.state_dict()
        new_state = {}

        for (key, _), val in zip(state_dict.items(), parameters):
            new_state[key] = torch.tensor(val)

        self.agent.model.load_state_dict(new_state)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Training already happens locally in your loop
        return self.get_parameters(config), 1, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, 1, {}
