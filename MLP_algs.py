import torch
from torch import nn
import numpy as np
from typing import List

# Base NN Class
class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: List[int]):
        super().__init__()
        self.in_channels = input_size
        self.hidden_channels = hidden_size

        layers = [nn.Linear(self.in_channels, self.hidden_channels[0]), nn.SiLU()]
        for i in range(len(self.hidden_channels) - 1):
            layers.append(nn.Linear(self.hidden_channels[i], self.hidden_channels[i+1]))
            layers.append(nn.SiLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



class MLP_actor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device =                       config["device"]
        self.n_multirotor =                 config["n_holonomic"]
        self.n_fixedwing =                  config["n_fixedwing"]
        self.n_agents = self.n_multirotor + self.n_fixedwing
        self.observation_dim_per_agent =    config["observation_dim_per_agent"]
        self.action_dim_per_agent =         config["action_dim_per_agent"]
        self.r_communication =              config["r_communication"]
        self.batch_size =                   config["batch_size"]
        self.n_envs =                       config["n_envs"]
        self.state_dim_per_agent =          config["state_dim_per_agent"]

        self.policy = MLP(
            input_size=self.state_dim_per_agent * self.n_agents,
            hidden_size=[self.action_dim_per_agent * self.n_agents]
        ).to(self.device)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)

        return self.policy(x)

class MLP_critic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.n_agents = config["n_agents"]
        self.observation_dim_per_agent = config["observation_dim_per_agent"]
        self.action_dim_per_agent = config["action_dim_per_agent"]

        self.q_value = MLP(
            input_size=(self.observation_dim_per_agent + self.action_dim_per_agent) * self.na,
            hidden_size=[(self.observation_dim_per_agent + self.action_dim_per_agent) * 2 * self.na,
                             (self.observation_dim_per_agent + self.action_dim_per_agent) * self.na,
                             (self.observation_dim_per_agent + self.action_dim_per_agent),
                             1]
        ).to(self.device)

    def forward(self, observation, action):
        return self.q_value(torch.cat((
            observation.reshape([observation.shape[0], -1]),
            action.reshape([action.shape[0], -1])), dim=1))