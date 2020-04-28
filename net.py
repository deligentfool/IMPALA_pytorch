import torch
import torch.nn as nn
import torch.nn.functional as F


class actor_critic_net(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(actor_critic_net, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.feature_layer = nn.Sequential(
            nn.Linear(self.observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.policy_layer = nn.Linear(128, self.action_dim)
        self.value_layer = nn.Linear(128, 1)

    def forward(self, observation):
        feature = self.feature_layer(observation)
        policy = self.policy_layer(feature)
        policy = F.softmax(policy, -1)
        value = self.value_layer(feature)
        return policy, value

    def act(self, observation):
        policy, value = self.forward(observation)
        dist = torch.distributions.Categorical(policy)
        action = dist.sample()
        return action