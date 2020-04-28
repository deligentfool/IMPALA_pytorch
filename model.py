import torch
import torch.nn as nn
import torch.nn.functional as F
from buffer import buffer
import numpy as np
from net import actor_critic_net


class actor_critic_agent(object):
    def __init__(self, env, buffer, learning_rate=1e-3, rho=1, c=1, gamma=0.99, entropy_weight=0.05, value_weight=0.5):
        self.env = env
        self.learning_rate = learning_rate
        self.rho = rho
        self.c = c
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.value_weight = value_weight

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.net = actor_critic_net(self.observation_dim, self.action_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.buffer = buffer

    def train(self, batch_size):
        data = self.buffer.get_data(batch_size)
        observations = torch.FloatTensor(data.observations)
        rewards = torch.FloatTensor(data.rewards).unsqueeze(-1)
        actions = torch.LongTensor(data.actions).unsqueeze(-1)
        next_observations = torch.FloatTensor(data.next_observations)
        dones = torch.FloatTensor(data.dones).unsqueeze(-1)
        behavior_policies = torch.FloatTensor(data.behavior_policies)

        target_policies, values = self.net.forward(observations)
        target_policies_action = target_policies.gather(2, actions)
        behavior_policies_action = behavior_policies.gather(2, actions)

        observations_list = self.split(observations)
        rewards_list = self.split(rewards)
        actions_list = self.split(actions)
        next_observations_list = self.split(next_observations)
        dones_list = self.split(dones)
        behavior_policies_action_list = self.split(behavior_policies_action)
        target_policies_action_list = self.split(target_policies_action)
        target_policies_list = self.split(target_policies)
        values_list = self.split(values)

        dist = torch.distributions.Categorical(target_policies_list[0])
        entropies = dist.entropy().unsqueeze(-1)

        rho_s = torch.exp(target_policies_action_list[0].log() - behavior_policies_action_list[0].log())
        clip_rho_s = torch.clamp_max(rho_s, self.rho)
        c_s = torch.clamp_max(rho_s, self.c)
        deltas = clip_rho_s * (rewards_list[0] + self.gamma * (1 - dones_list[0]) * values_list[1] - values_list[0])
        trajectory_size = rho_s.size(1)
        acc = 0
        vs_minus_v_xs = []
        for i in range(trajectory_size - 1, -1, -1):
            acc = deltas[:, i] + self.gamma * (1 - dones_list[0])[:, i] * c_s[:, i] * acc
            vs_minus_v_xs.append(acc.view(-1, 1))

        vs_minus_v_xs = torch.cat(vs_minus_v_xs[::-1], dim=1).unsqueeze(-1)
        vs = vs_minus_v_xs + values_list[0]

        rho_s_1 = torch.exp(target_policies_action_list[1].log() - behavior_policies_action_list[1].log())
        clip_rho_s_1 = torch.clamp_max(rho_s_1, self.rho)
        c_s_1 = torch.clamp_max(rho_s_1, self.c)
        deltas = clip_rho_s_1 * (rewards_list[1] + self.gamma * (1 - dones_list[1]) * values_list[2] - values_list[1])
        trajectory_size = rho_s_1.size(1)
        acc = 0
        vs_1_minus_v_xs_1 = []
        for i in range(trajectory_size - 1, -1, -1):
            acc = deltas[:, i] + self.gamma * (1 - dones_list[1])[:, i] * c_s_1[:, i] * acc
            vs_1_minus_v_xs_1.append(acc.view(-1, 1))

        vs_1_minus_v_xs_1 = torch.cat(vs_1_minus_v_xs_1[::-1], dim=1).unsqueeze(-1)
        vs_1 = vs_1_minus_v_xs_1 + values_list[1]

        advantage = clip_rho_s * (rewards_list[0] + self.gamma * (1 - dones_list[0]) * vs_1 - values_list[0])
        policy_loss = - target_policies_action_list[0].log() * advantage.detach()
        value_loss = (vs.detach() - values_list[0]).pow(2)
        loss = policy_loss + value_loss * self.value_weight - self.entropy_weight * entropies
        loss = loss.sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return value_loss.sum().item(), policy_loss.sum().item(), loss.item()

    def split(self, data):
        results = []
        length = data.size(1)
        for i in range(3):
            results.append(data[:, i: length + i - 2])
        return results

    def load_state_dict(self, params):
        self.net.load_state_dict(params)
