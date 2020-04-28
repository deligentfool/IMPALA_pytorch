import torch
import torch.nn as nn
import torch.nn.functional as F
from buffer import buffer
import numpy as np
from net import actor_critic_net


class actor_critic_agent(object):
    def __init__(self, env, buffer, learning_rate=1e-3, n_step=2, rho=1, c=1, gamma=0.99, entropy_weight=0.05):
        self.env = env
        self.learning_rate = learning_rate
        self.n_step = n_step
        self.rho = rho
        self.c = c
        self.gamma = gamma
        self.entropy_weight = entropy_weight

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.net = actor_critic_net(self.observation_dim, self.action_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.buffer = buffer
        self.weight_reward = None

    def train(self, batch_size):
        data = self.buffer.get_data(batch_size)
        observations = torch.FloatTensor(data.observations)
        rewards = torch.FloatTensor(data.rewards).unsqueeze(1)
        actions = torch.LongTensor(data.actions).unsqueeze(1)
        next_observations = torch.FloatTensor(data.next_observations)
        dones = torch.FloatTensor(data.dones).unsqueeze(1)
        behavior_policies = torch.FloatTensor(data.behavior_policies)

        target_policies, values = self.net.forward(observations)
        target_policies_action = target_policies.gather(1, actions)
        behavior_policies_action = behavior_policies.gather(1, actions)

        observations_list = self.split(observations)
        rewards_list = self.split(rewards)
        actions_list = self.split(actions)
        next_observations_list = self.split(next_observations)
        dones_list = self.split(dones)
        behavior_policies_action_list = self.split(behavior_policies_action)
        target_policies_action_list = self.split(target_policies_action)
        target_policies_list = self.split(target_policies)
        values_list = self.split(values)

        vs = 0
        for i in reversed(range(self.n_step)):
            rho = torch.clamp_max((target_policies_action_list[i].log() - behavior_policies_action_list[i].log()).exp(), self.rho)
            if i >= 1:
                c = torch.clamp_max((target_policies_action_list[i - 1].log() - behavior_policies_action_list[i - 1].log()).exp(), self.c)
            else:
                c = 1
            delta = rho * (rewards_list[i] + self.gamma * (1 - dones_list[i]) * values_list[i + 1] - values_list[i])
            vs += delta
            vs = vs * c * self.gamma
        vs = vs / self.gamma + values_list[0]

        vs_1 = 0
        for i in reversed(range(1, self.n_step + 1)):
            rho = torch.clamp_max((target_policies_action_list[i].log() - behavior_policies_action_list[i].log()).exp(), self.rho)
            if i >= 2:
                c = torch.clamp_max((target_policies_action_list[i - 1].log() - behavior_policies_action_list[i - 1].log()).exp(), self.c)
            else:
                c = 1
            delta = rho * (rewards_list[i] + self.gamma * (1 - dones_list[i]) * values_list[i + 1] - values_list[i])
            vs_1 += delta
            vs_1 = vs_1 * c * self.gamma
        vs_1 = vs_1 / self.gamma + values_list[1]

        dist = torch.distributions.Categorical(target_policies_list[0])
        entropies = dist.entropy().unsqueeze(1)
        value_loss = (vs.detach() - values_list[0]).pow(2).sum()
        delta = rewards_list[0] + vs_1 * self.gamma * (1 - dones_list[0]) - values_list[0]
        policy_loss = - (target_policies_action_list[0]).log() * delta.detach() - self.entropy_weight * entropies
        policy_loss = policy_loss.sum()
        loss = value_loss + policy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return value_loss.item(), policy_loss.item(), loss.item()

    def split(self, data):
        results = []
        length = len(data)
        for i in range(self.n_step + 2):
            results.append(data[i: length - self.n_step + i - 1])
        return results

    def load_state_dict(self, params):
        self.net.load_state_dict(params)

    def run(self):
        obs = self.env.reset()
        total_reward = 0
        while True:
            action = self.net.act(torch.FloatTensor(np.expand_dims(obs, 0))).item()
            behavior_policy, _ = self.net.forward(torch.FloatTensor(np.expand_dims(obs, 0)))
            next_obs, reward, done, info = self.env.step(action)
            self.buffer.store(obs, action, reward, next_obs, done, behavior_policy.squeeze(0).detach().numpy())
            total_reward += reward
            obs = next_obs
            if done:
                if self.weight_reward:
                    self.weight_reward = 0.99 * self.weight_reward + 0.01 * total_reward
                else:
                    self.weight_reward = total_reward
                break
        return self.weight_reward, total_reward