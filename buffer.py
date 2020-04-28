import numpy as np
import random
from collections import deque, namedtuple
import json
from NumpyEncoder import NumpyEncoder
import copy
import threading

class buffer(object):
    def __init__(self, capacity=None):
        self.capacity = capacity
        self.lock = threading.Lock()
        if self.capacity is not None:
            self.observations = deque(maxlen=self.capacity)
            self.actions = deque(maxlen=self.capacity)
            self.rewards = deque(maxlen=self.capacity)
            self.next_observations = deque(maxlen=self.capacity)
            self.dones = deque(maxlen=self.capacity)
            self.behavior_policies = deque(maxlen=self.capacity)
        else:
            self.observations = deque()
            self.actions = deque()
            self.rewards = deque()
            self.next_observations = deque()
            self.dones = deque()
            self.behavior_policies = deque()

    def store(self, obs, act, rew, next_obs, don, pol):
        self.lock.acquire()
        self.observations.append(obs)
        self.actions.append(act)
        self.rewards.append(rew)
        self.next_observations.append(next_obs)
        self.dones.append(don)
        self.behavior_policies.append(pol)
        self.lock.release()

    def get_data(self, batch_size=None):
        self.lock.acquire()
        if batch_size is not None:
            observations = [self.observations.popleft() for _ in range(batch_size)]
            actions = [self.actions.popleft() for _ in range(batch_size)]
            rewards = [self.rewards.popleft() for _ in range(batch_size)]
            next_observations = [self.next_observations.popleft() for _ in range(batch_size)]
            dones = [self.dones.popleft() for _ in range(batch_size)]
            behavior_policies = [self.behavior_policies.popleft() for _ in range(batch_size)]
        else:
            observations = copy.deepcopy(list(self.observations))
            actions = copy.deepcopy(list(self.actions))
            rewards = copy.deepcopy(list(self.rewards))
            next_observations = copy.deepcopy(list(self.next_observations))
            dones = copy.deepcopy(list(self.dones))
            behavior_policies = copy.deepcopy(list(self.behavior_policies))
            self.clear()

        traj_data = namedtuple('traj_data', ['observations', 'actions', 'rewards', 'next_observations', 'dones', 'behavior_policies'])(observations, actions, rewards, next_observations, dones, behavior_policies)
        self.lock.release()
        return traj_data

    def get_json_data(self, batch_size=None):
        traj_data = self.get_data(batch_size)
        traj_data = traj_data._asdict()
        json_data = json.dumps(traj_data, cls=NumpyEncoder)
        return json_data

    def clear(self):
        self.observations.clear()
        self.rewards.clear()
        self.actions.clear()
        self.next_observations.clear()
        self.dones.clear()
        self.behavior_policies.clear()

    def __len__(self):
        return len(self.dones)