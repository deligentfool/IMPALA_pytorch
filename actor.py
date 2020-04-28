import impala_pb2
import impala_pb2_grpc
import grpc
from model import actor_critic_agent
from buffer import buffer
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import gym
import pickle
import torch
import numpy as np


def send_trajectory(channel, trajectory):
    stub = impala_pb2_grpc.IMPALAStub(channel)
    response = stub.get_trajectory(impala_pb2.TrajectoryRequest(trajectory=trajectory))
    return response.message

def get_parameter(channel):
    stub = impala_pb2_grpc.IMPALAStub(channel)
    response = stub.send_parameter(impala_pb2.ParameterRequest(parameter='request from actor'))
    return response.message

def actor_run(actor_id, env_id):
    episode = 0
    weight_reward = None
    env = gym.make(env_id)
    actor_buffer = buffer()
    agent = actor_critic_agent(env, actor_buffer)
    writer = SummaryWriter('./log/actor_{}'.format(actor_id))
    channel = grpc.insecure_channel('localhost:50051')

    params = get_parameter(channel)
    params = pickle.loads(params)
    agent.load_state_dict(params)

    while True:
        obs = env.reset()
        total_reward = 0
        while True:
            action = agent.net.act(torch.FloatTensor(np.expand_dims(obs, 0))).item()
            behavior_policy, _ = agent.net.forward(torch.FloatTensor(np.expand_dims(obs, 0)))
            next_obs, reward, done, info = env.step(action)
            actor_buffer.store(obs, action, reward, next_obs, done, behavior_policy.squeeze(0).detach().numpy())
            total_reward += reward
            obs = next_obs
            if done:
                if weight_reward:
                    weight_reward = 0.99 * weight_reward + 0.01 * total_reward
                else:
                    weight_reward = total_reward
                episode += 1
                print('episode: {}  weight_reward: {:.2f}  reward: {:.2f}'.format(episode, weight_reward, total_reward))
            if len(actor_buffer) == 100:
                traj_data = actor_buffer.get_json_data()
                send_trajectory(channel, trajectory=traj_data)
                params = get_parameter(channel)
                params = pickle.loads(params)
                agent.load_state_dict(params)
            if done:
                break


if __name__ == '__main__':
    '''
    num_cpu = 1
    env_id = 'CartPole-v0'
    process = [mp.Process(target=actor_run, args=(i, env_id)) for i in range(num_cpu)]
    [p.start() for p in process]
    '''
    actor_run(0, 'CartPole-v0')