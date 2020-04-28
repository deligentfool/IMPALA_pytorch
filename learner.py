import impala_pb2
import impala_pb2_grpc
from model import actor_critic_agent
from buffer import buffer
from concurrent import futures
import json
from torch.utils.tensorboard import SummaryWriter
import pickle
import gym
import grpc
import torch


class learner(impala_pb2_grpc.IMPALAServicer):
    def __init__(self, env, learning_rate=1e-3, rho=1, c=1, gamma=0.99, entropy_weight=0.05, batch_size=32, capacity=1000, value_weight=0.5, save_freq=20, log=False, device='cpu'):
        self.buffer = buffer(capacity)
        self.agent = actor_critic_agent(env, self.buffer, learning_rate, rho, c, gamma, entropy_weight, value_weight, device)
        self.batch_size = batch_size
        self.save_freq = save_freq
        self.log = log
        self.writer = SummaryWriter('./log/learner')

    def train(self):
        train_step = 0
        while True:
            if len(self.buffer) >= self.batch_size:
                value_loss, policy_loss, loss = self.agent.train(self.batch_size)
                train_step += 1
                self.writer.add_scalar('value_loss', value_loss, train_step)
                self.writer.add_scalar('policy_loss', policy_loss, train_step)
                self.writer.add_scalar('loss', loss, train_step)
                print('train_step: {}  value_loss: {:.4f}  policy_loss: {:.4f}  loss: {:.4f}  buffer: {}'.format(train_step, value_loss, policy_loss, loss, len(self.buffer)))
                if train_step % self.save_freq == 0:
                    torch.save(self.agent.net, './model/model.pkl')


    def get_trajectory(self, request, context):
        traj_data = json.loads(request.trajectory)
        self.buffer.store(
            obs=traj_data['observations'],
            act=traj_data['actions'],
            rew=traj_data['rewards'],
            don=traj_data['dones'],
            pol=traj_data['behavior_policies']
        )
        return impala_pb2.TrajectoryResponse(message='from server data')

    def send_parameter(self, request, context):
        model_parameter = self.agent.net.state_dict()
        model_parameter = pickle.dumps(model_parameter)
        return impala_pb2.ParameterResponse(message=model_parameter)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = gym.make('CartPole-v0')
    learner_instance = learner(
        env=env,
        learning_rate=1e-3,
        rho=1.0,
        c=1.0,
        gamma=0.99,
        entropy_weight=0.001,
        batch_size=32,
        capacity=1000,
        save_freq=30,
        value_weight=0.5,
        log=False,
        device=device
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    impala_pb2_grpc.add_IMPALAServicer_to_server(learner_instance, server)
    server.add_insecure_port('localhost:43231')

    server.start()
    learner_instance.train()
    server.wait_for_termination()

