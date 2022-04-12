import ray 
import torch
import numpy as np
import gym
from torch import optim
from torch.nn import functional as F
from networks import ActorCritic

@ray.remote
class Worker:
    def __init__(self, worker_id, param_server, updates_counter, writer, evaluation_worker):
        self.env = gym.make("CartPole-v1")
        self.worker_model = ActorCritic()
        #self.optimizer = optim.Adam(lr=1e-4, params=worker_model.parameters.remote())
        self.updates_counter = updates_counter
        #self.worker_model = worker_model
        self.param_server = param_server
        self.writer = writer
        self.evaluation_worker = evaluation_worker
        
        # variables per episode
        self.values = []
        self.logprobs = []
        self.rewards = []
        self.actor_loss = 0
        self.critic_loss = 0
        self.len = 0
        self.G = torch.Tensor([0])
        
        # Learning parameters
        self.gamma = 0.95
        self.clc = 0.1
        self.epochs = 2000
        self.n_steps = 500
        

    def run_episode(self):
        # Set the weights of the internal worker model for this episode
        self.worker_model.set_weights(ray.get(self.param_server.get_weights.remote()))

        state = self.env.reset()
        state = torch.from_numpy(state).float()
        values, logprobs, rewards = [], [], []
        done = False
        j = 0
        self.G=torch.Tensor([0])
        while ( j < self.n_steps and done == False ):
            j += 1
            #policy, value = ray.get(self.worker_model.remote(state))
            policy, value = self.worker_model(state) 
            values.append(value) 
            logits = policy.view(-1)
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()
            logprob_  = logits[action]
            logprobs.append(logprob_)
            state_, _, done, info = self.env.step(action.detach().numpy())
            state  = torch.from_numpy(state_).float()
            if done:
                reward = -10
            else:
                reward = 1.0
                self.G = value.detach()
            rewards.append(reward)
        self.values = values 
        self.logprobs = logprobs
        self.rewards = rewards
    
    def update_params(self):
        rewards = torch.Tensor(self.rewards).flip(dims=(0,)).view(-1)
        logprobs = torch.stack(self.logprobs).flip(dims=(0,)).view(-1)
        values = torch.stack(self.values).flip(dims=(0,)).view(-1)
        Returns = []
        ret_ = self.G
        for r in range(rewards.shape[0]):
            ret_ = rewards[r] + self.gamma * ret_
            Returns.append(ret_)
        Returns = torch.stack(Returns).view(-1)
        Returns = F.normalize(Returns, dim=0)
        actor_loss = -1 * logprobs * (Returns - values.detach())
        critic_loss = torch.pow(values - Returns, 2)
        loss = actor_loss.sum() + self.clc * critic_loss.sum()
        loss.backward()
        #self.optimizer.step()
        
        self.actor_loss = actor_loss
        self.critic_loss = critic_loss
        self.len = len(rewards)
        print(f'Episode Len: {self.len}')
        return self.worker_model.get_gradients()


    def run(self):
        for i in range(self.epochs):
            #self.optimizer.zero_grad()
            self.run_episode()
            gradients = self.update_params()
            self.param_server.apply_gradients.remote(gradients)
            counter = ray.get(self.updates_counter.increment.remote())
            self.writer.add_scalar.remote('Episode_len', self.len, counter)
            if (counter % 100 == 0):
                self.evaluation_worker.run.remote(counter=counter)

@ray.remote
class EvaluationWorker:
    def __init__(self, param_server, updates_counter, writer, n_episodes=1, render=True):
        self.env = gym.make("CartPole-v1")
        self.worker_model = ActorCritic()
        self.worker_model.eval()
        self.updates_counter = updates_counter
        self.writer = writer
        self.render = render

        self.param_server = param_server
        
        # variables per episode
        self.values = []
        self.logprobs = []
        self.rewards = []
        self.len = 0
        
        # Learning parameters
        self.n_episodes = n_episodes

    def run_episode(self):

        state = self.env.reset()
        if self.render:
            self.env.render()
        state = torch.from_numpy(state).float()
        values, logprobs, rewards = [], [], []
        done = False
        j = 0
        while not done:
            j += 1
            #policy, value = ray.get(self.worker_model.remote(state))
            policy, value = self.worker_model(state) 
            values.append(value) 
            logits = policy.view(-1)
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()
            logprob_  = logits[action]
            logprobs.append(logprob_)
            state_, _, done, info = self.env.step(action.detach().numpy())
            state  = torch.from_numpy(state_).float()
            if self.render:
                self.env.render()
            if done:
                reward = -10
            else:
                reward = 1.0
            rewards.append(reward)
        self.values = values 
        self.logprobs = logprobs
        self.len = len(rewards)

    def run(self, counter):
        evaluation = 0 
        # Set the weights of the internal worker model for this episode
        self.worker_model.set_weights(ray.get(self.param_server.get_weights.remote()))

        for i in range(self.n_episodes):
            self.run_episode()
            evaluation += 1
            self.writer.add_scalar.remote(
                    'Evaluation/EpLen',
                    self.len,
                    counter 
                    )

 
