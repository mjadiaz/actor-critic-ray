import ray

from networks import ActorCritic
from torch import optim
import numpy as np

from torch.utils.tensorboard import SummaryWriter


@ray.remote
class Writer:
    def __init__(self, run_name):
        self.run_name = run_name
        self.writer  = SummaryWriter(self.run_name)

    def add_scalar(self, name, value, n_iter):
        self.writer.add_scalar(name, value, n_iter)


@ray.remote
class Counter:
    def __init__(self):
        self.counter = 0

    def increment(self):
        self.counter += 1
        return self.counter

    def get_counter(self):
        return self.counter


@ray.remote
class ParameterServer:
    def __init__(self, lr=1e-4):
        self.model = ActorCritic()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def apply_gradients(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        #return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()
