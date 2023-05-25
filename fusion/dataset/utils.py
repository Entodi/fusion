import numpy as np
import random
import torch


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
