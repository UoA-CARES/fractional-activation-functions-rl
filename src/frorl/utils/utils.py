from __future__ import annotations
from contextlib import contextmanager
import random
import numpy as np
import torch


def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def soft_update_params(source,target,tau:float):
    with torch.no_grad():
        for p,tp in zip(source.parameters(),target.parameters()):
            tp.data.mul_(1.0-tau).add_(tau*p.data)


@contextmanager
def evaluating(module):
    was_training=module.training
    module.eval()
    try: yield module
    finally: module.train(was_training)
