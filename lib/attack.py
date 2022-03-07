from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn


class Attack(metaclass=ABCMeta):
    def __init__(self, model: nn.Module):
        self.model = model

    @abstractmethod
    def __call__(self, tensor: torch.Tensor, *args, **kwargs):
        pass
