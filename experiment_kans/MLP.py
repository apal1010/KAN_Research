import torch
from torch import nn
from typing import Callable, Dict, Tuple

class MLP(nn.Module):
    def __init__(self, layers: Tuple[int], degree = None, device: str = 'cpu'):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i+1], device=device) for i in range(len(layers) - 1)])

    def forward(self, x: torch.Tensor):
        for layer in self.layers[:-1]:
            x = nn.functional.relu(layer(x))
        x = nn.functional.sigmoid(self.layers[-1](x))
        return x