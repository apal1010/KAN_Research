import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable, Dict, Tuple


class BasisTransformLayer(nn.Module):
    def __init__(self, grid_min=-2., grid_max=2., num_grids=8):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.grid = nn.Parameter(torch.linspace(grid_min, grid_max, num_grids), requires_grad=False)

    def forward(self, x):
        x = x.unsqueeze(-1)
        basis = torch.exp(-((x - self.grid) / ((self.grid_max - self.grid_min) / (self.num_grids - 1))) ** 2)
        return basis

class LinearCombinationLayer(nn.Module):
    def __init__(self, in_features, out_features, num_grids, spline_weight_init_scale=0.1):
        super().__init__()
        self.spline_weight = nn.Parameter(torch.randn(in_features, num_grids, out_features) * spline_weight_init_scale)

    def forward(self, transformed_basis):
        y = torch.einsum("bid,ido->bo", transformed_basis, self.spline_weight)
        return y

class RBFLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_min=-2., grid_max=2., num_grids=8, spline_weight_init_scale=0.1):
        super().__init__()
        self.basis_transform = BasisTransformLayer(grid_min, grid_max, num_grids)
        self.linear_combination = LinearCombinationLayer(in_features, out_features, num_grids, spline_weight_init_scale)

    def forward(self, x):
        basis = self.basis_transform(x)
        return self.linear_combination(basis)

class RBFKANLayer(nn.Module):
   def __init__(self, input_dim, output_dim, grid_min=-2., grid_max=2., num_grids=8, use_base_update=True, base_activation=nn.SiLU(), spline_weight_init_scale=0.1):
       super().__init__()
       self.input_dim = input_dim
       self.output_dim = output_dim
       self.use_base_update = use_base_update
       self.base_activation = base_activation
       self.spline_weight_init_scale = spline_weight_init_scale
       self.rbf_linear = RBFLinear(input_dim, output_dim, grid_min, grid_max, num_grids, spline_weight_init_scale)
       self.base_linear = nn.Linear(input_dim, output_dim) if use_base_update else None

   def forward(self, x):
       ret = self.rbf_linear(x)
       return ret

class RBFKAN(nn.Module):
   def __init__(self, layers, grid_min=-2., grid_max=2., degree=5, use_base_update=True, base_activation=nn.SiLU(), spline_weight_init_scale=0.1, device ='cpu'):
       super().__init__()
       self.layers = nn.ModuleList([RBFKANLayer(in_dim, out_dim, grid_min, grid_max, degree, use_base_update, base_activation, spline_weight_init_scale) for in_dim, out_dim in zip(layers[:-1], layers[1:])])

   def forward(self, x):
       for layer in self.layers:
           x = layer(x)
       return x