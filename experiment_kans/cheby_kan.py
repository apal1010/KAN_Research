import torch
import torch.nn as nn
from typing import Callable, Dict, Tuple

class BasisTransformLayer(nn.Module):
    def __init__(self, input_dim, degree):
        super(BasisTransformLayer, self).__init__()
        self.inputdim = input_dim
        self.degree = degree
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        x = torch.tanh(x)
        x = x.view((-1, self.inputdim, 1)).expand(-1, -1, self.degree + 1)
        x = x.acos()
        x *= self.arange
        x = x.cos()
        return x

class LinearCombinationLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(LinearCombinationLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        y = torch.einsum("bid,iod->bo", x, self.cheby_coeffs)
        return y

class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.basis_transform = BasisTransformLayer(input_dim, degree)
        self.linear_combination = LinearCombinationLayer(input_dim, output_dim, degree)

    def forward(self, x):
        basis_transformed = self.basis_transform(x)
        y = self.linear_combination(basis_transformed)
        return y


class ChebyKAN(nn.Module):
    def __init__(self, layers: Tuple[int], device: str = 'cpu', degree=5):
        super().__init__()
        self.layers = nn.ModuleList([ChebyKANLayer(layers[i], layers[i+1], degree=degree).to(device) for i in range(len(layers) - 1)])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
    