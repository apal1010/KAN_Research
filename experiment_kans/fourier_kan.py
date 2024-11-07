import torch as th
import numpy as np
from torch import nn
from typing import Callable, Dict, Tuple

#This is inspired by Kolmogorov-Arnold Networks but using 1d fourier coefficients instead of splines coefficients
#It should be easier to optimize as fourier are more dense than spline (global vs local)
#Once convergence is reached you can replace the 1d function with spline approximation for faster evaluation giving almost the same result
#The other advantage of using fourier over spline is that the function are periodic, and therefore more numerically bounded
#Avoiding the issues of going out of grid
class BasisTransformLayer(th.nn.Module):
    def __init__(self, inputdim, gridsize):
        super(BasisTransformLayer, self).__init__()
        self.inputdim = inputdim
        self.gridsize = gridsize

    def forward(self, x):
        xshp = x.shape
        k = th.reshape(th.arange(1, self.gridsize + 1, device=x.device), (1, 1, self.gridsize))
        xrshp = th.reshape(x, (x.shape[0], x.shape[1], 1))
        c = th.cos(k * xrshp)
        s = th.sin(k * xrshp)
        y = th.cat((c, s), -1)
        return y

class LinearCombinationLayer(th.nn.Module):
    def __init__(self, inputdim, outdim, gridsize):
        super(LinearCombinationLayer, self).__init__()
        self.inputdim = inputdim
        self.outdim = outdim
        self.gridsize = gridsize
        self.fouriercoeffs = th.nn.Parameter(th.randn(outdim, inputdim, 2 * gridsize) /
                                             (np.sqrt(inputdim) * np.sqrt(self.gridsize)))

    def forward(self, transformed_basis):
        y = th.einsum("bid,oid->bo", transformed_basis, self.fouriercoeffs)
        return y

class FourierLayer(th.nn.Module):
    def __init__(self, inputdim, outdim, gridsize):
        super(FourierLayer, self).__init__()
        self.basis_transform = BasisTransformLayer(inputdim, gridsize)
        self.linear_combination = LinearCombinationLayer(inputdim, outdim, gridsize)

    def forward(self, x):
        t = self.basis_transform(x)
        y = self.linear_combination(t)
        return y


class FourierKAN(nn.Module):
    def __init__(self, layers: Tuple[int], device: str = 'cpu', degree: int=5):
        super().__init__()
        self.layers = nn.ModuleList([FourierLayer(layers[i], layers[i+1], gridsize=degree).to(device) for i in range(len(layers) - 1)])

    def forward(self, x: th.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
    
if __name__ == '__main__':
    fkan = FourierKAN((1, 7, 8, 1), degree=6)
    x = th.randn(10, 1)
    y = fkan(x)
    