import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Dict, Tuple

def extend_grid(grid, k_extend=0):
    # pad k to left and right
    # grid shape: (batch, grid)
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

    for i in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
    return grid

def B_batch(x, grid, k=0, extend=True):

    # x shape: (size, x); grid shape: (size, grid)
    
    x = x.unsqueeze(dim=2)
    grid = grid.unsqueeze(dim=0)
    
    if k == 0:
        value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    else:
        B_km1 = B_batch(x[:,:,0], grid=grid[0], k=k - 1)
        
        value = (x - grid[:, :, :-(k + 1)]) / (grid[:, :, k:-1] - grid[:, :, :-(k + 1)]) * B_km1[:, :, :-1] + (
                    grid[:, :, k + 1:] - x) / (grid[:, :, k + 1:] - grid[:, :, 1:(-k)]) * B_km1[:, :, 1:]
    
    # in case grid is degenerate
    value = torch.nan_to_num(value)
    return value

def curve2coef(x_eval, y_eval, grid, k, lamb=1e-8):
    '''
    converting B-spline curves to B-spline coefficients using least squares.
    
    Args:
    -----
        x_eval : 2D torch.tensor
            shape (in_dim, out_dim, number of samples)
        y_eval : 2D torch.tensor
            shape (in_dim, out_dim, number of samples)
        grid : 2D torch.tensor
            shape (in_dim, grid+2*k)
        k : int
            spline order
        lamb : float
            regularized least square lambda
            
    Returns:
    --------
        coef : 3D torch.tensor
            shape (in_dim, out_dim, G+k)
    '''
    batch = x_eval.shape[0]
    in_dim = x_eval.shape[1]
    out_dim = y_eval.shape[2]
    n_coef = grid.shape[1] - k - 1
    mat = B_batch(x_eval, grid, k)
    mat = mat.permute(1,0,2)[:,None,:,:].expand(in_dim, out_dim, batch, n_coef)
    y_eval = y_eval.permute(1,2,0).unsqueeze(dim=3)
    device = mat.device
    
    #coef = torch.linalg.lstsq(mat, y_eval,
                             #driver='gelsy' if device == 'cpu' else 'gels').solution[:,:,:,0]
        
    XtX = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0,1,3,2), mat)
    Xty = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0,1,3,2), y_eval)
    n1, n2, n = XtX.shape[0], XtX.shape[1], XtX.shape[2]
    identity = torch.eye(n,n)[None, None, :, :].expand(n1, n2, n, n).to(device)
    A = XtX + lamb * identity
    B = Xty
    coef = (A.pinverse() @ B)[:,:,:,0]
    
    return coef

class BasisTransformLayer(nn.Module):
    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.5, scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0, base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, save_plot_data = True, device='cpu', sparse_init=False):

        super(BasisTransformLayer, self).__init__()
        # size 
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k

        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None,:].expand(self.in_dim, num+1)
        grid = extend_grid(grid, k_extend=k)
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)


    def forward(self, x):
        b_splines = B_batch(x, self.grid, self.k)
        return b_splines
    
class LinearCombinationLayer(nn.Module):
    def __init__(self, input_dim, output_dim, grid, num, k, noise_scale=0.5, device='cpu'):
        super(LinearCombinationLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.k = k
        self.scale_sp = torch.nn.Parameter(torch.ones(input_dim, output_dim)) # make scale trainable
        
        noises = (torch.rand(num+1, self.inputdim, self.outdim) - 1/2) * noise_scale / num

        self.coef = torch.nn.Parameter(curve2coef(grid[:,k:-k].permute(1,0), noises, grid, k))
        self.scale_sp = torch.nn.Parameter(torch.ones(input_dim, output_dim) )

    def forward(self, x):
        y_eval = torch.einsum('ijk,jlk->ijl', x, self.coef)
        y = self.scale_sp[None,:,:] * y_eval
        y = torch.sum(y, dim=1)
        return y

class SplineKANLayer(nn.Module):
    def __init__(self, in_dim=3, out_dim=2, num=5, degree=5, device='cpu'):
        super(SplineKANLayer, self).__init__()
        self.basis_transform = BasisTransformLayer(in_dim, out_dim, num = num, k=degree, device=device)
        self.linear_combination = LinearCombinationLayer(in_dim, out_dim, self.basis_transform.grid, num=num, k=degree, device=device)

    def forward(self, x):
        basis_transformed = self.basis_transform(x)
        y = self.linear_combination(basis_transformed)
        return y


class SplineKAN(nn.Module):
    def __init__(self, layers: Tuple[int], device: str = 'cpu', degree=5):
        super().__init__()
        self.layers = nn.ModuleList([SplineKANLayer(layers[i], layers[i+1], degree=degree, device=device) for i in range(len(layers) - 1)])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
    
if __name__ == "__main__":
    x = torch.rand(100,2)
    grid = torch.linspace(-1,1,steps=11)[None, :].expand(2, 11)
    B_batch(x, grid, k=3).shape