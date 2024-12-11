import torch as th
import numpy as np
from torch import nn
from typing import Callable, Dict, Tuple
import logging
import socket
from datetime import datetime, timedelta
from torch.autograd.profiler import record_function
import torch.cuda.profiler as profiler

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

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
    

def trace_handler(prof: th.profiler.profile):
    # Prefix for file names.
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{host_name}_{timestamp}"

    # Construct the trace file.
    prof.export_chrome_trace(f"{file_prefix}.json.gz")

    # Construct the memory timeline file.
    prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")
    
if __name__ == '__main__':
    device = 'cuda' if th.cuda.is_available() else 'cpu'

    model = FourierKAN(layers=(3, 64, 64, 1), device=device, degree=10).to(device)
    
    # th.cuda.memory._record_memory_history()
    
    inputs = th.randn(5, 3).to(device)  # Batch of 5, input dimension 3
    # for i in range(20):
    
    #     # Create some random input tensors
        
        
    #     # Run the tensors through the model
    #     outputs = model(inputs)
    
    # th.cuda.memory._dump_snapshot("my_snapshot.pickle")
    
    with th.profiler.profile(
        activities=[
            th.profiler.ProfilerActivity.CPU,
            th.profiler.ProfilerActivity.CUDA,
        ],
        schedule=th.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=trace_handler,
    ) as prof:
       # Run the PyTorch Model inside the profile context.
        for _ in range(20):
            prof.step()
            with record_function("## forward ##"):
                pred = model(inputs)


    file_prefix = 'smth'
    prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")
    
    