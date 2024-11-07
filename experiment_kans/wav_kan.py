'''This is a sample code for the simulations of the paper:
Bozorgasl, Zavareh and Chen, Hao, Wav-KAN: Wavelet Kolmogorov-Arnold Networks (May, 2024)

https://arxiv.org/abs/2405.12832
and also available at:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4835325
We used efficient KAN notation and some part of the code:https://github.com/Blealtan/efficient-kan

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from typing import Callable, Dict, Tuple

class BasisTransformLayer(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat'):
        super(BasisTransformLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type

        # Parameters for wavelet transformation
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))

    def forward(self, x):
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)
        else:
            x_expanded = x

        translation_expanded = self.translation.unsqueeze(0).expand(x.size(0), -1, -1)
        scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - translation_expanded) / scale_expanded

        # Implementation of different wavelet types
        if self.wavelet_type == 'mexican_hat':
            term1 = ((x_scaled ** 2)-1)
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi**0.25)) * term1 * term2
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0  # Central frequency
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
        elif self.wavelet_type == 'dog':
            dog = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
            wavelet = dog
        elif self.wavelet_type == 'meyer':
            v = torch.abs(x_scaled)
            pi = math.pi

            def meyer_aux(v):
                return torch.where(v <= 1/2, torch.ones_like(v), torch.where(v >= 1, torch.zeros_like(v), torch.cos(pi / 2 * nu(2 * v - 1))))

            def nu(t):
                return t**4 * (35 - 84*t + 70*t**2 - 20*t**3)

            wavelet = torch.sin(pi * v) * meyer_aux(v)
        elif self.wavelet_type == 'shannon':
            pi = math.pi
            sinc = torch.sinc(x_scaled / pi)
            window = torch.hamming_window(x_scaled.size(-1), periodic=False, dtype=x_scaled.dtype, device=x_scaled.device)
            wavelet = sinc * window
        else:
            raise ValueError("Unsupported wavelet type")

        return wavelet

class LinearCombinationLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearCombinationLayer, self).__init__()
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, wavelet):
        wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
        wavelet_output = wavelet_weighted.sum(dim=2)
        return self.bn(wavelet_output)

class WaveletKanLayer(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat'):
        super(WaveletKanLayer, self).__init__()
        self.wavelet_transform = WaveletTransform(in_features, out_features, wavelet_type)
        self.linear_comb = LinearCombinationLayer(in_features, out_features)

    def forward(self, x):
        wavelet = self.wavelet_transform(x)
        wavelet_output = self.linear_comb(wavelet)
        return wavelet_output


class WavKAN(nn.Module):
    def __init__(self, layers: Tuple[int], wavelet_type: str = 'mexican_hat', device: str = 'cpu'):
        super().__init__()
        self.layers = nn.ModuleList([WaveletKanLayer(layers[i], layers[i+1], wavelet_type=wavelet_type).to(device) for i in range(len(layers) - 1)])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x