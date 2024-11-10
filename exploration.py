import argparse
from typing import List, Callable, Dict, Tuple
import time
import numpy as np
import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from kan import create_dataset
from experiment_kans.MLP import MLP
from experiment_kans.fourier_kan import FourierKAN
from experiment_kans.cheby_kan import ChebyKAN
from experiment_kans.wav_kan import WavKAN
from experiment_kans.rbf_kan import RBFKAN
from experiment_kans.spline_kan import SplineKAN
from LayerTimer import LayerTimer
import matplotlib.pyplot as plt

import os
import re

EPS = 1e-10

def parse_breakdown_data(layer_timer_data, components, metric):
    breakdown_dict = {}
    lines = layer_timer_data.strip().split('\n')
    header = lines[1].split("  ")
    header = [s.strip() for s in header if s]
    metric_index = header.index(metric)
    
    for line in lines[4:]:
        parts = line.split("  ")
        parts = [s.strip() for s in parts if s]
        if len(parts) > 1 and parts[0] in components:
            breakdown_dict[parts[0]] = breakdown_dict.get(parts[0], 0) + float(parts[metric_index].strip('%')) / 100
    
    return breakdown_dict
        
    
def parse_layer_data(layer_timer_data):
    total_linear_comb_time_forward = 0.0
    total_basis_trans_time_forward = 0.0
    total_linear_comb_time_backward = 0.0
    total_basis_trans_time_backward = 0.0
    
    l_layers = 0
    b_layers = 0

    for layer_name, times in layer_timer_data.items():
        if "linear_combination" in layer_name:
            total_linear_comb_time_forward += times['forward_avg']
            total_linear_comb_time_backward += times['backward_avg']
            l_layers += 1
        elif "basis_transform" in layer_name:
            total_basis_trans_time_forward += times['forward_avg']
            total_basis_trans_time_backward +=times['backward_avg']
            b_layers += 1
            
            

    return {
        'lc_f' : total_linear_comb_time_forward,
        'lc_b' : total_linear_comb_time_backward,
        'bt_f' : total_basis_trans_time_forward,
        'bt_b' : total_basis_trans_time_backward,
        'ratio' : (total_basis_trans_time_forward) / ((total_basis_trans_time_forward) + (total_linear_comb_time_forward) + EPS)
    }

def profile_kan_model(
    KAN_type: str,
    device: str,
    layer_depths: List[int] = [1],
    hidden_size: List[int] = [1000],
    batch_size: int = 32,
    num_warmup: int = 10,
    num_iterations: int = 100
) -> Dict:
    """
    Profile KAN model performance with detailed timing for specific layers.
    
    Args:
        KAN_type: Type of KAN model ('fourierkan', 'chebykan', 'wav-kan', 'rbf-kan', 'mlp')
        device: Device to run on ('cuda' or 'cpu')
        layer_depths: List of depths to test
        hidden_size: List of hidden layer sizes to test
        batch_size: Batch size for testing
        num_warmup: Number of warmup iterations
        num_iterations: Number of iterations for timing
    """
    kan_classes = {
        'splinekan': SplineKAN,
        'fourierkan': FourierKAN,
        'chebykan': ChebyKAN,
        'wav-kan': WavKAN,
        'rbf-kan': RBFKAN,
        'mlp': MLP
    }
    
    if KAN_type not in kan_classes:
        raise ValueError(f"Unknown KAN type: {KAN_type}")
    
    results = {}
    
    for depth in layer_depths:
        for size in hidden_size:
            layers = [size] * (depth+1)
            model = kan_classes[KAN_type](layers, degree = 2, device=device)
            model.to(device)
            
            print(f"\nProfiling {KAN_type} with depth {depth} and size {size}")
            
            input_data = torch.randn(batch_size, size).to(device)
            target_data = torch.randn(batch_size, size).to(device)
            
            # Warmup
            for _ in range(num_warmup):
                model(input_data)
            
            # Timing
            layer_timer = LayerTimer(model)
            for _ in range(num_iterations):
                output = model(input_data)
                loss = torch.mean((output - target_data) ** 2)
                loss.backward()
            
            layer_timer_data = layer_timer.get_average_times()
            
            results[f'depth_{depth}_size_{size}'] = parse_layer_data(layer_timer_data)

            
    return results

def vary_by_degree(KAN_type: str, device: str, depths: int, hidden_size: int, batch_size: int, num_warmup: int, num_iterations: int, degrees: List[int]) -> Dict:
    """
    Profile KAN model performance with varying degree of basis transform.
    
    Args:
        KAN_type: Type of KAN model ('fourierkan', 'chebykan', 'wav-kan', 'rbf-kan', 'mlp')
        device: Device to run on ('cuda' or 'cpu')
        layer_depth: Depth of the KAN model
        hidden_size: Hidden layer size
        batch_size: Batch size for testing
        num_warmup: Number of warmup iterations
        num_iterations: Number of iterations for timing
    """
    kan_classes = {
        'splinekan': SplineKAN,
        'fourierkan': FourierKAN,
        'chebykan': ChebyKAN,
        'wav-kan': WavKAN,
        'rbf-kan': RBFKAN,
        'mlp': MLP
    }
    
    if KAN_type not in kan_classes:
        raise ValueError(f"Unknown KAN type: {KAN_type}")
    
    results = {}
    
    for depth in depths:
        for degree in degrees:
            layers = [hidden_size] * (depth+1)
            model = kan_classes[KAN_type](layers, degree = degree, device=device)
            model.to(device)
            
            print(f"\nProfiling {KAN_type} with degree {degree} and depth {depth}")
            
            input_data = torch.randn(batch_size, hidden_size).to(device)
            target_data = torch.randn(batch_size, hidden_size).to(device)
            
            # Warmup
            for _ in range(num_warmup):
                model(input_data)
            
            # Timing
            layer_timer = LayerTimer(model)
            for _ in range(num_iterations):
                output = model(input_data)
                loss = torch.mean((output - target_data) ** 2)
                loss.backward()
            
            layer_timer_data = layer_timer.get_average_times()
            
            results[f'degree_{depth}_{degree}'] = parse_layer_data(layer_timer_data)
        
    return results

def profile_kan_model_by_degree(
    KAN_type: str,
    device: str,
    depths: List[int] = [1],
    hidden_size: int = 1000,
    batch_size: int = 32,
    num_warmup: int = 10,
    num_iterations: int = 100,
    degrees: List[int] = [2, 4, 6, 8, 10],
    components: List[str] = ['aten::einsum']
) -> Dict:
    """
    Profile KAN model performance with varying degree of basis transform.
    
    Args:
        KAN_type: Type of KAN model ('fourierkan', 'chebykan', 'wav-kan', 'rbf-kan', 'mlp')
        device: Device to run on ('cuda' or 'cpu')
        depths: List of depths to test
        hidden_size: Hidden layer size
        batch_size: Batch size for testing
        num_warmup: Number of warmup iterations
        num_iterations: Number of iterations for timing
        degrees: List of degrees to test
        components: List of components to profile
    """
    kan_classes = {
        'splinekan': SplineKAN,
        'fourierkan': FourierKAN,
        'chebykan': ChebyKAN,
        'wav-kan': WavKAN,
        'rbf-kan': RBFKAN,
        'mlp': MLP
    }
    
    if KAN_type not in kan_classes:
        raise ValueError(f"Unknown KAN type: {KAN_type}")
    
    results = {}
    
    for depth in depths:
        for degree in degrees:
            layers = [hidden_size] * (depth + 1)
            model = kan_classes[KAN_type](layers, degree=degree, device=device)
            model.to(device)
            
            print(f"\nProfiling {KAN_type} with degree {degree} and depth {depth}")
            
            input_data = torch.randn(batch_size, hidden_size).to(device)
            target_data = torch.randn(batch_size, hidden_size).to(device)
            
            # Warmup
            for _ in range(num_warmup):
                model(input_data)
            
            # Profiling
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                for _ in range(num_iterations):
                    with record_function("model_inference"):
                        output = model(input_data)
            
            # Save profiling results
            breakdown_folder = f"breakdown_results/{KAN_type}/{device}"
            os.makedirs(breakdown_folder, exist_ok=True)
            breakdown_file = os.path.join(breakdown_folder, f"degree_{degree}_depth_{depth}.txt")
            with open(breakdown_file, "w") as f:
                f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
            
            # Parse breakdown data
            with open(breakdown_file, "r") as f:
                layer_timer_data = f.read()
            breakdown_data = parse_breakdown_data(layer_timer_data, components, 'CPU total %')
            results[f'degree_{degree}_depth_{depth}'] = breakdown_data
    
    return results

def plot_breakdown_degree(res: Dict, degrees: List[int], depths: List[int], model_type: str, device: str, components: List[str]):
    depth_degree_keys = sorted(res.keys(), key=lambda x: int(re.search(r'degree_(\d+)', x).group(1)))
    num_keys = len(depth_degree_keys)
    num_components = len(components)

    fig, axs = plt.subplots(len(depths), 1, figsize=(15, 5 * len(depths)))

    for i, depth in enumerate(depths):
        depth_keys = [key for key in depth_degree_keys if f'depth_{depth}' in key]
        breakdown_matrix = np.zeros((len(depth_keys), num_components))

        for k, key in enumerate(depth_keys):
            for j, component in enumerate(components):
                breakdown_matrix[k, j] = res[key].get(component, 0)

        im = axs[i].imshow(breakdown_matrix, cmap='viridis', aspect='auto')

        axs[i].set_xticks(np.arange(num_components))
        axs[i].set_yticks(np.arange(len(depth_keys)))
        axs[i].set_xticklabels(components)
        axs[i].set_yticklabels(depth_keys)

        plt.setp(axs[i].get_xticklabels(), rotation=45, rotation_mode="anchor")

        for k in range(len(depth_keys)):
            for j in range(num_components):
                text = axs[i].text(j, k, f"{breakdown_matrix[k, j]:.2f}", ha="center", va="center", color="w")

        axs[i].set_title(f"Breakdown of {model_type} on {device} for Depth {depth}")

    fig.tight_layout()
    plt.colorbar(im, ax=axs, orientation='horizontal', fraction=0.02, pad=0.04)
    plt.savefig(f"{model_type}_{device}_degree_breakdown.png")

def profile_kan_breakdown(
    KAN_type: str,
    device: str,
    layer_depths: List[int] = [1],
    hidden_size: List[int] = [1000],
    batch_size: int = 32,
    num_warmup: int = 10,
    num_iterations: int = 100,
    components: List[str] = ['aten::einsum']
) -> Dict:
    os.makedirs("breakdown_results", exist_ok=True)

    kan_classes = {
        'splinekan': SplineKAN,
        'fourierkan': FourierKAN,
        'chebykan': ChebyKAN,
        'wav-kan': WavKAN,
        'rbf-kan': RBFKAN,
        'mlp': MLP
    }

    if KAN_type not in kan_classes:
        raise ValueError(f"Unknown KAN type: {KAN_type}")

    results = {}

    for depth in layer_depths:
        for size in hidden_size:
            layers = [size] * (depth + 1)
            model = kan_classes[KAN_type](layers, degree=3, device=device)
            model.to(device)

            print(f"\nProfiling {KAN_type} with depth {depth} and size {size}")

            input_data = torch.randn(batch_size, size).to(device)
            target_data = torch.randn(batch_size, size).to(device)

            # Warmup
            for _ in range(num_warmup):
                model(input_data)

            # Profiling
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                for _ in range(num_iterations):
                    with record_function("model_inference"):
                        output = model(input_data)

            # Save profiling results
            breakdown_folder = f"breakdown_results/{KAN_type}/{device}"
            os.makedirs(breakdown_folder, exist_ok=True)
            breakdown_file = os.path.join(breakdown_folder, f"depth_{depth}_size_{size}.txt")
            with open(breakdown_file, "w") as f:
                f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

            # Parse breakdown data
            with open(breakdown_file, "r") as f:
                layer_timer_data = f.read()
            breakdown_data = parse_breakdown_data(layer_timer_data, components, 'CPU total %') 
            results[f'depth_{depth}_size_{size}'] = breakdown_data

    return results
    

def plot_results_depth_size(res: Dict, depths: List[int], sizes: List[int], model_type: str, device: str):
    linear_comb_times_t = np.zeros((len(depths), len(sizes)))
    basis_trans_times_t = np.zeros((len(depths), len(sizes)))
    linear_comb_times_f = np.zeros((len(depths), len(sizes)))
    basis_trans_times_f = np.zeros((len(depths), len(sizes)))
    basis_trans_percentages = np.zeros((len(depths), len(sizes)))
    total_times = np.zeros((len(depths), len(sizes)))

    for i, depth in enumerate(depths):
        for j, size in enumerate(sizes):
            key = f'depth_{depth}_size_{size}'
            linear_comb_times_f[i, j] = (res[key]['lc_f']) / depth
            basis_trans_times_f[i, j] = (res[key]['bt_f']) / depth
            linear_comb_times_t[i, j] = (res[key]['lc_f'] + res[key]['lc_b']) / depth
            basis_trans_times_t[i, j] = (res[key]['bt_f'] + res[key]['bt_b']) / depth
            basis_trans_percentages[i, j] = res[key]['ratio']
            total_times[i, j] = (res[key]['lc_f'] + res[key]['bt_f']) / depth

    fig, axs = plt.subplots(2, 2, figsize=(20,15))
    axs = axs.flatten()

    for i, depth in enumerate(depths):
        axs[0].plot(sizes, linear_comb_times_f[i, :], label=f'Depth {depth}')
        axs[1].plot(sizes, basis_trans_times_f[i, :], label=f'Depth {depth}')
        axs[2].plot(sizes, basis_trans_percentages[i, :], label=f'Depth {depth}')
        axs[3].plot(sizes, total_times[i, :], label=f'Depth {depth}')

    axs[0].set_title('Linear Combination Forward Time')
    axs[0].set_xlabel('Hidden Layer Size')
    axs[0].set_ylabel('Time (ms)')
    axs[0].legend()

    axs[1].set_title('Basis Transform Forward Time')
    axs[1].set_xlabel('Hidden Layer Size')
    axs[1].set_ylabel('Time (ms)')
    axs[1].legend()

    axs[2].set_title('Ratio of layer time spent on Basis Transform')
    axs[2].set_xlabel('Hidden Layer Size')
    axs[2].set_ylabel('Ratio')
    axs[2].legend()
    
    axs[3].set_title('Total forward time per layer')
    axs[3].set_xlabel('Hidden Layer Size')
    axs[3].set_ylabel('Time (ms)')
    axs[3].legend()

    plt.tight_layout()
    plt.savefig(f"{model_type}_{device}_exploration.png")

def plot_results_degree(res: Dict, degrees: List[int], model_type: str, depths, device: str):
    linear_comb_times_t = np.zeros((len(degrees), len(depths)))
    basis_trans_times_t = np.zeros((len(degrees), len(depths)))
    linear_comb_times_f = np.zeros((len(degrees), len(depths)))
    basis_trans_times_f = np.zeros((len(degrees), len(depths)))
    basis_trans_percentages = np.zeros((len(degrees), len(depths)))
    total_times = np.zeros((len(degrees), len(depths)))

    for i, degree in enumerate(degrees):
        for j, depth in enumerate(depths):
            key = f'degree_{depth}_{degree}'
            linear_comb_times_f[i, j] = (res[key]['lc_f']) / depth
            basis_trans_times_f[i, j] = (res[key]['bt_f']) / depth
            linear_comb_times_t[i, j] = (res[key]['lc_f'] + res[key]['lc_b']) / depth
            basis_trans_times_t[i, j] = (res[key]['bt_f'] + res[key]['bt_b']) / depth
            basis_trans_percentages[i, j] = res[key]['ratio']
            total_times[i, j] = (res[key]['lc_f'] + res[key]['bt_f']) / depth

    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    axs = axs.flatten()

    for j, depth in enumerate(depths):
        axs[0].plot(degrees, linear_comb_times_f[:, j], label=f'Depth {depth}')
        axs[1].plot(degrees, basis_trans_times_f[:, j], label=f'Depth {depth}')
        axs[2].plot(degrees, basis_trans_percentages[:, j], label=f'Depth {depth}')
        axs[3].plot(degrees, total_times[:, j], label=f'Depth {depth}')

    axs[0].set_title('Linear Combination Forward Time')
    axs[0].set_xlabel('Degree')
    axs[0].set_ylabel('Time (ms)')
    axs[0].legend()

    axs[1].set_title('Basis Transform Forward Time')
    axs[1].set_xlabel('Degree')
    axs[1].set_ylabel('Time (ms)')
    axs[1].legend()

    axs[2].set_title('Ratio of layer time spent on Basis Transform')
    axs[2].set_xlabel('Degree')
    axs[2].set_ylabel('Ratio')
    axs[2].legend()

    axs[3].set_title('Total forward time per layer')
    axs[3].set_xlabel('Degree')
    axs[3].set_ylabel('Time (ms)')
    axs[3].legend()

    plt.tight_layout()
    plt.savefig(f"{model_type}_{device}_degree_exploration.png")
    
def plot_breakdown_results(res: Dict, components: List[str], model_type: str, depths, device: str):
    depth_size_keys = sorted(res.keys(), key=lambda x: int(re.search(r'size_(\d+)', x).group(1)))
    num_keys = len(depth_size_keys)
    num_components = len(components)

    breakdown_matrix = np.zeros((num_keys, num_components))

    fig, axs = plt.subplots(len(depths), 1, figsize=(15, 5 * len(depths)))

    for i, depth in enumerate(depths):
        depth_keys = [key for key in depth_size_keys if f'depth_{depth}' in key]
        breakdown_matrix = np.zeros((len(depth_keys), num_components))

        for k, key in enumerate(depth_keys):
            for j, component in enumerate(components):
                breakdown_matrix[k, j] = res[key].get(component, 0)

        im = axs[i].imshow(breakdown_matrix, cmap='viridis', aspect='auto')

        axs[i].set_xticks(np.arange(num_components))
        axs[i].set_yticks(np.arange(len(depth_keys)))
        axs[i].set_xticklabels(components)
        axs[i].set_yticklabels(depth_keys)

        plt.setp(axs[i].get_xticklabels(), rotation=45, rotation_mode="anchor")

        for k in range(len(depth_keys)):
            for j in range(num_components):
                text = axs[i].text(j, k, f"{breakdown_matrix[k, j]:.2f}", ha="center", va="center", color="w")

        axs[i].set_title(f"Breakdown of {model_type} on {device} for Depth {depth}")

    fig.tight_layout()
    plt.colorbar(im, ax=axs, orientation='horizontal', fraction=0.02, pad=0.04)
    plt.savefig(f"{model_type}_{device}_breakdown.png")

def main():

    loss_fn = lambda x, y: torch.mean((x - y) ** 2)
    
    breakdown_components = {
        'splinekan':    ['aten::einsum', 'aten::copy_', 'aten::mul'],
        'fourierkan':   ['aten::einsum', 'aten::copy_', 'aten::mul', 'aten::cos', 'aten::sin'],
        'chebykan':     ['aten::einsum', 'aten::copy_', 'aten::acos', 'aten::cos', 'aten::tanh', 'aten::mul_'],
        'rbf-kan':      ['aten::einsum', 'aten::copy_', 'aten::exp', 'aten::sub', 'aten::pow', 'aten::div', 'aten::neg'],
    }
    
    # device = 'cuda'

    depths = [2, 4, 8, 12]
    # sizes = [10, 20, 40, 80, 160, 200, 400, 500, 800]#, 1600, 2000, 2400]
    sizes = [i * 50 for i in range(1, 15)]
    # depths = [1,2,4]
    # sizes = [10, 20]
    degrees = [i for i in range (2, 21)]
    for device in ['cpu']:
        for model_type in ['splinekan', 'fourierkan', 'chebykan', 'rbf-kan', 'mlp']:
            # results varying size and depth
            res = profile_kan_model(KAN_type=model_type, device=device, layer_depths=depths, hidden_size=sizes, batch_size=32, num_warmup=5, num_iterations=20)
            plot_results_depth_size(res, depths, sizes, model_type, device)
            # #results varying degree
            if model_type == 'mlp':
                continue
            res = vary_by_degree(KAN_type=model_type, device=device, depths=depths, hidden_size=10, batch_size=32, num_warmup=10, num_iterations=20, degrees=degrees)
            plot_results_degree(res, degrees, model_type, depths, device)
            # results for profiling breakdown
            breakdown_res = profile_kan_breakdown(KAN_type=model_type, device=device, layer_depths=depths, hidden_size=sizes, batch_size=32, num_warmup=3, num_iterations=10, components=breakdown_components[model_type])
            plot_breakdown_results(breakdown_res, breakdown_components[model_type], model_type, depths, device)
            
            # Degree varying breakdown
            breakdown_res_by_degree = profile_kan_model_by_degree( KAN_type=model_type, device=device, depths=depths, hidden_size=100, batch_size=32, num_warmup=3, num_iterations=10, degrees=degrees, components=breakdown_components[model_type])
            plot_breakdown_degree(breakdown_res_by_degree, degrees, depths, model_type, device, breakdown_components[model_type])
    

if __name__=='__main__':
    main()