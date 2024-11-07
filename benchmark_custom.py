import argparse
from typing import Callable, Dict, Tuple
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

from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator

def explore(
        KAN_type: str,
        device: str,
        layer_depths: Tuple[int] = [1],
        hidden_size: Tuple[int] = [1000],
    ):
    kan_classes = {
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
            layers = [size] * depth
            model = kan_classes[KAN_type](layers_hidden=[2] + layers + [1], device=device)
            model.to(device)

            # Create dummy input and output
            dummy_input = torch.randn(32, 2).to(device)
            dummy_output = torch.randn(32, 1).to(device)
            loss_fn = nn.MSELoss()

            # Forward pass
            start_time = time.time()
            pred = model(dummy_input)
            forward_time = (time.time() - start_time) * 1000

            # Backward pass
            start_time = time.time()
            loss = loss_fn(pred, dummy_output)
            loss.backward()
            backward_time = (time.time() - start_time) * 1000

            results[f'depth_{depth}_size_{size}'] = {
                'forward_time': forward_time,
                'backward_time': backward_time
            }

    return results
    

def benchmark(
        dataset: Dict[str, torch.Tensor],
        device: str,
        bs: int,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        model: nn.Module,
        reps: int,
        model_name: str
    ) -> Dict[str, float]:
    
    forward_times = []
    backward_times = []
    forward_mems = []
    backward_mems = []
    macs = None
    params = None
    for k in range(1 + reps):
        train_id = np.random.choice(dataset['train_input'].shape[0], bs, replace=False)
        tensor_input = dataset['train_input'][train_id]
        tensor_input = tensor_input.to(device)

        tensor_output = dataset['train_label'][train_id]
        tensor_output = tensor_output.to(device)

        if device == 'cpu':
            if (k > 0):
                t0 = time.time()
                pred = model(tensor_input)
                t1 = time.time()
                forward_times.append((t1 - t0) * 1000)
                train_loss = loss_fn(pred, tensor_output)
                t2 = time.time()
                train_loss.backward()
                t3 = time.time()
                backward_times.append((t3 - t2) * 1000)
            else:
                with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,
                             on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{model_name}')) as prof:
                    with record_function("model_inference"):
                        model(tensor_input)
                details = prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=20)
                with open(f'results/{model_name}_profiler.txt', 'w') as f:
                    print(details, file=f)
                
        elif device == 'cuda':
            if (k > 0):
                torch.cuda.reset_peak_memory_stats()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                pred = model(tensor_input)
                end.record()

                torch.cuda.synchronize()
                
                forward_times.append(start.elapsed_time(end))
                forward_mems.append(torch.cuda.max_memory_allocated())

                train_loss = loss_fn(pred, tensor_output)

                torch.cuda.reset_peak_memory_stats()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                train_loss.backward()
                end.record()

                torch.cuda.synchronize()
                backward_times.append(start.elapsed_time(end))
                backward_mems.append(torch.cuda.max_memory_allocated())
                    
            else:
                with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True,
                             on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{model_name}')) as prof:
                    with record_function("model_inference"):
                        model(tensor_input)
                details = prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=20)
                with open(f'results/{model_name}_profiler.txt', 'w') as f:
                    print(details, file=f)
                    
    return {
        'forward': np.mean(forward_times),
        'backward': np.mean(backward_times),
        'forward-memory': np.mean(forward_mems) / (1024 ** 3),
        'backward-memory': np.mean(backward_mems) / (1024 ** 3),
        'macs': macs if macs is not None else 0
    }
    # return None


def save_results(t: Dict[str, Dict[str, float]], out_path: str):
    maxlen = np.max([len(k) for k in t.keys()])
    with open(out_path, 'w') as f:
        print(f"{' '*maxlen}  |  {'forward':>11}  |  {'backward':>11}  |  {'forward':>11}  |  {'backward':>11}  |  {'num params':>11}  |  {'num trainable params':>20}  |  {'num macs':>20}", file=f)
        print(f"{' '*maxlen}  |  {'forward':>11}  |  {'backward':>11}  |  {'forward':>11}  |  {'backward':>11}  |  {'num params':>11}  |  {'num trainable params':>20}  |  {'num macs':>20}")
        print('-'*160, file=f)
        print('-'*160)
        for key in t.keys():
            print(f"{key:<{maxlen}}  |  {t[key]['forward']:8.2f} ms  |  {t[key]['backward']:8.2f} ms  |  {t[key]['forward-memory']:8.2f} GB  |  {t[key]['backward-memory']:8.2f} GB  |  {t[key]['params']:>11}  |  {t[key]['train_params']:>20}|  {t[key]['macs']:>20}", file=f)
            print(f"{key:<{maxlen}}  |  {t[key]['forward']:8.2f} ms  |  {t[key]['backward']:8.2f} ms  |  {t[key]['forward-memory']:8.2f} GB  |  {t[key]['backward-memory']:8.2f} GB  |  {t[key]['params']:>11}  |  {t[key]['train_params']:>20}|  {t[key]['macs']:>20}")


def count_params(model: nn.Module) -> Tuple[int, int]:
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pytorch_total_params, pytorch_total_params_train


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', default='times.txt', type=str)
    parser.add_argument('--method', choices=[
            'pykan', 'efficientkan', 'fourierkan',
            'fusedfourierkan', 'chebykan', 'cufkan',
            'fast-kan', 'faster-kan', 'rbf-kan',
            'sine-kan', 'relu-kan',
            'wav-kan', 'mlp', 'all'
        ],
        type=str)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--inp-size', type=int, default=2, help='The dimension of the input variables.')
    parser.add_argument('--hid-size', type=int, default=50, help='The dimension of the hidden layer.')
    parser.add_argument('--reps', type=int, default=10, help='Number of times to repeat execution and average.')
    parser.add_argument('--just-cuda', action='store_true', help='Whether to only execute the cuda version.')
    return parser


def main():
    parser = _create_parser()
    args = parser.parse_args()

    f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    dataset = create_dataset(
        f, 
        n_var=args.inp_size,
        ranges = [-1,1],
        train_num=1000, 
        test_num=1000,
        normalize_input=False,
        normalize_label=False,
        device='cpu',
        seed=0
    )
    loss_fn = lambda x, y: torch.mean((x - y) ** 2)
    
    res = {}
    
    
    kan_models = {
        'fourierkan': FourierKAN(layers=[args.inp_size, args.hid_size, 1], degree=5, device='cpu'),
        'chebykan': ChebyKAN(layers=[args.inp_size, args.hid_size, 1], device='cpu'),
        'mlp': MLP(layers=[args.inp_size, args.hid_size * 10, 1], device='cpu'),
        # 'wav-kan': WavKAN(layers=[args.inp_size, args.hid_size, 1], wavelet_type='dog', device='cpu'),
        'rbf-kan': RBFKAN(layers=[args.inp_size, args.hid_size, 1], device='cpu')
    }

    for model_name, model in kan_models.items():
        if args.method == model_name or args.method == 'all':
            for device in ['cpu', 'cuda']:
                dev_name = 'gpu' if device == 'cuda' else 'cpu'
                if device == 'cpu' and args.just_cuda:
                    continue
                model.to(device)
                res_key = f'{model_name}-{dev_name}'
                res[res_key] = benchmark(dataset, device, args.batch_size, loss_fn, model, args.reps, res_key)
                res[res_key]['params'], res[res_key]['train_params'] = count_params(model)
    save_results(res, args.output_path)
    

if __name__=='__main__':
    main()