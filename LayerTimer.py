import torch
import torch.nn as nn
from typing import Dict
from collections import defaultdict

class LayerTimer:
    def __init__(self, model: nn.Module, target_layer_names: list = None):
        """
        Timer for measuring execution time of specific layers.
        
        Args:
            model: PyTorch model to profile
            target_layer_names: List of layer names to track. If None, tracks all layers.
        """
        self.model = model
        self.target_layer_names = target_layer_names
        self.forward_times = defaultdict(list)
        self.backward_times = defaultdict(list)
        self.handles = []
        self.events = {}
        self._attach_hooks()
        
    def _attach_hooks(self):
        """Attach timing hooks to layers."""
        for name, module in self.model.named_modules():
            if self.target_layer_names is None or name in self.target_layer_names:
                # Create CUDA events for this layer
                self.events[name] = {
                    'forward_start': torch.cuda.Event(enable_timing=True),
                    'forward_end': torch.cuda.Event(enable_timing=True),
                    'backward_start': torch.cuda.Event(enable_timing=True),
                    'backward_end': torch.cuda.Event(enable_timing=True)
                }
                
                # Forward hooks
                self.handles.append(
                    module.register_forward_pre_hook(
                        lambda m, inp, name=name: self._forward_pre_hook(name)
                    )
                )
                self.handles.append(
                    module.register_forward_hook(
                        lambda m, inp, out, name=name: self._forward_hook(name)
                    )
                )
                
                # Backward hooks
                self.handles.append(
                    module.register_full_backward_hook(
                        lambda m, grad_in, grad_out, name=name: self._backward_hook(name)
                    )
                )
    
    def _forward_pre_hook(self, name: str):
        """Record start time of forward pass."""
        self.events[name]['forward_start'].record()
    
    def _forward_hook(self, name: str):
        """Record end time of forward pass."""
        self.events[name]['forward_end'].record()
        torch.cuda.synchronize()
        time_ms = self.events[name]['forward_start'].elapsed_time(self.events[name]['forward_end'])
        self.forward_times[name].append(time_ms)
    
    def _backward_hook(self, name: str):
        """Record backward pass time."""
        self.events[name]['backward_start'].record()
        self.events[name]['backward_end'].record()
        torch.cuda.synchronize()
        time_ms = self.events[name]['backward_start'].elapsed_time(self.events[name]['backward_end'])
        self.backward_times[name].append(time_ms)
    
    def clear(self):
        """Clear stored timing data."""
        self.forward_times.clear()
        self.backward_times.clear()
    
    def remove_hooks(self):
        """Remove all hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
    
    def get_average_times(self) -> Dict[str, Dict[str, float]]:
        """Get average forward and backward times for each layer."""
        def _reject_outliers(data, m=1.):
            """Reject outliers from data using modified Z-score method."""
            if len(data) < 2:
                return data
            median = torch.median(torch.tensor(data))
            diff = torch.abs(torch.tensor(data) - median)
            med_abs_deviation = torch.median(diff)
            modified_z_scores = 0.6745 * diff / med_abs_deviation
            return [d for d, mz in zip(data, modified_z_scores) if mz < m]

        results = {}
        for name in self.forward_times.keys():
            forward_times_filtered = _reject_outliers(self.forward_times[name])
            backward_times_filtered = _reject_outliers(self.backward_times[name])
            results[name] = {
                'forward_avg': sum(forward_times_filtered) / len(forward_times_filtered) if forward_times_filtered else 0,
                'backward_avg': sum(backward_times_filtered) / len(backward_times_filtered) if backward_times_filtered else 0,
                'total_avg': (sum(forward_times_filtered) + sum(backward_times_filtered)) / len(forward_times_filtered) if forward_times_filtered else 0
            }
        return results
        # results = {}
        # for name in self.forward_times.keys():
        #     results[name] = {
        #         'forward_avg': sum(self.forward_times[name]) / len(self.forward_times[name]),
        #         'backward_avg': sum(self.backward_times[name]) / len(self.backward_times[name]) if self.backward_times[name] else 0,
        #         'total_avg': (sum(self.forward_times[name]) + sum(self.backward_times[name])) / len(self.forward_times[name])
        #     }
        # return results