"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-24 16:23:19
LastEditors: ScopeX-ASU jiaqigu@asu.edu
LastEditTime: 2023-10-28 22:57:34
"""

from typing import Callable, Dict, Optional

import torch
from pyutils.torch_train import set_torch_deterministic
from torch import Tensor, nn

from .layers.gemm_conv2d import GemmConv2d
from .layers.gemm_linear import GemmLinear

__all__ = ["SparseBP_Base"]


class SparseBP_Base(nn.Module):
    _conv_linear = (GemmConv2d, GemmLinear)
    _linear = (GemmLinear,)
    _conv = (GemmConv2d,)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_parameters(self, random_state: int = None) -> None:
        for name, m in self.named_modules():
            if isinstance(m, self._conv_linear):
                if random_state is not None:
                    # deterministic seed, but different for different layer, and controllable by random_state
                    set_torch_deterministic(random_state + sum(map(ord, name)))
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def backup_phases(self) -> None:
        self.phase_backup = {}
        for layer_name, layer in self.fc_layers.items():
            self.phase_backup[layer_name] = {
                "weight": layer.weight.data.clone()
                if layer.weight is not None
                else None,
            }

    def restore_phases(self) -> None:
        for layer_name, layer in self.fc_layers.items():
            backup = self.phase_backup[layer_name]
            for param_name, param_src in backup.items():
                param_dst = getattr(layer, param_name)
                if param_src is not None and param_dst is not None:
                    param_dst.data.copy_(param_src.data)

    def set_phase_variation(self, flag: bool = True):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.set_phase_variation(flag)

    def set_global_temp_drift(self, flag: bool = True):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.set_global_temp_drift(flag)

    def set_gamma_noise(
        self, noise_std: float = 0.0, random_state: Optional[int] = None
    ) -> None:
        self.gamma_noise_std = noise_std
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.set_gamma_noise(noise_std, random_state=random_state)

    def set_crosstalk_noise(self, flag: bool = True) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.set_crosstalk_noise(flag)

    def set_weight_noise(self, noise_std: float = 0.005) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.set_weight_noise(noise_std)

    def set_flip_ratio(self, flip_ratio: float = 0.0001) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.set_flip_ratio(flip_ratio)

    # read locking index and locking centers from .pkl file
    def perform_lock(self, L_K: dict, G_size: dict, W_K: dict):
        for name, layer in self.named_modules():
            if isinstance(layer, self._conv_linear):
                layer.perform_lock(L_K=L_K[name], W_K=W_K[name], G_size=G_size[name])

    def calculate_signature(self, G_size):
        for name, layer in self.named_modules():
            if isinstance(layer, self._conv_linear):
                if isinstance(G_size, dict):
                    layer.calculate_signature(G_size=G_size[name])
                elif isinstance(G_size, int):
                    layer.calculate_signature(G_size=G_size)
                else:
                    raise NotImplementedError

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.set_weight_bitwidth(w_bit)

    def get_num_device(self) -> Dict[str, int]:
        total_mrr = 0  # total_mzi = 0
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                total_mrr += layer.in_channel_pad * layer.out_channel_pad

        return {"mrr": total_mrr}

    def load_parameters(self, param_dict: Dict[str, Dict[str, Tensor]]) -> None:
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        """
        for layer_name, layer_param_dict in param_dict.items():
            self.layers[layer_name].load_parameters(layer_param_dict)

    def build_obj_fn(self, X: Tensor, y: Tensor, criterion: Callable) -> Callable:
        def obj_fn(X_cur=None, y_cur=None, param_dict=None):
            if param_dict is not None:
                self.load_parameters(param_dict)
            if X_cur is None or y_cur is None:
                data, target = X, y
            else:
                data, target = X_cur, y_cur
            pred = self.forward(data)
            return criterion(pred, target)

        return obj_fn

    def enable_fast_forward(self) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.enable_fast_forward()

    def disable_fast_forward(self) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.disable_fast_forward()

    def sync_parameters(self, src: str = "weight") -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.sync_parameters(src=src)

    def build_weight(self) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.build_weight()

    def print_parameters(self) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.print_parameters()

    def switch_mode_to(self, mode: str) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.switch_mode_to(mode)

    def clear_phase_bias(self) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.clear_phase_bias()

    def reset_noise_schedulers(self):
        self.phase_variation_scheduler.reset()
        self.global_temp_scheduler.reset()
        self.crosstalk_scheduler.reset()

    def step_noise_scheduler(self, T=1):
        if self.phase_variation_scheduler is not None:
            for _ in range(T):
                self.phase_variation_scheduler.step()

        if self.global_temp_scheduler is not None:
            for _ in range(T):
                self.global_temp_scheduler.step()

    def backup_ideal_weights(self):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer._ideal_weight = layer.weight.detach().clone()

    def cycles(self, x_size):
        x = torch.randn(x_size, device=self.device)
        self.eval()

        def hook(m, inp, out):
            m._input_shape = inp[0].shape

        handles = []
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                handle = layer.register_forward_hook(hook)
                handles.append(handle)
        with torch.no_grad():
            self.forward(x)
        cycles = 0
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                cycles += layer.cycles(layer._input_shape, probe=False)
        for handle in handles:
            handle.remove()
        return cycles

    def probe_cycles(self, num_vectors=None):
        cycles = 0
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                ### it considers remapping already
                cycles += layer.cycles(probe=True, num_vectors=num_vectors)
        return cycles

    def set_enable_ste(self, enable_ste: bool) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer._enable_ste = enable_ste

    def set_noise_flag(self, noise_flag: bool) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer._noise_flag = noise_flag

    def set_enable_remap(self, enable_remap: bool) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.set_enable_remap(enable_remap)

    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
