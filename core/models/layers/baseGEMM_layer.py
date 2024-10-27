from typing import Any, Dict

import torch
from torch import Tensor, nn
from torch.types import Device

__all__ = ["GEMMBaseLayer"]


class GEMMBaseLayer(nn.Module):
    def __init__(
        self,
        *args,
        device: Device = torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.device = device

    def quant(self, quant_flag: bool, quant_bit: int) -> Tensor:
        """
        description: Refer to DOTA, HPCA'24
        """
        raise NotImplementedError

    def self_protection(self, budget: int) -> list[list]:
        """
        description: Performed layer-wise, sort them according to gradients as vulnarable weights
        """
        self.grad = Tensor
        w_grad_topk, w_idx_topk = self.weight.grad.detach().abs().view(-1).topk(budget)
        # convert protect_weight to number-of-ones format
        # protect_weight = weight_conversion()
        return w_grad_topk, w_idx_topk

    def set_weight_noise(self, noise_std: float = 0.005):
        """
        Description: For Noise-aware Training
        """
        self.weight_noise_std = noise_std

    def set_flip_ratio(self, flip_ratio: float = 0.0001):
        """
        Description: For Flip-aware Training
        """
        self.flip_ratio = flip_ratio

    def load_parameters(self, param_dict: Dict[str, Any]):
        for name, param in param_dict.items():
            getattr(self, name).data.copy_(param)

    def forward(self, x: Tensor):
        # Dependent on each function
        raise NotImplementedError

    def extra_repr(self) -> str:
        return ""

    def perform_lock(self, L_K: Tensor, W_K: Tensor, G_size: int, HD_con: int = 100):
        """
        Description: version.2,
        """
        # assert G_size * L_K.data.view(-1).shape[0] == self.weight_quantizer.w_q_com.data.view(-1).shape[0]

        if G_size == 1:
            s_idx = torch.randperm(self.weight_quantizer.w_q_com.numel())[
                :HD_con
            ].tolist()
            s_idx = set(s_idx)
            for i in s_idx:
                self.weight_quantizer.w_q_com.data.view(-1)[i] = W_K[L_K[i]]

        else:
            num_group = (self.weight_quantizer.w_q_com.numel() / G_size).__ceil__()
            if num_group > HD_con:
                s_idx = torch.randperm(num_group)[:HD_con].tolist()
                s_idx = set(s_idx)

                for i in s_idx:
                    self.weight_quantizer.w_q_com.data.view(-1)[i::num_group] = W_K[
                        L_K[i]
                    ]
            else:
                for i in range(num_group):
                    self.weight_quantizer.w_q_com.data.view(-1)[i::num_group] = W_K[
                        L_K[i]
                    ]

        self.weight_quantizer.from_two_com()

    def calculate_signature(self, G_size: int):
        """
        Descripton:
        """
        self.num_group = int(self.weight.data.numel() / G_size)
        if G_size == 1:
            self._MSB = torch.tensor(
                [
                    int(bin(x)[2:].zfill(self.weight_quantizer.N_bits)[0])
                    for x in self.weight_quantizer.w_q_com.data.view(-1).int()
                ]
            )
            self._golden_signature = self._MSB  # ^ self._MSB1
            print(self._golden_signature.shape)
        else:
            signature = []
            for i in range(self.num_group):
                weight_chunk = self.weight_quantizer.w_q_com.data.view(-1)[
                    i :: self.num_group
                ]
                SA = bin(
                    (weight_chunk.sum() / (2**self.weight_quantizer.N_bits))
                    .floor()
                    .int()
                )[-1]
                SB = bin(
                    (weight_chunk.sum() / (2 ** (self.weight_quantizer.N_bits - 1)))
                    .floor()
                    .int()
                )[-1]
                signature.append(int(SA + SB))
            self._golden_signature = torch.tensor(signature)
            print(self._golden_signature.shape)
