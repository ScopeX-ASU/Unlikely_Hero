import math

import torch
from pyutils.quantize import uniform_quantize_new
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.types import _size
from torch.utils.data import WeightedRandomSampler

from .quant_base import Qmodes, grad_scale, round_pass

__all__ = ["weight_quantizer_fn", "input_quantizer_fn", "output_quantizer_fn"]


class weight_quantizer_fn(torch.nn.Module):
    def __init__(
        self,
        size: _size,
        quant_flag: bool,
        qmode: Qmodes,
        N_bits: int,
        flip_ratio: float = 0.0001,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.quant_flag = quant_flag
        self.qmode = qmode
        self.N_bits = N_bits
        self.flip_ratio = flip_ratio
        self.device = device

        self.size = size
        self.w_q = Parameter(torch.Tensor(size=self.size))

        self.b_w = (
            2
            ** torch.linspace(
                start=self.N_bits - 1, end=0, steps=self.N_bits, device=self.device
            ).int()
        )

        if self.N_bits < 0:
            self.register_parameter("alpha", None)

        if self.qmode == Qmodes.kernel_wise:  # Channel-wise
            self.alpha = Parameter(torch.Tensor(self.size[0]))
        else:  # layer-wise quantization
            self.alpha = Parameter(torch.Tensor(1))

        self.register_buffer("init_state", torch.zeros(1))

    def forward(self, weight: Tensor):
        if self.quant_flag and self.training:
            assert self.N_bits >= 1, "N_bits value error"

            Qn = -(2 ** (self.N_bits - 1)) + 1 if self.N_bits > 1 else 0
            Qp = 2 ** (self.N_bits - 1) - 1 if self.N_bits > 1 else 1
            if self.init_state == 0:
                print(
                    f" Layer (mode: {self.qmode}): initialize weight scale for int{self.N_bits} quantization"
                )

                self.alpha.data.copy_(2 * weight.abs().mean() / math.sqrt(Qp))
                self.init_state.fill_(1)

            with torch.no_grad():
                g = 1.0 / math.sqrt(weight.numel() * Qp)

            self.alpha.data.clamp_(min=1e-4)
            # Method1: 31GB GPU memory (AlexNet w4a4 bs 2048)
            alpha = grad_scale(self.alpha, g).to(self.device)
            # print(weight.device, alpha.device)
            quantized_w = (weight / alpha).clamp(Qn, Qp)

            if self.flip_ratio >= 1e-6:
                flip_num = (weight.numel() * self.flip_ratio).__ceil__()
                # print(f"{flip_num} can be flipped in this layer")
                sampler = WeightedRandomSampler(
                    weights=torch.ones_like(weight.view(-1)),
                    num_samples=flip_num,
                    replacement=False,
                )
                flip_idx = [idx for idx in sampler]
                quantized_w.view(-1)[flip_idx].data.copy_(
                    (
                        quantized_w.view(-1)[flip_idx].int() ^ (1 << (self.N_bits - 1))
                    ).float()
                )
                w_q = round_pass(quantized_w) * alpha
            else:
                # print(f"Flip ratio is {self.flip_ratio}")
                w_q = round_pass((weight / alpha).clamp(Qn, Qp)) * alpha
            # Only store this self.w_q in DRAM, do not store
            self.w_q.data.copy_(w_q).to(self.device)
            return w_q

        elif not self.training:  # Only use w_q if inference
            return self.w_q
        else:
            return weight

    def to_two_com(self) -> Tensor:
        """
        Description: Return two's complemetary representation
        """
        w_q_com = self.w_q.data / self.alpha.data
        self.w_q_com = torch.where(w_q_com < 0, w_q_com + 2**self.N_bits, w_q_com)
        return self.w_q_com

    def from_two_com(self) -> Tensor:
        """
        Description: Return original int representation
        """
        assert self.w_q_com is not None, "The data are already in original format"
        self.w_q_ori = torch.where(
            self.w_q_com > 2 ** (self.N_bits - 1),
            self.w_q_com - 2**self.N_bits,
            self.w_q_com,
        )
        # return self.w_q_ori
        self.w_q.data.copy_(self.w_q_ori * self.alpha)
        return self.w_q


class input_quantizer_fn(torch.nn.Module):
    def __init__(
        self,
        quant_flag: bool,
        N_bits_a: int,
        scaling_range: float = 1.0,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.quant_flag = quant_flag
        self.N_bits_a = N_bits_a
        self.scaling_range = scaling_range
        self.device = device

        self.uniform_q = uniform_quantize_new(k=self.N_bits_a)
        self.scale = None
        self.zp = None

        if 1 <= self.N_bits_a <= 8:  # observer does not support higher than 8-bit
            self.obs = torch.quantization.observer.MovingAverageMinMaxObserver(
                averaging_constant=0.01,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                reduce_range=False,
                quant_min=0,  # ReLU will refine the values to be larger than 0
                quant_max=2**self.N_bits_a - 1,
            ).to(self.device)
        else:
            self.obs = None

    def forward(self, x: Tensor):
        if self.quant_flag:
            if self.obs is not None:
                if self.training:
                    self.obs(x)
                scale, zp = self.obs.calculate_qparams()
                self.scale = (self.scaling_range * scale).to(x)
                self.zp = zp.to(x)
                input_q = self.uniform_q(x, self.scale, self.zp)
            else:
                input_q = x
        else:
            input_q = x

        return input_q


class output_quantizer_fn(torch.nn.Module):
    def __init__(
        self,
        quant_flag: bool,
        N_bits_a: int,
        scaling_range: float = 1.0,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__()
        self.quant_flag = quant_flag
        self.N_bits_a = N_bits_a
        self.scaling_range = scaling_range
        self.device = device

        self.uniform_q = uniform_quantize_new(k=self.N_bits_a)
        self.scale = None
        self.zp = None

        if 1 <= self.N_bits_a <= 8:  # observer does not support higher than 8-bit
            self.obs = torch.quantization.observer.MovingAverageMinMaxObserver(
                averaging_constant=0.01,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                reduce_range=False,
                quant_min=0,
                quant_max=2 ** (self.N_bits_a - 0) - 1,
            ).to(self.device)
        else:
            self.obs = None

    def forward(self, x: Tensor) -> Tensor:
        if self.quant_flag:
            if self.obs is not None:
                if self.training:
                    self.obs(x)
                scale, zp = self.obs.calculate_qparams()
                self.scale = (self.scaling_range * scale).to(x)
                self.zp = zp.to(x)
                output_q = self.uniform_q(x, self.scale, self.zp)
            else:
                output_q = x
        else:
            output_q = x
        return output_q
