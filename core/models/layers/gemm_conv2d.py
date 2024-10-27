from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, init
from torch.nn.modules.utils import _pair
from torch.types import Device, _size

from core.models.quantize.quant_base import Qmodes
from core.models.quantize.quantizer import (
    input_quantizer_fn,
    output_quantizer_fn,
    weight_quantizer_fn,
)

from .baseGEMM_layer import GEMMBaseLayer
from .utils import Noise_scheduler

__all__ = ["GemmConv2d"]


# Self customized conv2d function, because normal F.conv2d() will cause the gradient calculation involves
# the same noise in weight, so self customized conv2d is needed to add different noise
def gemm_bp_conv2d_function(
    x: Tensor,
    weight: Tensor,  # weight data
    weight_size: _size,  # required weight size [outc, inc, k, k]
    stride: Union[int, _size] = 1,
    padding: Union[int, _size] = 0,
    dilation: Union[int, _size] = 1,
    groups: int = 1,
    mode: str = "defender",
    noise_scheduler: Noise_scheduler = None,
    weight_quantizer: weight_quantizer_fn = None,
    input_quantizer: input_quantizer_fn = None,
    grad_out_quantizer: input_quantizer_fn = None,
    output_quantizer: output_quantizer_fn = None,
    grad_input_quantizer: output_quantizer_fn = None,
    grad_weight_quantizer: output_quantizer_fn = None,
):
    """Support gradient calculation w.r.t weight"""
    assert (
        weight.shape == weight_size
    ), f"weights are in invalid shape, weight shape is {weight.shape}, expected is {weight_size}"
    assert mode in [
        "attacker",
        "defender",
    ], f"Only support attacker and defender, but got {mode}"

    class GemmConv2dFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: Tensor, weight: Tensor) -> Tensor:
            with torch.no_grad():
                # if weight.shape != weight_size: # p, q, k, k block shape
                # Assume weight is in [outc, inc, k, k] already
                w_q = weight_quantizer(weight)
                # print(f"Sampled quantized wieghts {w_q.mean()}")
                x_q = input_quantizer(x)

                if mode == "attacker":
                    x_noisy = noise_scheduler.add_input_noise(x_q)
                    weight_noisy = noise_scheduler.add_weight_noise(w_q)
                else:
                    x_noisy = x_q
                    weight_noisy = w_q

                # FIXME
                out = F.conv2d(
                    x_noisy,
                    weight=weight_noisy,
                    bias=None,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                if mode == "attacker":
                    out = noise_scheduler.add_output_noise(out)
                else:
                    out = out

                out = output_quantizer(out)

                ctx.input_size = x.size()
                ctx.save_for_backward(x_q, w_q)
                # torch.cuda.empty_cache() # slow down

            return out  # [bs, outc, h_out, w_out] with random noise

        @staticmethod
        def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
            # shape of grad_output is [bs, outc, hout, wout]
            grad_input = grad_weight = None

            # FIXME: SAVE x, weight or x_q, w_q ?
            x_q, w_q = ctx.saved_tensors

            grad_output = grad_out_quantizer(grad_output)

            if mode == "attacker":
                grad_output_1 = noise_scheduler.add_input_noise(grad_output)
                grad_output_2 = noise_scheduler.add_input_noise(grad_output)
            else:
                grad_output_1 = grad_output
                grad_output_2 = grad_output

            if ctx.needs_input_grad[0]:
                if mode == "attacker":
                    weight = noise_scheduler.add_weight_noise(w_q)
                else:
                    weight = w_q
                # calculate dy/dx, for the input of prior layer
                grad_input = torch.nn.grad.conv2d_input(
                    ctx.input_size,
                    weight,
                    grad_output_1,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )

            if ctx.needs_input_grad[1]:
                if mode == "attacker":
                    x = noise_scheduler.add_input_noise(x=x_q)
                else:
                    x = x_q
                # calculate dy/dw, weight gradient for this layer
                grad_weight = torch.nn.grad.conv2d_weight(
                    x,
                    weight_size,
                    grad_output_2,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )

            if (grad_input is not None) and (mode == "attacker"):
                grad_input = noise_scheduler.add_output_noise(grad_input)
            if (grad_weight is not None) and (mode == "attacker"):
                grad_weight = noise_scheduler.add_output_noise(grad_weight)

            if grad_input is not None:
                grad_input = grad_input_quantizer(grad_input)
            grad_weight = grad_weight_quantizer(grad_weight)

            return grad_input, grad_weight

    return GemmConv2dFunction.apply(x, weight)


class GemmConv2d(GEMMBaseLayer):
    """
    description: General Conv2d in DOTA's architecture
    """

    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
    ]
    __annotations__ = {"bias": Optional[torch.Tensor]}

    _in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size,
        stride: _size = 1,
        padding: _size = 0,
        dilation: _size = 1,
        groups: int = 1,
        bias: bool = True,
        noise_flag: bool = False,
        noise_level: float = 0.0,
        output_noise_level: float = 0.0,
        N_bits: int = 8,
        quant_flag: bool = False,
        N_bits_a: int = 8,
        scaling_range_in: float = 1.0,
        scaling_range_out: float = 1.0,
        mode: str = "defender",
        qmode: int = Qmodes.layer_wise,
        device: Device = torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    ) -> None:
        super(GemmConv2d, self).__init__(device=device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        assert (
            groups == 1
        ), f"Currently group convolution is not supported, but got group: {groups}"

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels).to(self.device))
        else:
            self.register_parameter("bias", None)

        # Noise related parameters and definations
        self.noise_flag = noise_flag
        self.noise_level = noise_level
        self.output_noise_level = output_noise_level
        self.mode = mode

        self.noise_scheduler = Noise_scheduler(
            # noise flag, QAT + NAT
            # back pure, forward + noise, quant, mode: defender, attacker
            self.noise_flag,
            noise_level=self.noise_level,
            out_noise_level=self.output_noise_level,
        )

        # Quantization related parameters and definations
        self.quant_flag = quant_flag
        self.N_bits = N_bits
        self.N_bits_a = N_bits_a
        self.scaling_range_in = scaling_range_in
        self.scaling_range_out = scaling_range_out
        self.qmode = qmode

        self.weight_noise_std = 0.005
        self.flip_ratio = 0.0001

        self.input_qunatizer = input_quantizer_fn(
            self.quant_flag,
            self.N_bits_a,
            scaling_range=self.scaling_range_in,
            device=self.device,
        )
        self.weight_quantizer = weight_quantizer_fn(
            size=[
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[0],
            ],
            quant_flag=self.quant_flag,
            qmode=self.qmode,
            N_bits=self.N_bits,
            flip_ratio=self.flip_ratio,
            device=self.device,
        )
        self.output_quantizer = output_quantizer_fn(
            self.quant_flag,
            self.N_bits_a,
            scaling_range=self.scaling_range_out,
            device=self.device,
        )
        self.grad_out_quantizer = input_quantizer_fn(
            self.quant_flag,
            self.N_bits_a,
            scaling_range=self.scaling_range_in,
            device=self.device,
        )
        self.grad_weight_quantizer = output_quantizer_fn(
            self.quant_flag,
            self.N_bits_a,
            scaling_range=self.scaling_range_out,
            device=self.device,
        )
        self.grad_input_quantizer = output_quantizer_fn(
            self.quant_flag,
            self.N_bits_a,
            scaling_range=self.scaling_range_out,
            device=self.device,
        )

        self.build_parameters()
        self.reset_parameters()

    def build_parameters(self) -> None:
        weight = torch.empty(
            self.out_channels,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[0],
            device=self.device,
        )
        self.weight = Parameter(weight)

    def reset_parameters(self) -> None:
        init.kaiming_normal_(self.weight.data)

    def load_parameters(self, param_dict: Dict[str, Any]) -> None:
        super().load_parameters(param_dict=param_dict)

    def build_weight(self) -> Tensor:
        """
        description:
        """
        # Random initialization of weights
        if self.weight is not None:
            weight = self.weight
        else:
            weight = torch.rand(
                size=[
                    self.out_channels,
                    self.in_channels,
                    self.kernel_size[0],
                    self.kernel_size[0],
                ]
            )
            #
        if self.weight_noise_std > 1e-6:
            # print(f"{self.weight_noise_std}")
            weight = weight * (1 + torch.randn_like(weight) * self.weight_noise_std)
        return weight

    def get_output_dim(self, img_height, img_width):
        h_out = (
            img_height - self.kernel_size[-1] + 2 * self.padding[-1]
        ) / self.stride[-1] + 1
        w_out = (img_width - self.kernel_size[-1] + 2 * self.padding[-1]) / self.stride[
            -1
        ] + 1
        return (int(h_out), int(w_out))

    def forward(self, x: Tensor) -> Tensor:
        if self.weight is None:
            weight = self.build_weight(noise_flag=False)
        else:
            if self.weight_noise_std >= 1e-6:
                weight = self.weight * (
                    1 + torch.randn_like(self.weight) * self.weight_noise_std
                )
                # print(f"self.weight_noise_std is {self.weight_noise_std}")
            else:
                # print(f"self.weight_noise_std is {self.weight_noise_std}")
                weight = self.weight

        out = gemm_bp_conv2d_function(
            x=x,
            weight=weight,
            weight_size=(
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[0],
            ),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            mode=self.mode,
            noise_scheduler=self.noise_scheduler,
            input_quantizer=self.input_qunatizer,
            weight_quantizer=self.weight_quantizer,
            output_quantizer=self.output_quantizer,
            grad_input_quantizer=self.grad_input_quantizer,
            grad_weight_quantizer=self.grad_weight_quantizer,
            grad_out_quantizer=self.grad_out_quantizer,
        )
        return out
