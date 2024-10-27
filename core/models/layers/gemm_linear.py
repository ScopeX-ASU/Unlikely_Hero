from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, init
from torch.types import Device, _size

# from core.models.quantize.quant_base import Qmodes, grad_scale, round_pass
from core.models.quantize.quantizer import (
    input_quantizer_fn,
    output_quantizer_fn,
    weight_quantizer_fn,
)

from .baseGEMM_layer import GEMMBaseLayer
from .utils import Noise_scheduler

__all__ = ["GemmLinear"]


def gemm_bp_linear(
    x: Tensor,
    weight: Tensor,
    weight_size: _size,
    bias: Optional[Tensor],
    mode: str = "defender",
    noise_scheduler: Noise_scheduler = None,
    weight_quantizer: weight_quantizer_fn = None,
    input_quantizer: input_quantizer_fn = None,
    grad_out_quantizer: input_quantizer_fn = None,
    output_quantizer: output_quantizer_fn = None,
    grad_input_quantizer: output_quantizer_fn = None,
    grad_weight_quantizer: output_quantizer_fn = None,
) -> Tensor:
    """fc layer"""
    assert weight.shape == weight_size, "invalid weight shape"
    # assert bias is None or bias is Tensor, 'invalid bias'
    assert mode in [
        "attacker",
        "defender",
    ], f"Only support attacker and defender, but got {mode}"

    class GemmLinearFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: Tensor, weight: Tensor) -> Tensor:
            with torch.no_grad():
                w_q = weight_quantizer(weight)
                x_q = input_quantizer(x)
                if mode == "attacker":
                    x_noisy = noise_scheduler.add_input_noise(x_q)
                    weight_noisy = noise_scheduler.add_weight_noise(w_q)
                else:
                    x_noisy = x_q
                    weight_noisy = w_q

                out = F.linear(input=x_noisy, weight=weight_noisy, bias=bias)

                if mode == "attacker":
                    out = noise_scheduler.add_output_noise(x=out)
                else:
                    out = out

                out = output_quantizer(out)

                ctx.save_for_backward(x_q, w_q)
            return out

        @staticmethod
        def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
            grad_input = grad_weight = None
            x_q, w_q = ctx.saved_tensors

            grad_output = grad_out_quantizer(grad_output)
            if mode == "attacker":
                grad_output_1 = noise_scheduler.add_input_noise(grad_output)
                grad_output_2 = noise_scheduler.add_input_noise(grad_output)
            else:
                grad_output_1, grad_output_2 = grad_output, grad_output

            if ctx.needs_input_grad[0]:
                if mode == "attacker":
                    weight = noise_scheduler.add_weight_noise(w_q)
                else:
                    weight = w_q

                grad_input = grad_output_1.matmul(weight)

            if ctx.needs_input_grad[1]:
                if mode == "attacker":
                    x = noise_scheduler.add_input_noise(x_q)
                else:
                    x = x_q

                grad_weight = grad_output_2.t().matmul(x)

            if (grad_input is not None) and (mode == "attacker"):
                grad_input = noise_scheduler.add_output_noise(grad_input)
            if (grad_output is not None) and (mode == "attacker"):
                grad_weight = noise_scheduler.add_output_noise(grad_weight)

            if grad_input is not None:
                grad_input = grad_input_quantizer(grad_input)
            grad_weight = grad_weight_quantizer(grad_weight)

            return grad_input, grad_weight

    return GemmLinearFunction.apply(x, weight)


class GemmLinear(GEMMBaseLayer):
    """
    description:
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    miniblock: int
    weight: Tensor
    __annotations__ = {"bias": Optional[Tensor]}

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        noise_flag: bool = False,
        noise_level: float = 0.0,
        output_noise_level: float = 0.0,
        N_bits: int = 8,
        quant_flag: bool = False,
        N_bits_a: int = -1,
        scaling_range_in: float = 1.0,
        scaling_range_out: float = 1.0,
        mode: str = "defender",
        device: Device = torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    ) -> None:
        super(GemmLinear, self).__init__(device=device)

        self.in_features = in_features
        self.out_features = out_features

        if bias:
            self.bias = Parameter(torch.Tensor(out_features).to(self.device))
        else:
            self.register_parameter("bias", None)

        self.noise_flag = noise_flag
        self.noise_level = noise_level
        self.output_noise_level = output_noise_level

        self.noise_scheduler = Noise_scheduler(
            self.noise_flag,
            noise_level=self.noise_level,
            out_noise_level=self.noise_level,
            # FIXME: noise_level should not be equal to out noise level
        )

        # Quantization related
        self.quant_flag = quant_flag
        self.N_bits_a = N_bits_a
        self.N_bits = N_bits
        self.scaling_range_in = scaling_range_in
        self.scaling_range_out = scaling_range_out
        self.mode = mode

        self.weight_noise_std = 0.005
        self.flip_ratio = 0.0001

        self.input_qunatizer = input_quantizer_fn(
            self.quant_flag,
            self.N_bits_a,
            scaling_range=self.scaling_range_in,
            device=self.device,
        )
        self.weight_quantizer = weight_quantizer_fn(
            size=[self.out_features, self.in_features],
            quant_flag=self.quant_flag,
            qmode=None,
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
        weight = torch.empty(self.out_features, self.in_features)
        self.weight = Parameter(weight)

    def reset_parameters(self) -> None:
        init.kaiming_normal_(self.weight.data)

    def load_parameters(self, param_dict: Dict[str, Any]) -> None:
        super().load_parameters(param_dict=param_dict)

    def build_weight(self) -> Tensor:
        if self.weight is not None:
            weight = self.weight
        else:
            weight = torch.rand(size=[self.out_features, self.in_features])
        if self.weight_noise_std > 1e-6:
            weight = weight * (1 + torch.randn_like(weight) * self.weight_noise_std)

        return weight

    def forward(self, x: Tensor):
        if self.weight is None:
            weight = self.build_weight()
        else:
            if self.weight_noise_std >= 1e-6:
                weight = self.weight * (
                    1 + torch.randn_like(self.weight) * self.weight_noise_std
                )
            else:
                weight = self.weight

        out = gemm_bp_linear(
            x=x,
            weight=weight,
            weight_size=(self.out_features, self.in_features),
            bias=self.bias,
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


# class _ActQ(nn.Module):
#     def __init__(self, in_features, **kwargs_q):
#         super(_ActQ, self).__init__()
#         self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
#         self.nbits = kwargs_q['nbits']
#         if self.nbits < 0:
#             self.register_parameter('alpha', None)
#             # self.register_parameter('zero_point', None)
#             return
#         # self.signed = kwargs_q['signed']
#         self.q_mode = kwargs_q['mode']
#         # print(kwargs_q)
#         self.offset = kwargs_q['offset']
#         self.zero_point = None
#         if self.q_mode == Qmodes.kernel_wise:
#             self.alpha = Parameter(torch.Tensor(in_features))
#             if self.offset:
#                 self.zero_point = Parameter(torch.Tensor(in_features))
#                 torch.nn.init.zeros_(self.zero_point)
#         else:
#             self.alpha = Parameter(torch.Tensor(1)) # has a initial value, but not 0
#             if self.offset:
#                 self.zero_point = Parameter(torch.Tensor([0]))  # initial value of zero_point is 0
#         # self.zero_point = Parameter(torch.Tensor([0]))
#         self.register_buffer('init_state', torch.zeros(1))
#         self.register_buffer('signed', torch.zeros(1))

#     def add_param(self, param_k, param_v):
#         self.kwargs_q[param_k] = param_v

#     def set_bit(self, nbits):
#         self.kwargs_q['nbits'] = nbits

#     def extra_repr(self):
#         # s_prefix = super(_ActQ, self).extra_repr()
#         if self.alpha is None:
#             return 'fake'
#         return '{}'.format(self.kwargs_q)

# class QuantAct(_ActQ):
#     def __init__(self, in_features, nbits=-1, signed=True, mode=Qmodes.layer_wise, offset=False, **kwargs):
#         super(QuantAct, self).__init__(in_features=in_features,
#                                        nbits=nbits, mode=mode, offset=offset)
#         self.offset = offset
#         self.signed = signed

#     def forward(self, x: Tensor):
#         if self.alpha is None:
#             return x

#         if self.training and self.init_state == 0:
#             print(
#                 f"Act layer: (mode: {self.q_mode}): initialize weight scale for int{self.nbits} quantization with offset: {self.offset}")
#             if self.q_mode == Qmodes.kernel_wise:
#                 print(f'Scale dimension: {self.alpha.shape}')
#             # choose implementation from https://github.com/YanjingLi0202/Q-ViT/blob/main/Quant.py
#             if x.min() < -1e-5:
#                 self.signed.data.fill_(1)

#             if self.signed == 1:    # Sysmetric quant, -127 ~ 127 etc.
#                 Qn = -2 ** (self.nbits - 1) + 1
#                 Qp = 2 ** (self.nbits - 1) - 1
#             else:                   # 0 ~ 255 etc.
#                 Qn = 0
#                 Qp = 2 ** self.nbits - 1

#             self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
#             if self.offset:
#                 self.zero_point.data.copy_(
#                     self.zero_point.data * 0.9 + 0.1 * (torch.min(x.detach()) - self.alpha.data * Qn))
#             self.init_state.fill_(1)

#         assert self.init_state == 1     # initial state changes

#         if self.signed:
#             Qn = -2 ** (self.nbits - 1) + 1
#             Qp = 2 ** (self.nbits - 1) - 1
#         else:
#             Qn = 0
#             Qp = 2 ** self.nbits - 1
#         with torch.no_grad():
#             g = 1.0 / math.sqrt(x.numel() * Qp)

#         self.alpha.data.clamp_(min=1e-4)    # cannot devide a zero-value
#         # Method1:
#         alpha = grad_scale(self.alpha, g)   # grad_scale ?

#         if self.offset:
#             zero_point = (self.zero_point.round() -
#                           self.zero_point).detach() + self.zero_point
#             zero_point = grad_scale(zero_point, g)
#             zero_point = zero_point.unsqueeze(0) if len(
#                 x.shape) == 2 else zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)
#         else:
#             zero_point = 0

#         if len(x.shape) == 2:
#             alpha = alpha.unsqueeze(0)
#             # zero_point = zero_point.unsqueeze(0)
#         elif len(x.shape) == 4:
#             alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)

#         x = round_pass((x / alpha + zero_point).clamp(Qn, Qp))
#         x = (x - zero_point) * alpha

#         # Method2:
#         # x = FunLSQ.apply(x, self.alpha, g, Qn, Qp)
#         return x


# def get_default_kwargs_q(kwargs_q, layer_type):
#     default = {
#         'nbits': 4
#     }
#     if isinstance(layer_type, GemmLinear):
#         default.update({
#             'mode': Qmodes.layer_wise})
#     elif isinstance(layer_type, _ActQ):
#         pass
#         # default.update({
#         #     'signed': 'Auto'})
#     else:
#         assert NotImplementedError
#         return
#     for k, v in default.items():
#         if k not in kwargs_q:
#             kwargs_q[k] = v
#     return kwargs_q
