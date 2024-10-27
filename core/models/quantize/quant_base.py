'''
Date: 2024-03-21 12:39:39
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-01 14:05:48
FilePath: /ONN_Reliable/core/models/quantize/quant_base.py
'''
import math
from enum import Enum

import torch
from torch import Tensor

__all__ = ["Qmodes", "truncation", "round_pass", "grad_scale"]


class Qmodes(Enum):
    layer_wise = 1
    kernel_wise = 2


# ========== Functions needed for following operations =====
def grad_scale(x: Tensor, scale: Tensor):
    y = x
    y_grad = x * scale
    # ! Why not directly use x * scale? The reason is we need the data of x but the gradient needs to be x * scale
    # which means: scaling factor should maintain, but the gradient of scaling factor should be scaled.
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x: Tensor):
    y = x.round()
    y_grad = x
    # What this function do: need the data after round() but the gradient of original x
    return y.detach() - y_grad.detach() + y_grad


def log_shift(value_fp):
    value_shift = 2 ** (torch.log2(value_fp).ceil())
    return value_shift


def clamp(input: Tensor, min: int, max: int, inplace=False):
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def get_quantized_range(num_bits, signed=True):
    if signed:
        n = 2 ** (num_bits - 1)
        return -n + 1, n - 1
    return 0, 2**num_bits - 1


def linear_quantize(input: Tensor, scale_factor: tuple[Tensor, int], inplace=False):
    if inplace:
        input.mul_(scale_factor).round_()
        return input
    return torch.round(scale_factor * input)


def linear_quantize_clamp(input, scale_factor, clamp_min, clamp_max, inplace=False):
    output = linear_quantize(input, scale_factor, inplace)
    return clamp(output, clamp_min, clamp_max, inplace)


def linear_dequantize(input: Tensor, scale_factor, inplace=False):
    if inplace:
        input.div_(scale_factor)
        return input
    return input / scale_factor


def truncation(fp_data: Tensor, nbits: int = 8):
    il = torch.log2(torch.max(fp_data.max(), fp_data.min().abs())) + 1
    il = math.ceil(il - 1e-5)
    qcode = nbits - il
    scale_factor = 2**qcode
    clamp_min, clamp_max = get_quantized_range(nbits, signed=True)
    q_data = linear_quantize_clamp(fp_data, scale_factor, clamp_min, clamp_max)
    q_data = linear_dequantize(q_data, scale_factor)
    return q_data, qcode


# ============================================================
