"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-24 16:23:50
LastEditors: ScopeX-ASU jiaqigu@asu.edu
LastEditTime: 2023-11-17 03:13:01
"""

from collections import OrderedDict
from typing import List, Union

import torch
from torch import Tensor, nn
from torch.types import Device, _size

from core.models.layers.utils import *
from core.models.quantize.quant_base import Qmodes

from .layers.activation import ReLUN
from .layers.gemm_conv2d import GemmConv2d
from .layers.gemm_linear import GemmLinear
from .sparse_bp_base import SparseBP_Base

__all__ = ["SparseBP_GEMM_CNN"]


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        stride: Union[int, _size] = 1,
        padding: Union[int, _size] = 0,
        dilation: Union[int, _size] = 1,
        groups: int = 1,
        bias: bool = False,
        noise_flag: bool = False,
        quant_flag: bool = True,
        noise_level: float = 0.0,
        output_noise_level: float = 0.0,
        qmode: int = Qmodes.layer_wise,
        N_bits: int = 8,  # Quantization related
        N_bits_a: int = 8,
        scaling_range_in: float = 1.0,
        scaling_range_out: float = 1.0,
        mode: str = "defender",
        act_thres: int = 6,
        device: Device = torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.conv = GemmConv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            noise_flag=noise_flag,
            noise_level=noise_level,
            output_noise_level=output_noise_level,
            quant_flag=quant_flag,
            N_bits=N_bits,
            N_bits_a=N_bits_a,
            scaling_range_in=scaling_range_in,
            scaling_range_out=scaling_range_out,
            mode=mode,
            qmode=qmode,
            device=device,
        )

        self.bn = nn.BatchNorm2d(out_channel)

        self.activation = ReLUN(act_thres, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.bn(self.conv(x)))


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        bias: bool = False,
        activation: bool = True,
        act_thres: int = 6,
        noise_flag: bool = True,
        noise_level: float = 0.0,
        output_noise_level: float = 0.0,
        quant_flag: bool = True,
        N_bits: int = 8,
        N_bits_a: int = -1,
        scaling_range_in: float = 1.0,
        scaling_range_out: float = 1.0,
        device: Device = torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.linear = GemmLinear(
            in_channel,
            out_channel,
            bias=bias,
            noise_flag=noise_flag,
            quant_flag=quant_flag,
            noise_level=noise_level,
            output_noise_level=output_noise_level,
            N_bits=N_bits,
            N_bits_a=N_bits_a,
            scaling_range_in=scaling_range_in,
            scaling_range_out=scaling_range_out,
            device=device,
        )

        self.activation = ReLUN(act_thres, inplace=True) if activation else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SparseBP_GEMM_CNN(SparseBP_Base):
    """Gemm CNN."""

    _conv_linear = (GemmConv2d, GemmLinear)
    _linear = (GemmLinear,)
    _conv = (GemmConv2d,)

    def __init__(
        self,
        img_height: int,
        img_width: int,
        in_channel: int,
        n_class: int,
        kernel_list: List[int] = [32],
        kernel_size_list: List[int] = [3],
        pool_out_size: int = 5,
        stride_list=[1],
        padding_list=[1],
        dilation_list=[1],
        groups=1,
        hidden_list: List[int] = [32],
        act_thres: int = 6,
        noise_flag: bool = False,
        quant_flag: bool = True,
        noise_level: float = 0.0,
        output_noise_level: float = 0.0,
        N_bits: int = 8,  # Quantization related
        N_bits_a: int = 8,
        scaling_range_in: float = 1.0,
        scaling_range_out: float = 1.0,
        mode: str = "defender",
        qmode: int = Qmodes.layer_wise,
        bias: bool = False,
        device: Device = torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.in_channel = in_channel
        self.n_class = n_class
        self.kernel_list = kernel_list
        self.kernel_size_list = kernel_size_list
        self.stride_list = stride_list
        self.padding_list = padding_list
        self.dilation_list = dilation_list
        self.groups = groups

        self.pool_out_size = pool_out_size
        self.hidden_list = hidden_list
        # =========== noise =============================
        self.noise_flag = noise_flag
        self.noise_level = noise_level
        self.output_noise_level = output_noise_level

        # =========== quantization =======================
        self.quant_flag = quant_flag
        self.qmode = qmode
        self.N_bits = N_bits
        self.N_bits_a = N_bits_a
        self.scaling_range_in = scaling_range_in
        self.scaling_range_out = scaling_range_out

        self.mode = mode
        self.act_thres = act_thres
        self.bias = bias
        self.device = device

        self.build_layers()
        self.reset_parameters()

    def build_layers(self):
        self.features = OrderedDict()
        for idx, out_channel in enumerate(self.kernel_list, 0):
            layer_name = "conv" + str(idx + 1)
            in_channel = self.in_channel if (idx == 0) else self.kernel_list[idx - 1]
            self.features[layer_name] = ConvBlock(
                in_channel=in_channel,
                out_channel=out_channel,
                kernel_size=self.kernel_size_list[idx],
                stride=self.stride_list[idx],
                padding=self.padding_list[idx],
                dilation=self.dilation_list[0],
                groups=self.groups,
                bias=self.bias,
                act_thres=self.act_thres,
                noise_flag=self.noise_flag,
                quant_flag=self.quant_flag,
                noise_level=self.noise_level,
                output_noise_level=self.output_noise_level,
                N_bits=self.N_bits,
                N_bits_a=self.N_bits_a,
                scaling_range_in=self.scaling_range_in,
                scaling_range_out=self.scaling_range_out,
                mode=self.mode,
                qmode=self.qmode,
                device=self.device,
            )

        self.features = nn.Sequential(self.features)

        if self.pool_out_size > 0:
            self.pool2d = nn.AdaptiveAvgPool2d(self.pool_out_size)
            feature_size = (
                self.kernel_list[-1] * self.pool_out_size * self.pool_out_size
            )
        else:
            self.pool2d = None
            img_height, img_width = self.img_height, self.img_width
            for layer in self.modules():
                if isinstance(layer, GemmConv2d):
                    img_height, img_width = layer.get_output_dim(img_height, img_width)
            feature_size = img_height * img_width * self.kernel_list[-1]

        self.classifier = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_list, 0):
            layer_name = "fc" + str(idx + 1)
            in_channel = feature_size if idx == 0 else self.hidden_list[idx - 1]
            out_channel = hidden_dim
            self.classifier[layer_name] = LinearBlock(
                in_channel,
                out_channel,
                bias=self.bias,
                activation=True,
                act_thres=self.act_thres,
                noise_flag=self.noise_flag,
                quant_flag=self.quant_flag,
                noise_level=self.noise_level,
                output_noise_level=self.output_noise_level,
                N_bits=self.N_bits,
                N_bits_a=self.N_bits_a,
                scaling_range_in=self.scaling_range_in,
                scaling_range_out=self.scaling_range_out,
                device=self.device,
            )

        layer_name = "fc" + str(len(self.hidden_list) + 1)
        self.classifier[layer_name] = GemmLinear(
            self.hidden_list[-1] if len(self.hidden_list) > 0 else feature_size,
            self.n_class,
            bias=self.bias,
            noise_flag=self.noise_flag,
            quant_flag=self.quant_flag,
            noise_level=self.noise_level,
            output_noise_level=self.output_noise_level,
            N_bits=self.N_bits,
            N_bits_a=self.N_bits_a,
            scaling_range_in=self.scaling_range_in,
            scaling_range_out=self.scaling_range_out,
            device=self.device,
        )
        self.classifier = nn.Sequential(self.classifier)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        if self.pool2d is not None:
            x = self.pool2d(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
