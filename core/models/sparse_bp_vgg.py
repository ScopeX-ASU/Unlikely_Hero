from typing import Union

import torch
from torch import Tensor, nn
from torch.types import Device, _size

from core.models.layers.utils import *
from core.models.quantize.quant_base import Qmodes

from .layers.activation import ReLUN
from .layers.gemm_conv2d import GemmConv2d
from .layers.gemm_linear import GemmLinear
from .sparse_bp_base import SparseBP_Base

__all__ = [
    "SparseBP_GEMM_VGG8",
    "SparseBP_GEMM_VGG11",
    "SparseBP_GEMM_VGG13",
    "SparseBP_GEMM_VGG16",
    "SparseBP_GEMM_VGG19",
]

cfg_32 = {
    "vgg8": [64, "M", 128, "M", 256, "M", 512, "M", 512, "M"],
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "vgg19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}

cfg_64 = {
    "vgg8": [64, "M", 128, "M", 256, "M", 512, "M", 512, "GAP"],
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "GAP"],
    "vgg13": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        "M",
        512,
        512,
        "M",
        512,
        512,
        "GAP",
    ],
    "vgg16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "GAP",
    ],
    "vgg19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "GAP",
    ],
}


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        bias: bool = False,
        stride: Union[int, _size] = 1,
        padding: Union[int, _size] = 0,
        noise_flag: bool = False,
        quant_flag: bool = True,
        noise_level: float = 0.0,
        output_noise_level: float = 0.0,
        qmode: int = Qmodes.layer_wise,
        N_bits: int = 8,
        N_bits_a: int = 8,
        scaling_range_in: float = 1.0,
        scaling_range_out: float = 1.0,
        mode: str = "defender",
        act_thres: int = 6,
        device: Device = torch.device("cuda"),
    ) -> None:
        super().__init__()
        self.conv = GemmConv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
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

        self.activation = (
            ReLUN(act_thres, inplace=True) if act_thres <= 6 else nn.ReLU(inplace=True)
        )

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
        noise_flag: bool = False,
        noise_level: float = 0.0,
        output_noise_level: float = 0.0,
        quant_flag: bool = True,
        N_bits: int = 8,
        N_bits_a: int = 8,
        scaling_range_in: float = 1.0,
        scaling_range_out: float = 1.0,
        mode: str = "defender",
        device: Device = torch.device("cuda"),
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
            mode=mode,
            device=device,
        )

        self.activation = (
            (
                ReLUN(act_thres, inplace=True)
                if act_thres <= 6
                else nn.ReLU(inplace=True)
            )
            if activation
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class VGG(SparseBP_Base):
    """GEMM VGG, the architecture with reference to DOTA HPCA 2024"""

    def __init__(
        self,
        vgg_name: str,
        img_height: int,
        img_width: int,
        in_channel: int,
        n_class: int,
        # block_list: List[int] = [8],
        act_thres: float = 6.0,
        bias: bool = False,
        noise_flag: bool = False,
        quant_flag: bool = True,
        noise_level: float = 0.0,
        output_noise_level: float = 0.0,
        N_bits: int = 8,
        N_bits_a: int = 8,
        scaling_range_in: float = 1.0,
        scaling_range_out: float = 1.0,
        mode: str = "defender",
        qmode: int = Qmodes.layer_wise,
        device: Device = torch.device("cuda"),
    ) -> None:
        super().__init__()

        self.vgg_name = vgg_name
        self.img_height = img_height
        self.img_width = img_width
        self.in_channel = in_channel
        self.n_class = n_class

        # list of block size
        # self.block_list = block_list

        self.act_thres = act_thres
        self.bias = bias

        # ============ noise ======================
        self.noise_flag = noise_flag
        self.noise_level = noise_level
        self.output_noise_level = output_noise_level

        # ============ quantization ================
        self.quant_flag = quant_flag
        self.N_bits = N_bits
        self.N_bits_a = N_bits_a
        self.qmode = qmode
        self.scaling_range_in = scaling_range_in
        self.scaling_range_out = scaling_range_out

        self.mode = mode
        self.device = device

        self.build_layers()
        self.reset_parameters()

    def build_layers(self):
        cfg = cfg_32 if self.img_height == 32 else cfg_64
        self.features, convNum = self._make_layers(cfg[self.vgg_name])
        # build FC layers
        ## lienar layer use the last miniblock
        if (
            self.img_height == 64 and self.vgg_name == "vgg8"
        ):  ## model is too small, do not use dropout
            classifier = []
        else:
            classifier = [nn.Dropout(0.5)]
        classifier += [
            GemmLinear(
                512,
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
                mode=self.mode,
                device=self.device,
            )
        ]
        self.classifier = nn.Sequential(*classifier)

    def _make_layers(self, cfg):
        layers = []
        in_channel = self.in_channel
        convNum = 0

        for x in cfg:
            # MaxPool2d
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == "GAP":
                layers += [nn.AdaptiveAvgPool2d((1, 1))]
            else:
                # conv + BN + RELU
                layers += [
                    ConvBlock(
                        in_channel,
                        x,
                        kernel_size=3,
                        bias=self.bias,
                        stride=1,
                        padding=1,
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
                        act_thres=self.act_thres,
                        device=self.device,
                    )
                ]
                in_channel = x
                convNum += 1
        return nn.Sequential(*layers), convNum

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def SparseBP_GEMM_VGG8(*args, **kwargs):
    return VGG("vgg8", *args, **kwargs)


def SparseBP_GEMM_VGG11(*args, **kwargs):
    return VGG("vgg11", *args, **kwargs)


def SparseBP_GEMM_VGG13(*args, **kwargs):
    return VGG("vgg13", *args, **kwargs)


def SparseBP_GEMM_VGG16(*args, **kwargs):
    return VGG("vgg16", *args, **kwargs)


def SparseBP_GEMM_VGG19(*args, **kwargs):
    return VGG("vgg19", *args, **kwargs)


def test():
    device = torch.device("cuda")
    net = SparseBP_GEMM_VGG8(
        32,
        32,
        3,
        10,
        [4, 4, 4, 4, 4, 4, 4, 4],
        32,
        32,
        mode="usv",
        v_max=10.8,
        v_pi=4.36,
        act_thres=6.0,
        photodetect=True,
        bias=False,
        device=device,
    ).to(device)

    x = torch.randn(2, 3, 32, 32).to(device)
    print(net)
    y = net(x)
    print(y.shape)


if __name__ == "__main__":
    test()
