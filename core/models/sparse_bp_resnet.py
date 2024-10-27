from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.activation import ReLU
from torch.types import Device, _size

from core.models.layers.activation import ReLUN
from core.models.layers.gemm_conv2d import GemmConv2d
from core.models.layers.gemm_linear import GemmLinear
from core.models.layers.utils import *
from core.models.quantize.quant_base import Qmodes
from core.models.sparse_bp_base import SparseBP_Base

__all__ = [
    "SparseBP_GEMM_ResNet18",
    "SparseBP_GEMM_ResNet20",
    "SparseBP_GEMM_ResNet32",
    "SparseBP_GEMM_ResNet34",
    "SparseBP_GEMM_ResNet50",
    "SparseBP_GEMM_ResNet101",
    "SparseBP_GEMM_ResNet152",
]


def conv3x3(
    in_planes,
    out_planes,
    stride: Union[int, _size] = 1,
    padding: Union[int, _size] = 0,
    bias: bool = False,
    # unique parameters, noise
    noise_flag: bool = False,
    noise_level: float = 0.0,
    output_noise_level: float = 0.0,
    # unique parameters, quantization
    quant_flag: bool = True,
    N_bits: int = 8,
    N_bits_a: int = 8,
    scaling_range_in: float = 1.0,
    scaling_range_out: float = 1.0,
    qmode: int = Qmodes.layer_wise,
    # unique parameters, model
    mode: str = "defender",
    device: Device = torch.device("cuda"),
):
    conv = GemmConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        # unique parameters, noise
        noise_flag=noise_flag,
        noise_level=noise_level,
        output_noise_level=output_noise_level,
        # unique parameters, quantization
        quant_flag=quant_flag,
        N_bits=N_bits,
        N_bits_a=N_bits_a,
        scaling_range_in=scaling_range_in,
        scaling_range_out=scaling_range_out,
        qmode=qmode,
        # unqiue parameters, model
        mode=mode,
        device=device,
    )
    return conv


def conv1x1(
    in_planes,
    out_planes,
    stride: Union[int, _size] = 1,
    padding: Union[int, _size] = 0,
    bias: bool = False,
    # unique parameters, noise
    noise_flag: bool = False,
    noise_level: float = 0.0,
    output_noise_level: float = 0.0,
    # unique parameters, quantization
    quant_flag: bool = True,
    N_bits: int = 8,
    N_bits_a: int = 8,
    scaling_range_in: float = 1.0,
    scaling_range_out: float = 1.0,
    qmode: int = Qmodes.layer_wise,
    # unique parameters, model
    mode: str = "defender",
    device: Device = torch.device("cuda"),
):
    conv = GemmConv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=padding,
        bias=bias,
        # unique parameters, noise
        noise_flag=noise_flag,
        noise_level=noise_level,
        output_noise_level=output_noise_level,
        # unique parameters, quantization
        quant_flag=quant_flag,
        N_bits=N_bits,
        N_bits_a=N_bits_a,
        scaling_range_in=scaling_range_in,
        scaling_range_out=scaling_range_out,
        qmode=qmode,
        # unqiue parameters, model
        mode=mode,
        device=device,
    )

    return conv


def Linear(
    in_channel,
    out_channel,
    bias: bool = False,
    # noise
    noise_flag: bool = False,
    noise_level: float = 0.0,
    output_noise_level: float = 0.0,
    # quantization
    quant_flag: bool = True,
    N_bits: int = 8,
    N_bits_a: int = 8,
    scaling_range_in: float = 1.0,
    scaling_range_out: float = 1.0,
    # model
    mode: str = "defender",
    device: Device = torch.device("cuda"),
):
    linear = GemmLinear(
        in_channel,
        out_channel,
        bias=bias,
        # unique parameters, noise
        noise_flag=noise_flag,
        noise_level=noise_level,
        output_noise_level=output_noise_level,
        # unique parameters, quantization
        quant_flag=quant_flag,
        N_bits=N_bits,
        N_bits_a=N_bits_a,
        scaling_range_in=scaling_range_in,
        scaling_range_out=scaling_range_out,
        # unique parameters, model
        mode=mode,
        device=device,
    )
    return linear


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        # unique parameters, noise
        noise_flag: bool = False,
        noise_level: float = 0.0,
        output_noise_level: float = 0.0,
        # quantization
        quant_flag: bool = True,
        N_bits: int = 8,
        N_bits_a: int = 8,
        scaling_range_in: float = 1.0,
        scaling_range_out: float = 1.0,
        qmode: int = Qmodes.layer_wise,
        # model
        mode: str = "defender",
        act_thres: int = 6,
        device: Device = torch.device("cuda"),
    ) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(
            in_planes,
            planes,
            stride=stride,
            padding=1,
            bias=False,
            # noise
            noise_flag=noise_flag,
            noise_level=noise_level,
            output_noise_level=output_noise_level,
            # quantization param
            quant_flag=quant_flag,
            N_bits=N_bits,
            N_bits_a=N_bits_a,
            scaling_range_in=scaling_range_in,
            scaling_range_out=scaling_range_out,
            qmode=qmode,
            # model param
            mode=mode,
            device=device,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = (
            ReLUN(act_thres, inplace=True) if act_thres <= 6 else ReLU(inplace=True)
        )
        self.conv2 = conv3x3(
            planes,
            planes,
            stride=1,
            padding=1,
            bias=False,
            # noise param
            noise_flag=noise_flag,
            noise_level=noise_level,
            output_noise_level=output_noise_level,
            # quantization param
            quant_flag=quant_flag,
            N_bits=N_bits,
            N_bits_a=N_bits_a,
            scaling_range_in=scaling_range_in,
            scaling_range_out=scaling_range_out,
            qmode=qmode,
            # model param
            mode=mode,
            device=device,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = (
            ReLUN(act_thres, inplace=True) if act_thres <= 6 else ReLU(inplace=True)
        )

        self.shortcut = nn.Identity()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(
                    in_planes,
                    self.expansion * planes,
                    stride=stride,
                    padding=0,
                    bias=False,
                    # unique parameters,noise
                    noise_flag=noise_flag,
                    noise_level=noise_level,
                    output_noise_level=output_noise_level,
                    # quantization param
                    quant_flag=quant_flag,
                    N_bits=N_bits,
                    N_bits_a=N_bits_a,
                    scaling_range_in=scaling_range_in,
                    scaling_range_out=scaling_range_out,
                    qmode=qmode,
                    # model param
                    mode=mode,
                    device=device,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        act_thres: int = 6,
        # unique parameters, noise
        noise_flag: bool = False,
        noise_level: float = 0.0,
        output_noise_level: float = 0.0,
        # quant param
        quant_flag: bool = True,
        N_bits: int = 8,
        N_bits_a: int = 8,
        scaling_range_in: float = 1.0,
        scaling_range_out: float = 1.0,
        qmode: int = Qmodes.layer_wise,
        # model param
        mode: str = "defender",
        device: Device = torch.device("cuda"),
    ) -> None:
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(
            in_planes,
            planes,
            stride=1,
            padding=0,
            bias=False,
            # noise param
            noise_flag=noise_flag,
            noise_level=noise_level,
            output_noise_level=output_noise_level,
            # quant param
            quant_flag=quant_flag,
            N_bits=N_bits,
            N_bits_a=N_bits_a,
            scaling_range_in=scaling_range_in,
            scaling_range_out=scaling_range_out,
            qmode=qmode,
            # model param
            mode=mode,
            device=device,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = (
            ReLUN(act_thres, inplace=True) if act_thres <= 6 else ReLU(inplace=True)
        )
        self.conv2 = conv3x3(
            planes,
            planes,
            stride=stride,
            padding=1,
            bias=False,
            # noise param
            noise_flag=noise_flag,
            noise_level=noise_level,
            output_noise_level=output_noise_level,
            # quant param
            quant_flag=quant_flag,
            N_bits=N_bits,
            N_bits_a=N_bits_a,
            scaling_range_in=scaling_range_in,
            scaling_range_out=scaling_range_out,
            qmode=qmode,
            # model param
            mode=mode,
            device=device,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = (
            ReLUN(act_thres, inplace=True) if act_thres <= 6 else ReLU(inplace=True)
        )
        self.conv3 = conv1x1(
            planes,
            self.expansion * planes,
            stride=1,
            padding=0,
            bias=False,
            # noise param
            noise_flag=noise_flag,
            noise_level=noise_level,
            output_noise_level=output_noise_level,
            # quant param
            quant_flag=quant_flag,
            N_bits=N_bits,
            N_bits_a=N_bits_a,
            scaling_range_in=scaling_range_in,
            scaling_range_out=scaling_range_out,
            qmode=qmode,
            # model param
            mode=mode,
            device=device,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.act3 = (
            ReLUN(act_thres, inplace=True) if act_thres <= 6 else ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(
                    in_planes,
                    self.expansion * planes,
                    stride=stride,
                    padding=0,
                    bias=False,
                    # noise param
                    noise_flag=noise_flag,
                    noise_level=noise_level,
                    output_noise_level=output_noise_level,
                    # quant param
                    quant_flag=quant_flag,
                    N_bits=N_bits,
                    N_bits_a=N_bits_a,
                    scaling_range_in=scaling_range_in,
                    scaling_range_out=scaling_range_out,
                    qmode=qmode,
                    # model param
                    mode=mode,
                    device=device,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.act3(out)
        return out


class ResNet(SparseBP_Base):
    """MRR ResNet (Shen+, Nature Photonics 2017). Support sparse backpropagation. Blocking matrix multiplication."""

    def __init__(
        self,
        block,
        num_blocks,
        in_planes,
        img_height: int,
        img_width: int,
        in_channel: int,
        n_class: int,
        bias: bool = False,
        act_thres: float = 6.0,
        # unique parameters, noise
        noise_flag: bool = False,
        noise_level: float = 0.0,
        output_noise_level: float = 0.0,
        # quant param
        quant_flag: bool = True,
        N_bits: int = 8,
        N_bits_a: int = 8,
        scaling_range_in: float = 1.0,
        scaling_range_out: float = 1.0,
        qmode: int = Qmodes.layer_wise,
        # model param
        mode: str = "defender",
        device: Device = torch.device("cuda"),
    ) -> None:
        super().__init__()

        # resnet params
        self.block = block
        self.num_blocks = num_blocks
        self.in_planes = in_planes
        self.img_height = img_height
        self.img_width = img_width

        self.in_channel = in_channel
        self.n_class = n_class

        # list of block size
        self.bias = bias
        self.act_thres = act_thres

        # unique parameters, noise:
        self.noise_flag = noise_flag
        self.noise_level = noise_level
        self.output_noise_level = output_noise_level

        # quant params
        self.quant_flag = quant_flag
        self.N_bits = N_bits
        self.N_bits_a = N_bits_a
        self.scaling_range_in = scaling_range_in
        self.scaling_range_out = scaling_range_out
        self.qmode = qmode
        self.mode = mode

        self.device = device

        # build layers
        blkIdx = 0
        self.conv1 = conv3x3(
            in_channel,
            self.in_planes,
            stride=1
            if self.img_height <= 64
            else 2,  # downsample for imagenet, dogs, cars, changed to self.img_height
            padding=1,
            bias=False,
            # unique parameters
            noise_flag=self.noise_flag,
            noise_level=self.noise_level,
            output_noise_level=self.output_noise_level,
            quant_flag=self.quant_flag,
            N_bits=self.N_bits,
            N_bits_a=self.N_bits_a,
            scaling_range_in=self.scaling_range_in,
            scaling_range_out=self.scaling_range_out,
            qmode=self.qmode,
            mode=self.mode,
            device=self.device,
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        blkIdx += 1

        self.layer1 = self._make_layer(
            block,
            in_planes,
            num_blocks[0],
            stride=1,
            device=device,
        )
        blkIdx += 1

        self.layer2 = self._make_layer(
            block,
            in_planes * 2,
            num_blocks[1],
            stride=2,
            device=device,
        )
        blkIdx += 1

        self.layer3 = self._make_layer(
            block,
            in_planes * 4,
            num_blocks[2],
            stride=2,
            device=device,
        )
        blkIdx += 1

        self.layer4 = self._make_layer(
            block,
            in_planes * 8,
            num_blocks[3],
            stride=2,
            device=device,
        )
        blkIdx += 1

        n_channel = in_planes * 8 if num_blocks[3] > 0 else in_planes * 4
        self.linear = Linear(
            n_channel * block.expansion,
            self.n_class,
            bias=False,
            # unique parameters
            noise_flag=self.noise_flag,
            noise_level=self.noise_level,
            output_noise_level=self.output_noise_level,
            quant_flag=self.quant_flag,
            N_bits=self.N_bits,
            N_bits_a=self.N_bits_a,
            scaling_range_in=self.scaling_range_in,
            scaling_range_out=self.scaling_range_out,
            mode=self.mode,
            device=self.device,
        )
        self.drop_masks = None

    def _make_layer(
        self,
        block,
        planes,
        num_blocks,
        stride,
        device: Device = torch.device("cuda"),
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    # unique parameters, noise
                    noise_flag=self.noise_flag,
                    noise_level=self.noise_level,
                    output_noise_level=self.output_noise_level,
                    # quantization
                    quant_flag=self.quant_flag,
                    N_bits=self.N_bits,
                    N_bits_a=self.N_bits_a,
                    scaling_range_in=self.scaling_range_in,
                    scaling_range_out=self.scaling_range_out,
                    qmode=self.qmode,
                    # model
                    mode=self.mode,
                    device=self.device,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if x.size(-1) > 64:  # 224 x 224, e.g., cars, dogs, imagenet
            out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)

        out = self.layer3(out)
        out = self.layer4(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def SparseBP_GEMM_ResNet18(*args, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], 64, *args, **kwargs)


def SparseBP_GEMM_ResNet20(*args, **kwargs):
    return ResNet(BasicBlock, [3, 3, 3, 0], 16, *args, **kwargs)


def SparseBP_GEMM_ResNet32(*args, **kwargs):
    return ResNet(BasicBlock, [5, 5, 5, 0], 16, *args, **kwargs)


def SparseBP_GEMM_ResNet34(*args, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], 64, *args, **kwargs)


def SparseBP_GEMM_ResNet50(*args, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], 64, *args, **kwargs)


def SparseBP_GEMM_ResNet101(*args, **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], 64, *args, **kwargs)


def SparseBP_GEMM_ResNet152(*args, **kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], 64, *args, **kwargs)


def test():
    device = torch.device("cuda")
    net = SparseBP_GEMM_ResNet18(
        in_channel=3,
        n_class=10,
        act_thres=6,
        # unique paramaters
        img_height=32,
        img_width=32,
        device=device,
    ).to(device)

    x = torch.randn(2, 3, 32, 32).to(device)
    print(net)
    y = net(x)
    print(y.shape)


if __name__ == "__main__":
    test()
