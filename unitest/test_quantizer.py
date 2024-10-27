'''
Date: 2024-03-21 12:39:39
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-01 13:55:09
FilePath: /ONN_Reliable/unitest/test_quantizer.py
'''
import torch

from core.models.quantize.quant_base import Qmodes
from core.models.quantize.quantizer import (
    input_quantizer_fn,
    output_quantizer_fn,
    weight_quantizer_fn,
)


def test_input_quantizer():
    test_tensor = torch.rand(size=[1, 3, 4, 4])
    input_quantizer = input_quantizer_fn(True, 8)

    test_input = input_quantizer(test_tensor)
    print(f"scale={input_quantizer.scale}, zero point={input_quantizer.zp}")
    print(
        f"loss after input quantization = {(test_tensor - test_input).norm(p=2) / test_input.norm(p=2)}"
    )
    print(
        f"quantized input / scale + zp = {test_input / input_quantizer.scale + input_quantizer.zp}"
    )


def test_weight_quantizer():
    test_tensor = torch.randn(size=[1, 3, 4, 4])

    weight_quantizer = weight_quantizer_fn(
        size=test_tensor.shape, quant_flag=True, qmode=Qmodes.layer_wise, N_bits=8
    )

    test_weight = weight_quantizer(test_tensor)

    print(f"scale={weight_quantizer.alpha}")
    print(f"loss after weight quantization = {(test_tensor - test_weight).norm(p=2)}")
    print(f"quantized weight / scale = {test_weight / weight_quantizer.alpha}")


def test_output_quantizer():
    test_tensor = torch.randn(size=[1, 3, 4, 4])

    output_quantizer = output_quantizer_fn(True, 8)

    test_output = output_quantizer(test_tensor)

    print(f"scale={output_quantizer.scale}, zero point={output_quantizer.zp}")
    print(
        f"loss after output quantization = {(test_tensor - test_output).norm(p=2) / test_output.norm(p=2)}"
    )
    print(
        f"quantized output / scale + zp = {test_output / output_quantizer.scale + output_quantizer.zp}"
    )


# test_input_quantizer()
# test_weight_quantizer()
test_output_quantizer()
