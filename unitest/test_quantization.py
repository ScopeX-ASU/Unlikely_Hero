'''
Date: 2024-03-22 16:05:08
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-01 13:55:03
FilePath: /ONN_Reliable/unitest/test_quantization.py
'''
import torch

from core.models.layers.gemm_conv2d import GemmConv2d


def main():
    device = torch.device("cuda")
    test_tensor = torch.rand(size=[4, 1, 8, 8]).to(device)

    # input_quantizer = input_quantizer_fn(True, 8)
    # output_quantizer = output_quantizer_fn(True, 8)
    # test_tensor_q = input_quantizer(test_tensor)
    conv_1 = GemmConv2d(
        in_channels=1,
        out_channels=2,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        quant_flag=False,
        device=device,
    )
    res = conv_1(test_tensor)

    conv_2 = GemmConv2d(
        in_channels=1,
        out_channels=2,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        quant_flag=True,
        device=device,
    )
    conv_2.weight.data.copy_(conv_1.weight.data)

    res_q = conv_2(test_tensor)

    # Simple mean() as the loss function
    loss = res.sum() / test_tensor.numel()
    loss_q = res_q.sum() / test_tensor.numel()

    print(loss.item(), loss_q.item())
    print(f"difference={(res_q-res).norm(p=2) / res.norm(p=2)}")

    loss.backward()
    loss_q.backward()

    print(conv_1.weight.grad)
    print(conv_2.weight.grad)

    print(
        f"unquantized: {conv_1.weight.grad.mean()}, quantized: {conv_2.weight.grad.mean()}"
    )
    print()


if __name__ == "__main__":
    main()
