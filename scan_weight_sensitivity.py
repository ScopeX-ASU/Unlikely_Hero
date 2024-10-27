import argparse

import torch
from pyutils.config import configs
from pyutils.general import logger as lg
from pyutils.torch_train import load_model

from core.builder import (
    make_attacker_loader,
    make_criterion,
    make_dataloader,
    make_model,
)
from core.models.layers.gemm_conv2d import GemmConv2d
from core.models.layers.gemm_linear import GemmLinear
from core.models.layers.utils import calculate_grad_hessian


def reset_model(model):
    load_model(
        model,
        configs.checkpoint.restore_checkpoint,
        ignore_size_mismatch=int(configs.checkpoint.no_linear),
    )


def calculate_taylor_series(model, N_bits: int):
    for layer in model.modules():
        if isinstance(layer, (GemmConv2d, GemmLinear)):
            series_term = (
                layer.weight._first_grad.data
                * (layer.weight_quantizer.w_q_com.data - (2**N_bits - 1)).sign()
                + layer.weight._second_grad.data * (2**N_bits - 1) / 2
            )
            layer.weight._taylor_series = series_term


if __name__ == "__main__":
    N_quart = 10

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive=True)
    device = torch.device("cuda")

    _, validation_loader = make_dataloader()
    criterion = make_criterion().to(device)
    attacker_loader = make_attacker_loader()

    model = make_model(device=device)

    reset_model(model)

    for name, module in model.named_modules():
        if isinstance(module, (GemmConv2d, GemmLinear)):
            module.weight_quantizer.to_two_com()

    calculate_grad_hessian(
        model,
        train_loader=validation_loader,
        criterion=criterion,
        mode="defender",
        num_samples=1,
        device=device,
    )
    calculate_taylor_series(model=model, N_bits=configs.quantize.N_bits)

    sensitivity_stat = {}
    sensitivity_global = []
    sensitivity_bin = []
    sensitivity_items = []

    for name, layer in model.named_modules():
        if isinstance(layer, (GemmConv2d, GemmLinear)):
            sensitivity_global.extend(layer.weight._taylor_series.data.view(-1))

    sensitivity_global = torch.tensor(sensitivity_global)

    range_S = sensitivity_global.max() - sensitivity_global.min()
    for i in range(N_quart):
        quartile = torch.quantile(sensitivity_global, i / N_quart)

    sensitivity_bin.append(quartile.item())

    # lg.info(f"Length of sensitivity_global is {sensitivity_global.shape}")
    # lg.info(f"Sensitivity bins are {sensitivity_bin}")

    for i in range(N_quart):
        count = (
            (
                (sensitivity_global >= sensitivity_bin[i])
                & (sensitivity_global <= sensitivity_bin[i + 1])
            )
            .sum()
            .item()
        )
        sensitivity_items.append(count)

    lg.info(f"Sensitivity statistics are {sensitivity_items}")
