import argparse
import os
from copy import deepcopy

import numpy as np
import torch
from pyutils.config import configs
from pyutils.general import ensure_dir
from pyutils.general import logger as lg
from pyutils.torch_train import load_model, set_torch_deterministic

from core.builder import (
    make_attacker_loader,
    make_criterion,
    make_dataloader,
    make_model,
)
from core.models.attack_defense.attacker import grad_attacker, grad_attacker_LSB
from core.models.layers.gemm_conv2d import GemmConv2d
from core.models.layers.gemm_linear import GemmLinear
from core.models.layers.utils import calculate_grad_hessian
from train_pretrain import validate


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


def perform_one_attack(model_copy, criterion, inf_ov, HD_con, random_int: int):
    attacker = grad_attacker(
        model=model_copy,
        criterion=criterion,
        N_sample=1,
        inf_ov=inf_ov,
        HD_con=HD_con,
        protected_index={},
        random_int=random_int,
        device=device,
    )
    attacker.pbs_top(attacker_loader=attacker_loader)

    res = validate(
        model=model_copy,
        validation_loader=validation_loader,
        epoch=-3,
        criterion=criterion,
        accuracy_vector=[],
        loss_vector=[],
        device=device,
    )

    # lg.info(f"Accuracy is {res}")
    return model_copy


def scan_grad_attacker(model_copy, criterion):
    i, h, s = 10, 1, 1

    set_torch_deterministic(configs.noise.random_state + (i + h) * s)

    model_attack = perform_one_attack(
        model_copy=model_copy, criterion=criterion, inf_ov=i, HD_con=h, random_int=s
    )

    calculate_grad_hessian(
        model_attack,
        train_loader=validation_loader,
        criterion=criterion,
        mode="defender",
        num_samples=1,
        device=device,
    )
    calculate_taylor_series(model=model_attack, N_bits=configs.quantize.N_bits)

    sensitivity_stat = {}

    for name, layer in model_attack.named_modules():
        if isinstance(layer, (GemmConv2d, GemmLinear)):
            # lg.info(f"For layer: {name}")
            sensitivity = []

            for i in range(20):
                quartile = torch.quantile(
                    layer.weight._taylor_series.data.view(-1), i / 20
                )
                sensitivity.append(quartile.item())

            sensitivity_stat[name] = sensitivity_stat.get(name, 0) + torch.tensor(
                sensitivity
            )
            # lg.info(f"Average for layer {name} after attack is {layer.weight._taylor_series.data.median()}")

            folder = f"./EXP_data/layer_sensitivity/{configs.model.name}"
            ensure_dir(folder)
            np.savetxt(
                os.path.join(
                    folder,
                    f"Layer_{name}_{configs.quantize.N_bits}_bit_after_attack.csv",
                ),
                np.array(sensitivity),
                delimiter=",",
            )

    lg.info(f"Statistics are {sensitivity_stat}")


def perform_one_attack_protect(model_copy, criterion, inf_ov, HD_con, random_int: int):
    attacker = grad_attacker_LSB(
        model=model_copy,
        criterion=criterion,
        N_sample=1,
        inf_ov=inf_ov,
        HD_con=HD_con,
        protected_index={},
        random_int=random_int,
        device=device,
    )

    attacker.pbs_top(attacker_loader=attacker_loader)

    res = validate(
        model=model_copy,
        validation_loader=validation_loader,
        epoch=-3,
        criterion=criterion,
        accuracy_vector=[],
        loss_vector=[],
        device=device,
    )

    # lg.info(f"Accuracy is {res}")
    return model_copy


def scan_grad_attacker_protected(model_copy, criterion):
    i, h, s = 10, 1, 1

    set_torch_deterministic(configs.noise.random_state + (i + h) * s)

    model_ptct = perform_one_attack_protect(
        model_copy=model_copy, criterion=criterion, inf_ov=i, HD_con=h, random_int=s
    )

    calculate_grad_hessian(
        model_ptct,
        train_loader=validation_loader,
        criterion=criterion,
        mode="defender",
        num_samples=1,
        device=device,
    )
    calculate_taylor_series(model=model_ptct, N_bits=configs.quantize.N_bits)

    sensitivity_stat_ptct = {}
    for name, layer in model_ptct.named_modules():
        if isinstance(layer, (GemmConv2d, GemmLinear)):
            # lg.info(f"For layer: {name}")
            sensitivity = []

            for i in range(20):
                quartile = torch.quantile(
                    layer.weight._taylor_series.data.view(-1), i / 20
                )
                sensitivity.append(quartile.item())

            sensitivity_stat_ptct[name] = sensitivity_stat_ptct.get(
                name, 0
            ) + torch.tensor(sensitivity)
            lg.info(
                f"Average for layer {name} after protection is {layer.weight._taylor_series.data.median()}"
            )

            folder = f"./EXP_data/layer_sensitivity/{configs.model.name}"
            ensure_dir(folder)
            np.savetxt(
                os.path.join(
                    folder,
                    f"Layer_{name}_{configs.quantize.N_bits}_bit_after_protection.csv",
                ),
                np.array(sensitivity),
                delimiter=",",
            )

    lg.info(f"Statistics are {sensitivity_stat_ptct}")


if __name__ == "__main__":
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

    for name, layer in model.named_modules():
        if isinstance(layer, (GemmConv2d, GemmLinear)):
            lg.info(f"For layer: {name}")
            sensitivity = []

            for i in range(20):
                quartile = torch.quantile(
                    layer.weight._taylor_series.data.view(-1), i / 20
                )
                sensitivity.append(quartile.item())

            sensitivity_stat[name] = sensitivity_stat.get(name, 0) + torch.tensor(
                sensitivity
            )
            lg.info(
                f"Average for layer {name} is {layer.weight._taylor_series.data.median()}"
            )

            folder = f"./EXP_data/layer_sensitivity/{configs.model.name}"
            ensure_dir(folder)
            np.savetxt(
                os.path.join(folder, f"Layer_{name}_{configs.quantize.N_bits}_bit.csv"),
                np.array(sensitivity),
                delimiter=",",
            )

    lg.info(f"Statistics are {sensitivity_stat}")

    model_copy = deepcopy(model)
    scan_grad_attacker(model_copy=model_copy, criterion=criterion)

    reset_model(model)
    model_copy = deepcopy(model)
    scan_grad_attacker_protected(model_copy=model_copy, criterion=criterion)
