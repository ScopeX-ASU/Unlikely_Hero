'''
Date: 2024-10-01 13:49:57
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-01 13:52:33
FilePath: /ONN_Reliable/scan_locker.py
'''
import argparse
import os
import pickle
from copy import deepcopy

import torch
from pyutils.config import configs
from pyutils.general import ensure_dir
from pyutils.torch_train import load_model, set_torch_deterministic

from core.builder import make_criterion, make_dataloader, make_model
from core.models.attack_defense.post_locker import smart_locker
from core.models.layers.gemm_conv2d import GemmConv2d
from core.models.layers.gemm_linear import GemmLinear


def reset_model(model):
    load_model(
        model,
        configs.checkpoint.restore_checkpoint,
        ignore_size_mismatch=int(configs.checkpoint.no_linear),
    )


def generate_statistics(model, criterion, device, validation_loader, eta: float):
    locker = smart_locker(
        model=model, criterion=criterion, cluster_method="normal", device=device
    )
    L_K, W_K, G_size = locker.smart_locking(
        eta=eta,
        val_loader=validation_loader,
    )

    model.calculate_signature(G_size=G_size)
    locker.calculate_mem_ov()

    return L_K, W_K, G_size


def scan_locker(model, validation_loader, criterion, eta: float = 0.0):
    model_copy = deepcopy(model)

    set_torch_deterministic(configs.noise.random_state + eta * 100)

    L_K, W_K, G_size = generate_statistics(
        model=model_copy,
        validation_loader=validation_loader,
        criterion=criterion,
        eta=eta,
        device=device,
    )
    folder = (
        f"./EXP_data/Locker/{configs.model.name}/sens-aware"  # + Customized Locking
    )
    ensure_dir(folder)
    f_save = open(
        os.path.join(folder, f"{configs.quantize.N_bits}_bit_NoO_grad_LK_{eta}.pkl"),
        "wb",
    )
    pickle.dump(L_K, f_save)
    f_save = open(
        os.path.join(folder, f"{configs.quantize.N_bits}_bit_NoO_grad_WK_{eta}.pkl"),
        "wb",
    )
    pickle.dump(W_K, f_save)
    f_save = open(
        os.path.join(folder, f"{configs.quantize.N_bits}_bit_NoO_grad_G_{eta}.pkl"),
        "wb",
    )
    pickle.dump(G_size, f_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive=True)
    device = torch.device("cuda")

    _, validation_loader = make_dataloader()
    criterion = make_criterion().to(device)

    model = make_model(device=device)
    reset_model(model)

    for name, module in model.named_modules():
        if isinstance(module, (GemmConv2d, GemmLinear)):
            module.weight_quantizer.to_two_com()

    scan_locker(
        model=model,
        validation_loader=validation_loader,
        eta=configs.defense.eta,
        criterion=criterion,
    )
