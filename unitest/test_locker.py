'''
Date: 2024-10-01 13:49:57
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-01 13:53:49
FilePath: /ONN_Reliable/unitest/test_locker.py
'''
import argparse
import os
import pickle

import torch
from pyutils.config import configs
from pyutils.general import ensure_dir
from pyutils.torch_train import load_model, set_torch_deterministic

from core.builder import make_criterion, make_dataloader, make_model
from core.models.attack_defense.post_locker import smart_locker

# from core.models.sparse_bp_vgg import SparseBP_GEMM_VGG8
from core.models.layers.gemm_conv2d import GemmConv2d
from core.models.layers.gemm_linear import GemmLinear


def reset_model(model):
    load_model(
        model,
        configs.checkpoint.restore_checkpoint,
        ignore_size_mismatch=int(configs.checkpoint.no_linear),
    )


def test_locker(model, validation_loader, eta: float, criterion, w_per: float = 0):
    set_torch_deterministic(configs.noise.random_state)
    locker = smart_locker(
        model=model, w_percent=w_per, criterion=criterion, device=device
    )
    I_K_res, labels_res, W_K_res = locker.smart_locking(
        val_loader=validation_loader, mode="err-first", eta=eta, N_trial=1
    )
    locker.calculate_mem_ov(I_K_res=I_K_res)

    folder = f"./log/locker/{configs.model.name}"
    ensure_dir(folder)
    f_save = open(
        os.path.join(folder, f"{configs.quantize.N_bits}_bit_NoO_grad_IK.pkl"), "wb"
    )
    pickle.dump(I_K_res, f_save)
    f_save = open(
        os.path.join(folder, f"{configs.quantize.N_bits}_bit_NoO_grad_Labels.pkl"), "wb"
    )
    pickle.dump(labels_res, f_save)
    f_save = open(
        os.path.join(folder, f"{configs.quantize.N_bits}_bit_NoO_grad_WK.pkl"), "wb"
    )
    pickle.dump(W_K_res, f_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive=True)
    device = torch.device("cuda")

    # set_torch_deterministic(configs.noise.random_state)
    _, validation_loader = make_dataloader()
    criterion = make_criterion().to(device)

    model = make_model(device=device)
    reset_model(model)

    for name, module in model.named_modules():
        if isinstance(module, (GemmConv2d, GemmLinear)):
            module.weight_quantizer.to_two_com()

    test_locker(
        model=model,
        validation_loader=validation_loader,
        eta=1,
        criterion=criterion,
        w_per=0.5,
    )
