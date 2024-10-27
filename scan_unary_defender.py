import argparse
import os
import pickle
from copy import deepcopy

import torch
from pyutils.config import configs
from pyutils.general import ensure_dir
from pyutils.torch_train import load_model, set_torch_deterministic
from tqdm import tqdm

from core.builder import (
    make_criterion,
    make_dataloader,
    make_model,
    make_defender_small_loader,
)
from core.models.attack_defense.unary_defender import unary_defender
from core.models.layers.gemm_conv2d import GemmConv2d
from core.models.layers.gemm_linear import GemmLinear


def reset_model(model):
    load_model(
        model,
        configs.checkpoint.restore_checkpoint,
        ignore_size_mismatch=int(configs.checkpoint.no_linear),
    )


def gen_protected_index(
    model,
    validation_loader,
    small_loader,
    criterion,
    HD_con: int,
    salience: str,
    rt_ov: int = 0,
    mem_ov: float = 0.0,
    w_per: float = 0.0,
):
    defender_ins = unary_defender(
        model=model,
        mem_ov=mem_ov,
        w_percent=w_per,
        HD_con=HD_con,
        rt_ov=rt_ov,
        criterion=criterion,
        device=device,
        temperature=0.001,
    )
    # assert prot_idx is dict, f"TypeError, the return prot_idx should be a dict, but got {type(prot_idx)}"
    prot_idx = defender_ins.weight_protection(
        val_loader=validation_loader,
        small_loader=small_loader,
        method="importance",
        salience=salience,
    )
    # lg.info(f"Protected index is {prot_idx}")
    return prot_idx


def test_defender(
    model,
    validation_loader,
    criterion,
    HD_con: int,
    rt_ov: int = 0,
    mem_ov: float = 0,
    w_per: float = 0.0,
):
    defender_ins = unary_defender(
        model=model,
        mem_ov=mem_ov,
        w_percent=w_per,
        rt_ov=rt_ov,
        HD_con=HD_con,
        criterion=criterion,
        device=device,
    )
    prot_idx = defender_ins.weight_protection(
        val_loader=validation_loader, method="even", salience="second-order"
    )
    return prot_idx


def scan_defender():
    for w_per in tqdm([0.02, 0.075, 0.08, 0.09]):
        model_copy = deepcopy(model)
        h = 100

        set_torch_deterministic(configs.noise.random_state + (4 + w_per) * 10000)
        prot_idx = test_defender(
            model=model_copy,
            validation_loader=validation_loader,
            criterion=criterion,
            HD_con=h,
            w_per=w_per,
            rt_ov=10,
            mem_ov=0.0,
        )

        folder = f"./log/defender/{configs.model.name}/Weight_Percentage/Even_sampling"
        ensure_dir(folder)
        f_save = open(
            os.path.join(
                folder, f"{configs.quantize.N_bits}_bit_NoO_grad_Wper_{w_per}.pkl"
            ),
            "wb",
        )
        pickle.dump(prot_idx, f_save)


def scan_IS_defender(salience):
    for w_per in tqdm([0.0002, 0.0005, 0.001, 0.002, 0.01, 0.02, 0.04]):
        model_copy = deepcopy(model)
        h = 100

        set_torch_deterministic(configs.noise.random_state + (4 + w_per) * 10000)
        prot_idx = gen_protected_index(
            model=model_copy,
            validation_loader=validation_loader,
            small_loader=small_loader,
            criterion=criterion,
            w_per=w_per,
            HD_con=h,
            rt_ov=10,
            salience=salience,
        )

        folder = f"./EXP_data/defender/{configs.model.name}/new_sampling"
        ensure_dir(folder)
        f_save = open(
            os.path.join(
                folder, f"{configs.quantize.N_bits}_bit_NoO_grad_Wper_{w_per}.pkl"
            ),
            "wb",
        )
        pickle.dump(prot_idx, f_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive=True)
    device = torch.device("cuda")

    # set_torch_deterministic(configs.noise.random_state)
    _, validation_loader = make_dataloader()
    small_loader = make_defender_small_loader()

    criterion = make_criterion().to(device)

    model = make_model(device=device)
    reset_model(model)

    for name, module in model.named_modules():
        if isinstance(module, (GemmConv2d, GemmLinear)):
            module.weight_quantizer.to_two_com()

    scan_defender()
    scan_IS_defender(salience=configs.defense.salience)
