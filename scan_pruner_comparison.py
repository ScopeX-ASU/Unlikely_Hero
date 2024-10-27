import argparse
import os
from copy import deepcopy

import numpy as np
import torch
from pyutils.config import configs
from pyutils.general import ensure_dir
from pyutils.general import logger as lg
from pyutils.torch_train import load_model, set_torch_deterministic
from tqdm import tqdm

from core.builder import (
    make_attacker_loader,
    make_criterion,
    make_dataloader,
    make_model,
)
from core.models.attack_defense.attacker import grad_attacker
from core.models.attack_defense.post_pruner import post_pruner
from core.models.layers.gemm_conv2d import GemmConv2d
from core.models.layers.gemm_linear import GemmLinear
from train_pretrain import validate


def reset_model(model):
    load_model(
        model,
        configs.checkpoint.restore_checkpoint,
        ignore_size_mismatch=int(configs.checkpoint.no_linear),
    )


def test_grad_attacker(
    Nit: int,
    model,
    attacker_loader,
    validation_loader,
    criterion,
    inv_ov: int,
    HD_con: int,
    random_int: int,
    G: int,
):
    attacker = grad_attacker(
        model=model,
        criterion=criterion,
        N_sample=Nit,
        inf_ov=inv_ov,
        HD_con=HD_con,
        protected_index={},
        random_int=random_int,
        device=device,
    )

    attacker.pbs_top(attacker_loader=attacker_loader)

    res = validate(
        model=model,
        validation_loader=validation_loader,
        epoch=-3,
        criterion=criterion,
        accuracy_vector=[],
        loss_vector=[],
        device=device,
    )
    lg.info(f"Accuracy after attack is {res}")

    corrector = post_pruner(dirty_model=model, device=device)

    corrector.perform_correction(G_size=G)

    res_recover = validate(
        model=model,
        validation_loader=validation_loader,
        epoch=-3,
        criterion=criterion,
        accuracy_vector=[],
        loss_vector=[],
        device=device,
    )

    lg.info(f"Accuracy after Recovery is {res_recover}")

    return res_recover


def scan_grad_attacker(attacker_loader, G_size):
    final_mean_list, final_std_list = [], []
    for i in tqdm(
        [60, 100, 150, 200, 280, 320, 260, 420, 480, 540, 1060, 1600, 2120, 2400]
    ):
        for h in [100]:
            res_list = []
            for s in range(5):  # 5 small datasets
                set_torch_deterministic(configs.noise.random_state + (i + h) * s)
                model_copy = deepcopy(model)

                res_recover = test_grad_attacker(
                    Nit=1,
                    inv_ov=i,
                    HD_con=h,
                    model=model_copy,
                    attacker_loader=attacker_loader,
                    validation_loader=validation_loader,
                    criterion=criterion,
                    random_int=s,
                    G=G_size,
                )
                res_list.append(res_recover)

            mean = np.mean(res_list)
            std = np.std(res_list)

            final_mean_list.append(round(mean, 3))
            final_std_list.append(round(std, 3))

    # Output the results to the csv files
    folder = f"./EXP_data/comparison/{configs.model.name}/pruning"
    ensure_dir(folder)
    np.savetxt(
        os.path.join(
            folder, f"{configs.quantize.N_bits}_bit_grad_mean_pruning_G_{G_size}.csv"
        ),
        np.array(final_mean_list),
        delimiter=",",
    )
    np.savetxt(
        os.path.join(
            folder, f"{configs.quantize.N_bits}_bit_grad_std_pruning_{G_size}.csv"
        ),
        np.array(final_std_list),
        delimiter=",",
    )


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
    attacker_loader = make_attacker_loader()

    for name, module in model.named_modules():
        if isinstance(module, (GemmConv2d, GemmLinear)):
            module.weight_quantizer.to_two_com()

    for G_size in [16]:  # pruning group size
        model.calculate_signature(G_size=G_size)
        scan_grad_attacker(attacker_loader=attacker_loader, G_size=G_size)
