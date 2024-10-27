import argparse
import os
import pickle
from copy import deepcopy

import numpy as np
import torch
from pyutils.config import configs
from pyutils.general import ensure_dir
from pyutils.torch_train import load_model, set_torch_deterministic
from tqdm import tqdm

from core.builder import (
    make_attacker_loader,
    make_criterion,
    make_dataloader,
    make_model,
)
from core.models.attack_defense.attacker import grad_attacker
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
    protect_index: dict,
):
    attacker = grad_attacker(
        model=model,
        criterion=criterion,
        N_sample=Nit,
        inf_ov=inv_ov,
        HD_con=HD_con,
        device=device,
        protected_index=protect_index,
        random_int=random_int,
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
    return res


def scan_grad_attacker(
    model,
    attacker_loader,
    validation_loader,
    criterion,
    protect_index: dict,
    salience,
    w_per,
):
    final_mean_list, final_std_list = [], []
    for i in tqdm([20, 80, 160, 400, 900]):
        for h in tqdm([100]):
            res_list = []
            for s in range(5):
                model_copy = deepcopy(model)
                set_torch_deterministic(configs.noise.random_state + (i + h) * s)

                res = test_grad_attacker(
                    Nit=1,
                    inv_ov=i,
                    HD_con=h,
                    model=model_copy,
                    attacker_loader=attacker_loader,
                    validation_loader=validation_loader,
                    criterion=criterion,
                    random_int=s,
                    protect_index=protect_index,
                )

                res_list.append(res)

            mean = np.mean(res_list)
            std = np.std(res_list)
            final_mean_list.append(round(mean, 3))
            final_std_list.append(round(std, 3))

    # Output the results to the csv files
    folder = f"./EXP_data/salience_ranking/{configs.model.name}"
    ensure_dir(folder)
    np.savetxt(
        os.path.join(
            folder,
            f"{configs.quantize.N_bits}_bit_grad_mean_salience_{salience}_wper_{w_per}.csv",
        ),
        np.array(final_mean_list),
        delimiter=",",
    )
    np.savetxt(
        os.path.join(
            folder,
            f"{configs.quantize.N_bits}_bit_grad_std_salience_{salience}_wper_{w_per}.csv",
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

    # set_torch_deterministic(configs.noise.random_state)
    _, validation_loader = make_dataloader()
    criterion = make_criterion().to(device)

    attacker_loader = make_attacker_loader()

    model = make_model(device=device)
    reset_model(model)
    for name, module in model.named_modules():
        if isinstance(module, (GemmConv2d, GemmLinear)):
            module.weight_quantizer.to_two_com()

    for wper in tqdm(
        [0.0002, 0.001, 0.002, 0.01]
    ):  # Percentage of weights to be protected in Unary
        file_Path = os.path.join(
            f"./EXP_data/salience_ranking/{configs.model.name}/{configs.defense.salience}",
            f"{configs.quantize.N_bits}_bit_NoO_grad_Wper_{wper}.pkl",
        )
        with open(file_Path, "rb") as fo:
            prot_idx = pickle.load(fo, encoding="bytes")
            fo.close()

        scan_grad_attacker(
            model=model,
            validation_loader=validation_loader,
            attacker_loader=attacker_loader,
            criterion=criterion,
            protect_index=prot_idx,
            salience=configs.defense.salience,
            w_per=wper,
        )
