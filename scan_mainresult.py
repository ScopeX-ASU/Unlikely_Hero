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
from core.models.attack_defense.post_recovery import post_corrector
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
    protected_index: dict,
    random_int: int,
    L_K,
    W_K,
    G,
):
    attacker = grad_attacker(
        model=model,
        criterion=criterion,
        N_sample=Nit,
        inf_ov=inv_ov,
        HD_con=HD_con,
        protected_index=protected_index,
        random_int=random_int,
        device=device,
    )

    attacker.pbs_top(attacker_loader=attacker_loader)
    # Attack under Unary protection
    res_unary = validate(
        model=model,
        validation_loader=validation_loader,
        epoch=-3,
        criterion=criterion,
        accuracy_vector=[],
        loss_vector=[],
        device=device,
    )
    # lg.info(f"Accuracy after attack [TCU provided] is {res_unary}")

    corrector = post_corrector(dirty_model=model, device=device)

    corrector.perform_correction(L_K=L_K, W_K=W_K, G_size=G)
    # Attack under Unary protection + Weight Locking
    res_recover = validate(
        model=model,
        validation_loader=validation_loader,
        epoch=-3,
        criterion=criterion,
        accuracy_vector=[],
        loss_vector=[],
        device=device,
    )
    # lg.info(f"Accuracy after Recovery is {res_recover}")

    return res_unary, res_recover


def scan_grad_attacker(
    prot_idx: dict, attacker_loader, L_K, W_K, G_size, eta, w_per: float = 0.0
):
    final_mean_list, final_std_list = [], []
    final_mean_list_unary, final_std_list_unary = [], []
    for i in tqdm([60, 200, 420, 1060, 1600, 2120, 2400]):  # Inference overhead
        for h in [100]:  # Hamming Disatnce constraint
            res_list = []  # Unary + Locking
            res_unary_list = []
            for s in range(5):
                set_torch_deterministic(configs.noise.random_state + (i + h) * s)
                model_copy = deepcopy(model)

                res_unary, res_recover = test_grad_attacker(
                    Nit=1,
                    inv_ov=i,
                    HD_con=h,
                    model=model_copy,
                    attacker_loader=attacker_loader,
                    validation_loader=validation_loader,
                    criterion=criterion,
                    protected_index=prot_idx,
                    random_int=s,
                    L_K=L_K,
                    W_K=W_K,
                    G=G_size,
                )
                res_unary_list.append(res_unary)
                res_list.append(res_recover)

            mean = np.mean(res_list)
            std = np.std(res_list)
            mean_unary = np.mean(res_unary_list)
            std_unary = np.std(res_unary_list)

            final_mean_list.append(round(mean, 3))
            final_std_list.append(round(std, 3))
            final_mean_list_unary.append(round(mean_unary, 3))
            final_std_list_unary.append(round(std_unary, 3))

    # Output the results to the csv files
    folder = f"./EXP_data/mainresult/{configs.model.name}/sens-aware"
    ensure_dir(folder)
    np.savetxt(
        os.path.join(
            folder,
            f"{configs.quantize.N_bits}_bit_grad_mean_wper_{w_per}_eta_{eta}_UL.csv",
        ),
        np.array(final_mean_list),
        delimiter=",",
    )
    np.savetxt(
        os.path.join(
            folder,
            f"{configs.quantize.N_bits}_bit_grad_std_wper_{w_per}_eta_{eta}_UL.csv",
        ),
        np.array(final_std_list),
        delimiter=",",
    )
    np.savetxt(
        os.path.join(
            folder,
            f"{configs.quantize.N_bits}_bit_grad_mean_wper_{w_per}_eta_{eta}_OnlyUnary.csv",
        ),
        np.array(final_mean_list_unary),
        delimiter=",",
    )
    np.savetxt(
        os.path.join(
            folder,
            f"{configs.quantize.N_bits}_bit_grad_std_wper_{w_per}_eta_{eta}_OnlyUnary.csv",
        ),
        np.array(final_std_list_unary),
        delimiter=",",
    )

    return model


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
    attacker_loader = make_attacker_loader()

    for name, module in model.named_modules():
        if isinstance(module, (GemmConv2d, GemmLinear)):
            module.weight_quantizer.to_two_com()
    # Read the weight index being protected by Unary
    file_Path = os.path.join(
        f"./EXP_data/defender/{configs.model.name}/new_sampling",
        f"{configs.quantize.N_bits}_bit_NoO_grad_Wper_{configs.defense.W_per}.pkl",
    )
    with open(file_Path, "rb") as fo:
        prot_idx = pickle.load(fo, encoding="bytes")
        fo.close()
    # Read the locking centers to be locked
    file_Path = os.path.join(
        f"./EXP_data/Locker/{configs.model.name}",
        f"{configs.quantize.N_bits}_bit_NoO_grad_LK_{configs.defense.eta}.pkl",
    )
    with open(file_Path, "rb") as fo:
        L_K = pickle.load(fo, encoding="bytes")
        fo.close()
    # Read the weight index to be locked
    file_Path = os.path.join(
        f"./EXP_data/Locker/{configs.model.name}",
        f"{configs.quantize.N_bits}_bit_NoO_grad_WK_{configs.defense.eta}.pkl",
    )
    with open(file_Path, "rb") as fo:
        W_K = pickle.load(fo, encoding="bytes")
        fo.close()
    # Read the group size
    file_Path = os.path.join(
        f"./EXP_data/Locker/{configs.model.name}",
        f"{configs.quantize.N_bits}_bit_NoO_grad_G_{configs.defense.eta}.pkl",
    )
    with open(file_Path, "rb") as fo:
        G_size = pickle.load(fo, encoding="bytes")
        fo.close()

    model.calculate_signature(G_size=G_size)
    # Attack the model under full protection
    scan_grad_attacker(
        attacker_loader=attacker_loader,
        prot_idx=prot_idx,
        w_per=configs.defense.W_per,
        L_K=L_K,
        W_K=W_K,
        G_size=G_size,
        eta=configs.defense.eta,
    )
