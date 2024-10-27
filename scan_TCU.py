'''
Date: 2024-10-01 13:49:57
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-01 13:51:35
FilePath: /ONN_Reliable/scan_TCU.py
'''
import argparse
import os
import pickle

import torch
from pyutils.config import configs
from pyutils.general import logger as lg
from pyutils.torch_train import load_model
from tqdm import tqdm

from core.builder import make_criterion, make_dataloader, make_model
from core.models.attack_defense.unary_defender import unary_defender
from core.models.layers.gemm_conv2d import GemmConv2d
from core.models.layers.gemm_linear import GemmLinear


def reset_model(model):
    load_model(
        model,
        configs.checkpoint.restore_checkpoint,
        ignore_size_mismatch=int(configs.checkpoint.no_linear),
    )


def calculate_mem_ov(prot_idx: dict, w_per: float = 0.0):
    defender_ins = unary_defender(
        model=model,
        mem_ov=0.0,
        w_percent=w_per,
        HD_con=100,
        rt_ov=10,
        criterion=criterion,
        device=device,
    )
    mem_ov = defender_ins.cal_mem_ov(ptct_idx=prot_idx, mode="truncated")
    lg.info(f"memory overhead for {w_per} is {mem_ov}")


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

    W_per = tqdm([0.002])
    for w_per in W_per:
        file_Path = os.path.join(
            f"./EXP_data/defender/{configs.model.name}/new_sampling",
            f"{configs.quantize.N_bits}_bit_NoO_grad_Wper_{w_per}.pkl",
        )

        with open(file_Path, "rb") as fo:
            prot_idx = pickle.load(fo, encoding="bytes")
            fo.close()

        calculate_mem_ov(prot_idx=prot_idx, w_per=w_per)
