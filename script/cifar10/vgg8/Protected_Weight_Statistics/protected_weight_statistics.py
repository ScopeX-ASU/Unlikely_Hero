import torch
import numpy as np
import argparse
from core.models.layers.gemm_conv2d import GemmConv2d
from core.models.layers.gemm_linear import GemmLinear

from pyutils.torch_train import load_model
from pyutils.config import configs
from pyutils.general import logger as lg, ensure_dir
from core.builder import make_model, make_dataloader, make_criterion, make_attacker_loader

from core.models.attack.attacker import grad_attacker
from core.models.attack.defender import defender

from train_pretrain import validate
from pyutils.torch_train import set_torch_deterministic
from copy import deepcopy
import os
from tqdm import tqdm
import pickle

def reset_model(model):
    load_model(
        model,
        configs.checkpoint.restore_checkpoint,
        ignore_size_mismatch=int(configs.checkpoint.no_linear),
    )

def calculate_statistics(prot_idx: dict):
    defender_ins = defender(
        model=model, mem_ov=0., w_percent=w_per, 
        HD_con = 100, rt_ov=10, criterion=criterion, device=device
    )
    defender_ins.cal_statistics(prot_idx)

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

    W_per = tqdm([0.001, 0.002, 0.005, 0.01, 0.02])
    for w_per in W_per:
        file_Path = os.path.join(f"./EXP_data/defender/{configs.model.name}/{configs.defense.salience}", f"{configs.quantize.N_bits}_bit_NoO_grad_Wper_{w_per}.pkl")
            
        with open(file_Path, 'rb') as fo:
            prot_idx = pickle.load(fo, encoding='bytes')
            fo.close()

        calculate_statistics(prot_idx=prot_idx)