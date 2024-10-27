'''
Date: 2024-10-01 13:49:57
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-01 13:55:21
FilePath: /ONN_Reliable/unitest/test_taylor_series.py
'''
import argparse

import torch
from pyutils.config import configs
from pyutils.torch_train import load_model

from core.builder import make_criterion, make_dataloader, make_model
from core.models.attack_defense.unary_defender import unary_defender
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


def test_taylor_series(
    model,
    validation_loader,
    criterion,
    HD_con: int,
    rt_ov: int = 0,
    mem_ov: float = 0.0,
    w_per: float = 0.0,
):
    res = validate(
        model=model,
        validation_loader=validation_loader,
        epoch=-3,
        criterion=criterion,
        accuracy_vector=[],
        loss_vector=[],
        device=device,
    )

    defender_ins = unary_defender(
        model=model,
        mem_ov=mem_ov,
        w_percent=w_per,
        HD_con=HD_con,
        rt_ov=rt_ov,
        criterion=criterion,
        device=device,
    )
    calculate_grad_hessian(
        model,
        train_loader=validation_loader,
        criterion=criterion,
        mode="defender",
        num_samples=1,
        device=device,
    )
    defender_ins.calculate_taylor_series()

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

    test_taylor_series(
        model=model,
        validation_loader=validation_loader,
        criterion=criterion,
        HD_con=100,
        rt_ov=10,
        mem_ov=0.0,
        w_per=0.01,
    )
