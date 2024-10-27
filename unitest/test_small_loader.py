'''
Date: 2024-10-01 13:49:46
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-01 13:55:15
FilePath: /ONN_Reliable/unitest/test_small_loader.py
'''
import argparse

import torch
from pyutils.config import configs
from pyutils.general import logger as lg

from core.builder import make_defender_small_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive=True)
    device = torch.device("cuda")

    validation_dataset, defender_small_loader = make_defender_small_loader()
    lg.info(f"{len(validation_dataset)}")

    for i, (data, target) in enumerate(defender_small_loader):
        lg.info(data.shape)
