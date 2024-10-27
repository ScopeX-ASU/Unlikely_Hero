"""
Date: 2023-11-15 22:09:29
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2023-11-15 22:09:29
FilePath: /L2ight_Robust/script/cifar10/vgg8/Sparsity/vgg_sparsity.py
"""
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = "Experiment/log/cifar100/resnet18/comparison/pruning"
script = "scan_pruner_comparison.py"
config_file = "config/cifar100/resnet18/recovery/recovery.yml"

configs.load(config_file, recursive=True)

def task_launcher(args):
    pres = ["python3", script, config_file]
    (N_bits, G_size) = args
    with open(os.path.join(root, f"Pruning_Nbit_{N_bits}_Gsize_{G_size}.log"), "w") as wfid:
        exp = [
        ]
        logger.info(f"running command {pres + exp}")
        logger.info(f"N bit is {N_bits}")

        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        (
            8,
            16  # Group size of cluster
        )
    ]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
