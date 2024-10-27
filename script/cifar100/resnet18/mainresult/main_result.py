"""
Date: 2023-11-15 22:09:29
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2023-11-15 22:09:29
FilePath: /L2ight_Robust/script/cifar100/resnet18/Sparsity/vgg_sparsity.py
"""
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = "Experiment/log/cifar100/resnet18/mainresult"
script = "scan_mainresult.py"
config_file = "config/cifar100/resnet18/recovery/recovery.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ["python3", script, config_file]
    (N_bits, W_per, eta, attacking_mode) = args

    with open(os.path.join(root, f"mainresult_Nbit_{N_bits}_mode_{attacking_mode}_W_per_{W_per}_eta_{eta}.log"), "w") as wfid:
        exp = [
            f"--quantize.N_bits={N_bits}",
            f"--defense.eta={eta}",
            f"--defense.W_per={W_per}",
            f"--run.attack_mode={attacking_mode}"
        ]

        logger.info(f"running command {pres + exp}")
        # logger.info(f"eta is  {eta}")
        # logger.info(f"W_per is {W_per}")
        # logger.info(f"Attacking mode is {attacking_mode}")

        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        (
            8,
            0.002,
            1,
            "grad"
        )
    ]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
