import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = "Experiment/log/cifar10/vgg8/attacker/performance"
script = "scan_attacker.py"
config_file = "config/cifar10/vgg8/attacker/attacker.yml"
configs.load(config_file, recursive=True)

def task_launcher(args):
    pres = [
        "python3", 
        script, 
        config_file
    ]
    (N_bit, attack_mode, checkpoint) = args
    
    with open(os.path.join(root, f"attacker_Nbits_{N_bit}_attack_mode_{attack_mode}.log"), "w") as wfid:
        exp = [
            f"--quantize.N_bits={N_bit}",
            f"--run.attack_mode={attack_mode}",
            f"--checkpoint.restore_checkpoint={checkpoint}"
        ]
        logger.info(f"running command {pres + exp}")
        # logger.info(f"N bits: {N_bit}")
        # logger.info(f"Checkpoint is : {checkpoint}")
        # logger.info(f"Attacking mode is : {attack_mode}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        (
            8,  # Quantization bits for model
            "grad"
            "checkpoint/cifar100/resnet18/pretrain/SparseBP_GEMM_ResNet18_N-8_Na-8__acc-60.61_epoch-191.pt"
        )
    ]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")