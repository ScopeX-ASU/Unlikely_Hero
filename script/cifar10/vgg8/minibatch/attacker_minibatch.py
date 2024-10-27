import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = "Experiment/log/cifar10/vgg8/attacker/minibatch"
script = "scan_minibatch.py"
config_file = "config/cifar10/vgg8/attacker/attacker.yml"
configs.load(config_file, recursive=True)

def task_launcher(args):
    pres = ["python3", script, config_file]
    (
        batchsize
    ) = args
    
    for bs_i in batchsize:
        with open(os.path.join(root, f"attacker_samplesize_{batchsize}.log"), "w") as wfid:
            exp = [
                f"--run.attack_sample_size={[bs_i]}"
            ]
            logger.info(f"running command {pres + exp}")
            subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        (
            [8, 512],
        )
    ]

    tasks += [
       (
            [16, 256],
        ) 
    ]

    tasks += [
       (
            [32, 128],
        ) 
    ]

    tasks += [
       (
            [64],
        ) 
    ]

    with Pool(4) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")