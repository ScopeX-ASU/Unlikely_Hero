import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = f"Experiment/log/cifar10/vgg8/recovery/sens-aware"
script = "scan_recovery.py"
config_file = "config/cifar10/vgg8/recovery/recovery.yml"
configs.load(config_file, recursive=True)

def task_launcher(args):
    pres = [
        "python3", 
        script, 
        config_file
    ]
    eta, N_bit = args

    with open(os.path.join(root, f"WeightLocking_Nbit_{N_bit}_eta_{eta}.log"), "w") as wfid:
        exp = [
            f"--defense.eta={eta}",
            f"--quantize.N_bits={N_bit}"
        ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [(
            1, # acceptable accuracy drop for model,
            8
        )]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")