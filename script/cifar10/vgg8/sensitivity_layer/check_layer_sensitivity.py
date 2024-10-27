import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = "Experiment/log/cifar10/vgg8/layer_sensitivity"
script = "scan_layer_sensitivity.py"
config_file = "config/cifar10/vgg8/defender/defender.yml"

configs.load(config_file, recursive=True)

def task_launcher(args):
    pres = ["python3", script, config_file]
    N_bits = args
    with open(os.path.join(root, f"Layer_Sensitivity_Nbit_{N_bits}.log"), "w") as wfid:
        exp = [
        ]
        logger.info(f"running command {pres + exp}")
        logger.info(f"N bit is {N_bits}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [8]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
