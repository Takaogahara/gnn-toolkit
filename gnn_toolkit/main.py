import torch
import argparse
import mlflow.pytorch
from mango import Tuner
from datetime import datetime

from logo import print_logo
from utils import ConfigFile, TelegramReport
from run import start

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###############################################################################
# Parser
###############################################################################
parser = argparse.ArgumentParser(description="Train GNN models")
parser.add_argument("--cfg",
                    type=str,
                    required=True,
                    help="Path to config file")
args = parser.parse_args()


###############################################################################
# Load config and initialize experiment
###############################################################################
cfg = ConfigFile()
config = cfg.get_raw_config(args.cfg)

###############################################################################
# Hyperparameter search
###############################################################################


class MangoExperiment:
    def __init__(self, config):
        self.param_space = config
        self.objective = start
        self.iter = config["RUN_MANGO_ITER"][0]
        self.optim = "Bayesian"
        self.initial_rnd = 1
        self.domain_size = None

        self.telegram = config["RUN_TELEGRAM_VERBOSE"][0]
        self.epoch = config["DATA_NUM_EPOCH"][0]
        self.task = config["TASK"][0]
        model = config["MODEL_ARCHITECTURE"][0]

        experiment_name = config["RUN_NAME"][0]
        if experiment_name == "None":
            now = datetime.now()
            now = now.strftime("%d-%m-%Y_%H:%M:%S")
            experiment_name = f"{self.task}_{model}_{now}"

        config["MLFLOW_NAME"] = [experiment_name]
        mlflow.set_tracking_uri(config["RUN_MLFLOW_URI"][0])
        mlflow.create_experiment(f"{experiment_name}")

    def execute(self):
        print_logo()
        print("initializing hyperparameter search...")
        print(f"Running on: {device}")
        TelegramReport.start_mango(device, self.task, self.iter,
                                   self.epoch, self.telegram)

        tuner = Tuner(self.param_space, self.objective,
                      conf_dict={"optimizer": self.optim,
                                 "num_iteration": self.iter,
                                 "initial_random": self.initial_rnd,
                                 "domain_size": self.domain_size})
        results = tuner.minimize()

        print(f"\nBest parameters: {results['best_params']}\n")
        print(f"Best accuracy: {results['best_objective']}")
        TelegramReport.end_mango(results, self.telegram)


exp = MangoExperiment(config)
exp.execute()
