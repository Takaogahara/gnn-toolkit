import os
import uuid
import torch
import argparse
from ray import tune
import mlflow.pytorch
from ray.tune.schedulers import ASHAScheduler

from logo import print_logo
from utils import get_ray_config, TelegramReport
from run import gnn_toolkit, test_best_model

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
ray_space = get_ray_config(args.cfg)


###############################################################################
# Hyperparameter search
###############################################################################
def trial_str_creator(trial):
    return "{}_{}".format(trial.trainable_name, trial.trial_id)


class RayExperiment:
    def __init__(self, ray_space):
        self.param_space = ray_space
        self.objective = gnn_toolkit
        self.n_samples = ray_space["RUN_RAY_SAMPLES"][0]
        self.max_epoch = ray_space["RUN_RAY_MAX_EPOCH"][0]
        self.cpu = ray_space["RUN_RAY_CPU"][0]
        self.gpu = ray_space["RUN_RAY_GPU"][0]
        self.budget = ray_space["RUN_RAY_TIME_BUDGET_S"][0]
        self.resume = ray_space["RUN_RAY_RESUME"][0]

        self.mlflow_uri = ray_space["RUN_MLFLOW_URI"][0]
        self.telegram = ray_space["RUN_TELEGRAM_VERBOSE"][0]
        self.epoch = ray_space["SOLVER_NUM_EPOCH"][0]
        self.task = ray_space["TASK"][0]

        self.experiment_name = ray_space["RUN_NAME"][0]
        if self.experiment_name == "None":
            self.experiment_name = str(uuid.uuid4()).split("-")[0]

        ray_space["MLFLOW_NAME"] = [self.experiment_name]
        try:
            mlflow.create_experiment(self.experiment_name)
        except Exception:
            pass
        ray_space["mlflow"] = {"experiment_name": self.experiment_name,
                               "tracking_uri": mlflow.get_tracking_uri()}

    def execute(self):
        print_logo()
        print(f"Running on: {device}\n")
        TelegramReport.start_eval(device, self.task,
                                  self.n_samples, self.epoch,
                                  self.telegram)

        scheduler = ASHAScheduler(max_t=self.max_epoch,
                                  grace_period=1, reduction_factor=1.2)

        resources = {"cpu": self.cpu, "gpu": self.gpu}
        result = tune.run(tune.with_parameters(self.objective),
                          name=self.experiment_name,
                          config=self.param_space,
                          resources_per_trial=resources,
                          num_samples=self.n_samples,
                          scheduler=scheduler,
                          metric="loss",
                          mode="min",
                          local_dir="./ray_results",
                          trial_name_creator=trial_str_creator,
                          trial_dirname_creator=trial_str_creator,
                          verbose=3,
                          resume=self.resume,
                          raise_on_failed_trial=False)

        best_trial = result.get_best_trial("loss", "min", "last")
        print(f"\nBest parameters: {best_trial.config}\n")
        print(f"Best loss: {best_trial.last_result['loss']}")
        TelegramReport.end_eval(best_trial, self.telegram)

        return best_trial

    def test_model(self, best_trial):
        best_config = best_trial.config

        chkp_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
        best_trial.config["chkp_path"] = chkp_path

        test_best_model(best_config)
        print(f"Best trial dir: {best_trial.logdir}")


ray_tune = RayExperiment(ray_space)
best_trial = ray_tune.execute()
ray_tune.test_model(best_trial)
