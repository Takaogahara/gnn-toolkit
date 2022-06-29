import torch
import mlflow.pytorch
from ray.tune.integration.mlflow import mlflow_mixin

from utils import TelegramReport, extract_configs
from data.dataloader import default_dataloader
from models.model import model_selection
from solvers.default_solver import solver_selection
from train.default_train import fit


def get_information(dataloader):
    """Calculate positive class weight to use in solver

    Args:
        train_dataloader (Pytorch dataloader): Train dataloader

    Returns:
        float: Weight
    """
    try:
        num_pos = sum(dataloader.dataset.data.Labels)
        num_neg = len(dataloader.dataset.data.Labels) - num_pos
        pos_weight = num_neg / num_pos

        num_edge_features = dataloader.dataset.num_edge_features
        num_features = dataloader.dataset.num_features

    except Exception:
        num_pos = int(sum(dataloader.dataset.data.y.numpy())[0])
        num_neg = len(dataloader.dataset.data.y.numpy()) - num_pos
        pos_weight = num_neg / num_pos

        num_edge_features = dataloader.dataset.data.num_edge_features
        num_features = dataloader.dataset.data.num_features

    return pos_weight, num_edge_features, num_features


@mlflow_mixin
def gnn_toolkit(config, checkpoint_dir=None):
    """Start run

    Args:
        parameters (dict): Mango parsed parameters

    Returns:
        int: Best loss
    """
    parameters = extract_configs(config)
    mlflow.set_experiment(parameters["MLFLOW_NAME"])

    # * Load DataLoaders
    loader_train, loader_test = default_dataloader(parameters)

    # * Get information
    pos_weight, num_edge, num_feat = get_information(loader_train)
    parameters["AUTO_LOSS_FN_POS_WEIGHT"] = pos_weight
    parameters["MODEL_EDGE_DIM"] = num_edge
    parameters["MODEL_FEAT_SIZE"] = num_feat

    # * Log Parameters in mlflow
    for key in parameters.keys():
        mlflow.log_param(key, parameters[key])

    # * Load Model
    model = model_selection(parameters)
    _ = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # * Get optimizer, loss function, scheduler
    optimizer, loss_fn = solver_selection(model, parameters)
    torch_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=parameters["SOLVER_SCHEDULER_GAMMA"])

    # * Train
    best_loss = fit(model, parameters, optimizer, loss_fn,
                    loader_train, loader_test, torch_scheduler)

    TelegramReport.report_run(parameters["RUN_TELEGRAM_VERBOSE"])
    return best_loss
