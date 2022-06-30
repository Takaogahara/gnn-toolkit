import os
import torch
import numpy as np
from ray import tune
from tqdm import tqdm
import mlflow.pytorch
from ray.tune.integration.mlflow import mlflow_mixin

from .default_metrics import log_metrics
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@mlflow_mixin
def fit(model, parameters: dict, optimizer, loss_fn,
        loader_train, loader_test, torch_scheduler):
    """Control training phase

    Args:
        model (_type_): Generated model
        parameters (dict): Parameters YAML file
        optimizer (_type_): Generated optimizer
        loss_fn (_type_): Generated loss function
        loader_train (_type_): Train dataloaders
        loader_test (_type_): Test dataloaders
        scheduler (_type_): Pytorch scheduler

    Returns:
        int: Best loss
    """
    num_epoch = parameters["SOLVER_NUM_EPOCH"]

    # * Start run
    # print("\n############################################## Start")

    for epoch in range(1, num_epoch+1):
        # * TRAIN
        model.train()
        loss_tr = _train_epoch(parameters, model, optimizer, loss_fn,
                               loader_train, epoch, num_epoch)
        mlflow.log_metric(key="Train loss", value=float(loss_tr), step=epoch)

        # * TEST
        model.eval()
        loss_ts = _test_epoch(parameters, model, loss_fn,
                              loader_test, epoch)
        mlflow.log_metric(key="Test loss", value=float(loss_ts), step=epoch)

        torch_scheduler.step()

        # * Save checkpoint
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=loss_ts)
    # print(f"Finishing training with best test loss: {best_loss}")
    # print("############################################## End\n")


def _train_epoch(parameters, model, optimizer, loss_fn,
                 loader, current_epoch: int, num_epoch: int):
    """Train model for one epoch

    Args:
        parameters (dict): Parameters YAML file
        model (_type_): Generated model
        optimizer (_type_): Generated optimizer
        loss_fn (_type_): Generated loss function
        loader (_type_): Dataloader
        current_epoch (int): Current epoch
        num_epoch (int): Total epochs

    Returns:
        float: Loss
    """
    task = parameters["TASK"]

    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 1

    txt = f"Epoch {current_epoch}/{num_epoch}"
    unit = "batch"
    with tqdm(loader, ncols=120, unit=unit,
              desc=txt, disable=True) as bar:
        for batch in bar:

            # * Use GPU
            batch.to(device)

            # * Reset gradients
            optimizer.zero_grad()

            # * Passing the node features and the connection info
            pred = model(x=batch.x.float(),
                         edge_index=batch.edge_index,
                         edge_attr=batch.edge_attr.float(),
                         batch=batch.batch)

            # * Calculating the loss and gradients
            loss = loss_fn(torch.squeeze(pred),
                           torch.squeeze(batch.y.float()))
            loss.backward()

            # * Clip gradients to aviod exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            # * Update gradients
            optimizer.step()

            # * Update tracking
            running_loss += loss.detach().item()
            step += 1

            # * Save pred results
            all_preds.append(
                np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
            all_labels.append(batch.y.cpu().detach().numpy())

            # * Update progress bar
            partial = running_loss/step
            bar.set_postfix_str(f"loss: {round(partial, 5)}")

        all_preds = np.concatenate(all_preds).ravel()
        all_labels = np.concatenate(all_labels).ravel()

        log_metrics(task, "train", current_epoch, all_preds, all_labels)

        return running_loss/step


@torch.no_grad()
def _test_epoch(parameters, model, loss_fn, loader,
                current_epoch: int):
    """Test model for one epoch

    Args:
        parameters (dict): Parameters YAML file
        model (_type_): Generated model
        loss_fn (_type_): Generated loss function
        loader (_type_): Dataloader
        current_epoch (int): Current epoch

    Returns:
        float: Loss
    """
    task = parameters["TASK"]

    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0

    for batch in loader:
        # * Use GPU
        batch.to(device)

        # * Passing the node features and the connection info
        pred = model(x=batch.x.float(),
                     edge_index=batch.edge_index,
                     edge_attr=batch.edge_attr.float(),
                     batch=batch.batch)

        # * Calculating the loss and gradients
        loss = loss_fn(torch.squeeze(pred),
                       torch.squeeze(batch.y.float()))

        # * Update tracking
        running_loss += loss.item()
        step += 1

        # * Save pred results
        all_preds.append(
            np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(batch.y.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    log_metrics(task, "test", current_epoch, all_preds, all_labels)

    return running_loss/step
