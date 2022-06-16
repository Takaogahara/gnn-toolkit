import torch
import numpy as np
from tqdm import tqdm

from .default_metrics import log_metrics
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    num_epoch = parameters["DATA_NUM_EPOCH"]

    # * Start run
    print("\n############################################## Start")
    best_loss = 1000
    early_stopping_counter = 0

    for epoch in range(1, num_epoch+1):
        # * TRAIN
        model.train()
        loss = _train_epoch(parameters, model, optimizer, loss_fn,
                            loader_train, epoch, num_epoch)

        # * TEST
        model.eval()
        if (epoch % 5 == 0) or (epoch == 1):
            loss = _test_epoch(parameters, model, loss_fn,
                               loader_test, epoch)

            # * Update update best loss
            if float(loss) < best_loss:
                best_loss = loss
                early_stopping_counter = 0

            # * Update esarly stop
            else:
                early_stopping_counter += 1

        torch_scheduler.step()

        if early_stopping_counter > 5:
            print("Early stopping due to no improvement.\n")
            print(f"Finishing training with best test loss: {best_loss}")
            print("############################################## End\n")
            return [best_loss]

    print(f"Finishing training with best test loss: {best_loss}")
    print("############################################## End\n")
    return [best_loss]


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
    name = parameters["RUN_NAME"]

    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 1

    txt = f"Epoch {current_epoch}/{num_epoch}"
    unit = "batch"
    with tqdm(loader, ncols=120, unit=unit, desc=txt) as bar:
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

        log_metrics(task, name, "train", current_epoch, all_preds, all_labels)

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
    name = parameters["RUN_NAME"]

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

    log_metrics(task, name, "test", current_epoch, all_preds, all_labels)

    return running_loss/step
