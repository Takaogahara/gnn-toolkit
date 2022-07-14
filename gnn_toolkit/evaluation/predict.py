import torch
import numpy as np

from utils import extract_configs
from data.dataloader import default_dataloader
from models.model import model_selection
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _get_information(dataloader):
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


def predict_model(config):
    """Test best parameters with validation dataset

    Args:
        config (dict): Ray Tune parsed parameters
    """
    all_preds = []
    all_labels = []
    parameters = extract_configs(config)

    # * Load DataLoaders
    loader_valid = default_dataloader(parameters, checkpoint=True)

    # * Get information
    pos_weight, num_edge, num_feat = _get_information(loader_valid)
    parameters["AUTO_LOSS_FN_POS_WEIGHT"] = pos_weight
    parameters["MODEL_EDGE_DIM"] = num_edge
    parameters["MODEL_FEAT_SIZE"] = num_feat

    # * Load Model
    model = model_selection(parameters, checkpoint=True)

    # * Predict
    with torch.no_grad():
        for batch in loader_valid:
            # * Use GPU
            batch.to(device)

            # * Passing the node features and the connection info
            pred = model(x=batch.x.float(),
                         edge_index=batch.edge_index,
                         edge_attr=batch.edge_attr.float(),
                         batch=batch.batch)

            # * Save pred results
            all_preds.append(np.rint(torch.sigmoid(pred).cpu(
            ).detach().numpy()))
            all_labels.append(batch.y.cpu().detach().numpy())

        all_preds = np.concatenate(all_preds).ravel()
        all_labels = np.concatenate(all_labels).ravel()

    correct = (all_preds == all_labels).sum().item()
    print(f"Predicted accuracy: {correct / len(all_labels)}")
