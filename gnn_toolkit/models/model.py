import torch

from . import core
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


available_models = {"transformer": core.Transformer,
                    "gat": core.GAT,
                    "ecc": core.ECC,
                    "cgc": core.CGC,
                    "attentivefp": core.Attentive}


def model_selection(parameters: dict):
    """Select model

    Args:
        parameters (dict): Parameters YAML file

    Raises:
        RuntimeError: If selection don't match

    Returns:
        Pytorch: Model
    """
    model_name = parameters["MODEL_ARCHITECTURE"]
    is_transfer = parameters["MODEL_USE_TRANSFER"]
    transfer_path = parameters["MODEL_TRANSFER_PATH"]

    if is_transfer:
        model = torch.load(transfer_path)

    else:
        if model_name.lower() not in available_models:
            raise RuntimeError("Wrong model, Available: \n"
                               f"{available_models.keys()}")

        model = available_models[model_name.lower()]
        model = model(model_params=parameters)

    model = model.to(device)

    return model
