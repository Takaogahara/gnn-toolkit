import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


available_optim = ["sgd", "adam"]


def solver_selection(model, parameters: dict):
    """Select solver

    Args:
        model (_type_): Generated model
        parameters (dict): Parameters YAML file

    Returns:
        _type_: Optimizer
        _type_: Loss function
    """
    optim = _get_optimizer(model, parameters)
    loss_fn = _get_lossfn(parameters)

    return optim, loss_fn


def _get_optimizer(model, parameters: dict):
    """Create optimizer

    Args:
        model (_type_): Generated model
        parameters (dict): Parameters YAML file

    Raises:
        RuntimeError: If selection don't match

    Returns:
        _type_: Optimizer
    """
    optim_name = parameters["SOLVER_OPTIMIZER"]
    lr = parameters["SOLVER_LEARNING_RATE"]
    l2_penalty = parameters["SOLVER_L2_PENALTY"]
    sgd_momentum = parameters["SOLVER_SGD_MOMENTUM"]
    adam_beta_1 = parameters["SOLVER_ADAM_BETA_1"]
    adam_beta_2 = parameters["SOLVER_ADAM_BETA_2"]

    if optim_name.lower() not in available_optim:
        raise RuntimeError("Wrong optimizer, Available: \n"
                           f"{available_optim}")

    if optim_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr,
                                    momentum=sgd_momentum,
                                    weight_decay=l2_penalty)

    elif optim_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     betas=(adam_beta_1, adam_beta_2),
                                     weight_decay=l2_penalty)

    return optimizer


def _get_lossfn(parameters: dict):
    """Create loss function

    Args:
        parameters (dict): Parameters YAML file

    Returns:
        _type_: Loss function
    """

    task = parameters["TASK"]
    optim_weight_input = parameters["SOLVER_LOSS_FN_POS_WEIGHT"]
    optim_weight_auto = parameters["AUTO_LOSS_FN_POS_WEIGHT"]

    if task == "Classification":
        #  * < 1 increases precision, > 1 increases recall
        if optim_weight_input != "Auto":
            weight = torch.tensor([optim_weight_input],
                                  dtype=torch.float32).to(device)

        else:
            weight = torch.tensor([optim_weight_auto],
                                  dtype=torch.float32).to(device)

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)

    elif task == "Regression":
        loss_fn = torch.nn.MSELoss()

    return loss_fn
