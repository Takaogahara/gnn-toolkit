from filelock import FileLock
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from .core import MoleculeDataset, benchmark_MoleculeNet

available_loaders = ["default", "moleculenet"]


def default_dataloader(parameters: dict):
    """Create dafault Dataloader

    Args:
        parameters (dict): Mango parsed parameters
    """
    task = parameters["TASK"]
    loader = parameters["DATA_DATALOADER"]
    batch_size = parameters["SOLVER_BATCH_SIZE"]
    premade = parameters["DATA_PREMADE"]

    path_raw = parameters["DATA_RAW_PATH"]
    name_train = parameters["DATA_RAW_FILE_NAME_TRAIN"]
    name_test = parameters["DATA_RAW_FILE_NAME_TEST"]
    name_val = parameters["DATA_RAW_FILE_NAME_VAL"]

    if loader.lower() not in available_loaders:
        raise RuntimeError("Wrong loader, Available: \n"
                           f"{available_loaders}")

    # TODO: "Universal" dataloader for premade

    if loader.lower() == "default":
        # * Get dataset
        # TODO: FileLock premade
        if premade:
            # Use pre made dataset
            set_train = MoleculeDataset(path_raw, name_train, task)
            set_test = MoleculeDataset(path_raw, name_test, task, test=True)
            _ = MoleculeDataset(path_raw, name_val, task, val=True)

        else:
            # Get random split (80, 10, 10)
            with FileLock(f"{path_raw}raw/{name_train}.lock"):
                dataset = MoleculeDataset(path_raw, name_train, task)
                lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
                lengths += [len(dataset) - sum(lengths)]
                set_train, set_test, _ = random_split(dataset, lengths)
                set_train = set_train.dataset
                set_test = set_test.dataset

        loader_train = DataLoader(set_train,
                                  batch_size=batch_size,
                                  shuffle=True)
        loader_test = DataLoader(set_test,
                                 batch_size=batch_size,
                                 shuffle=True)

    elif loader.lower() == "moleculenet":
        loader_train, loader_test, _ = benchmark_MoleculeNet(path_raw, "HIV",
                                                             batch_size)

    return loader_train, loader_test
