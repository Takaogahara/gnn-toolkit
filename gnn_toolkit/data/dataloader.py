from filelock import FileLock
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from .core import benchmark_MoleculeNet, MoleculeDataset

available_loaders = ["default", "moleculenet"]


def default_dataloader(parameters: dict):
    """Create dafault Dataloader

    Args:
        parameters (dict): Mango parsed parameters
    """
    task = parameters["TASK"]
    loader = parameters["DATA_DATALOADER"]
    # batch_size = parameters["SOLVER_BATCH_SIZE"]
    batch_size = 32
    premade = parameters["DATA_PREMADE"]

    path_raw = parameters["DATA_ROOT_PATH"]
    name_train = parameters["DATA_FILE_NAME"]

    if loader.lower() not in available_loaders:
        raise RuntimeError("Wrong loader, Available: \n"
                           f"{available_loaders}")

    # * Get dataset
    if loader.lower() == "default":
        if premade:
            # Get premade split
            with FileLock(f"{path_raw}raw/{name_train}.lock"):
                dataset = MoleculeDataset(path_raw, name_train, task, True)
                data_path = dataset.processed_paths

                set_train = [x for x in data_path if "data_train_" in x]
                set_test = [x for x in data_path if "data_test_" in x]
                _ = [x for x in data_path if "data_test_" in x]

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
