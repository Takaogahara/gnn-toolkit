from filelock import FileLock
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from .core import benchmark_MoleculeNet, MoleculeDataset

available_loaders = ["default", "moleculenet"]
moleculenet_dataset = ["esol", "freesolv", "lipo", "pcba",
                       "muv", "hiv", "bace", "bbpb", "tox21",
                       "toxcast", "sider", "clintox"]


def default_dataloader(parameters: dict, checkpoint=False):
    """Create dafault Dataloader

    Args:
        parameters (dict): Mango parsed parameters
    """
    task = parameters["TASK"]
    loader = parameters["DATA_DATALOADER"]
    batch_size = parameters["SOLVER_BATCH_SIZE"]
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
                set_train, set_test, valid_set = _get_premade_loaders(dataset)

        else:
            # Get random split (80, 10, 10)
            # TODO: Check for errors in random split
            with FileLock(f"{path_raw}raw/{name_train}.lock"):
                dataset = MoleculeDataset(path_raw, name_train, task)
                lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
                lengths += [len(dataset) - sum(lengths)]
                set_train, set_test, valid_set = random_split(dataset, lengths)
                set_train = set_train.dataset
                set_test = set_test.dataset
                valid_set = valid_set.dataset

        ldl_train = DataLoader(set_train,
                               batch_size=batch_size,
                               shuffle=True)
        ldl_test = DataLoader(set_test,
                              batch_size=batch_size,
                              shuffle=True)
        ldl_test = DataLoader(valid_set,
                              batch_size=batch_size,
                              shuffle=True)

    elif loader.lower() in moleculenet_dataset:
        ldl_train, ldl_test, ldl_valid = benchmark_MoleculeNet(path_raw,
                                                               loader.lower(),
                                                               batch_size)

    if checkpoint:
        return ldl_valid
    else:
        return ldl_train, ldl_test


def _get_premade_loaders(dataset):
    train = [dataset[x].index for x in range(len(dataset)
                                             ) if dataset[x].set == "Train"]
    test = [dataset[x].index for x in range(len(dataset)
                                            ) if dataset[x].set == "Test"]
    valid = [dataset[x].index for x in range(len(dataset)
                                             ) if dataset[x].set == "Valid"]

    set_train = dataset[train]
    set_test = dataset[test]
    set_valid = dataset[valid]

    return set_train, set_test, set_valid
