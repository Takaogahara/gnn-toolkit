from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader


def benchmark_MoleculeNet(root, name, batch_size,
                          transform=None, pre_transform=None, pre_filter=None):
    """
    root = Where the dataset should be stored. This folder is split
    into raw_dir (downloaded dataset) and processed_dir (processed data).
    """

    dataset = MoleculeNet(root, name=name,
                          transform=transform,
                          pre_transform=pre_transform,
                          pre_filter=pre_filter).shuffle()

    N = len(dataset) // 10
    set_test = dataset[N:2 * N]
    set_train = dataset[2 * N:]
    set_val = dataset[:N]

    loader_train = DataLoader(set_train, batch_size=batch_size,
                              shuffle=True)
    loader_test = DataLoader(set_test, batch_size=batch_size)
    loader_val = DataLoader(set_val, batch_size=batch_size)

    return loader_train, loader_test, loader_val
