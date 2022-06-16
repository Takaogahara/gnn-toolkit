import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import deepchem as dc
import torch
from torch_geometric.data import Dataset


class MoleculeDataset(Dataset):
    def __init__(self, root, filename, task, test=False, val=False,
                 transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.test = test
        self.validation = val
        self.task = task
        self.filename = filename
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        elif self.validation:
            return [f'data_val_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

        # * Featurize molecule
        txt = "Generating graphs"
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0],
                               desc=txt, ncols=120):
            f = featurizer.featurize(mol["Smiles"])
            data = f[0].to_pyg_graph()
            data.y = self._get_label(mol["Labels"])
            data.smiles = mol["Smiles"]
            if self.test:
                torch.save(data,
                           os.path.join(self.processed_dir,
                                        f'data_test_{index}.pt'))
            elif self.validation:
                torch.save(data,
                           os.path.join(self.processed_dir,
                                        f'data_val_{index}.pt'))
            else:
                torch.save(data,
                           os.path.join(self.processed_dir,
                                        f'data_{index}.pt'))

    def _get_label(self, label):
        label = np.asarray([label])

        if self.task == "Classification":
            lbl = torch.tensor(label, dtype=torch.int64)
        elif self.task == "Regression":
            lbl = torch.tensor(label, dtype=torch.float16)

        return lbl

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_test_{idx}.pt'))
        elif self.validation:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_val_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_{idx}.pt'))
        return data
