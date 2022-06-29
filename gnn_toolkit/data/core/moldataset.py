import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import deepchem as dc
import torch
from torch_geometric.data import Dataset


class MoleculeDataset(Dataset):
    def __init__(self, root, filename, task, premade=False,
                 transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.task = task
        self.filename = filename
        self.premade = premade
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

        if self.premade:
            df_train = self.data[self.data["Set"] == "Train"]
            df_test = self.data[self.data["Set"] == "Test"]
            df_valid = self.data[self.data["Set"] == "Valid"]

            list_train = [f'data_train_{i}.pt' for i in list(df_train.index)]
            list_test = [f'data_test_{i}.pt' for i in list(df_test.index)]
            list_valid = [f'data_val_{i}.pt' for i in list(df_valid.index)]

            return list_train + list_test + list_valid

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

            # * Save premade
            try:
                data.set = mol["Set"]
                data.index = index
                if data.set == "Test":
                    torch.save(data,
                               os.path.join(self.processed_dir,
                                            f'data_test_{index}.pt'))
                elif data.set == "Valid":
                    torch.save(data,
                               os.path.join(self.processed_dir,
                                            f'data_val_{index}.pt'))
                elif data.set == "Train":
                    torch.save(data,
                               os.path.join(self.processed_dir,
                                            f'data_train_{index}.pt'))
            except KeyError:
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
        try:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_test_{idx}.pt'))
        except FileNotFoundError:
            try:
                data = torch.load(os.path.join(self.processed_dir,
                                               f'data_val_{idx}.pt'))
            except FileNotFoundError:
                try:
                    data = torch.load(os.path.join(self.processed_dir,
                                                   f'data_train_{idx}.pt'))
                except FileNotFoundError:
                    data = torch.load(os.path.join(self.processed_dir,
                                                   f'data_{idx}.pt'))
        return data
