import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ModuleList

from torch_geometric.nn import CGConv
from torch_geometric.nn import global_max_pool as gmp
torch.manual_seed(8)


class CGC(torch.nn.Module):
    def __init__(self, model_params):
        super(CGC, self).__init__()
        feature_size = model_params["MODEL_FEAT_SIZE"]
        edge_dim = model_params["MODEL_EDGE_DIM"]

        self.n_layers = model_params["MODEL_NUM_LAYERS"]
        self.dropout_rate = model_params["MODEL_DROPOUT_RATE"]
        dense_neurons = model_params["MODEL_DENSE_NEURONS"]

        self.gnn_layers = ModuleList([])

        # CGC block
        self.cgc1 = CGConv(feature_size,
                           dim=edge_dim, batch_norm=True)

        # CGC, Transform, BatchNorm block
        for i in range(self.n_layers):
            self.gnn_layers.append(CGConv(feature_size,
                                          dim=edge_dim, batch_norm=True))

        # Linear layers
        self.linear1 = Linear(feature_size, dense_neurons)
        self.bn2 = BatchNorm1d(dense_neurons)
        self.linear2 = Linear(dense_neurons, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        # Initial CGC
        x = self.cgc1(x, edge_index, edge_attr)

        for i in range(self.n_layers):
            x = self.gnn_layers[i](x, edge_index, edge_attr)

        # Pooling
        x = gmp(x, batch)

        # Output block
        x = F.dropout(x, p=0.0, training=self.training)  # dropout_rate
        x = torch.relu(self.linear1(x))
        x = self.bn2(x)
        x = self.linear2(x)

        if torch.isnan(torch.mean(self.linear2.weight)):
            raise RuntimeError("Exploding gradients. Tune learning rate")

        return x
