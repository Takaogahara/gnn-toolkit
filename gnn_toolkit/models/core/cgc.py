import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList

from torch_geometric.nn import CGConv
from torch_geometric.nn import global_mean_pool
torch.manual_seed(8)


class CGC(torch.nn.Module):
    def __init__(self, model_params):
        super(CGC, self).__init__()
        feature_size = model_params["MODEL_FEAT_SIZE"]
        edge_dim = model_params["MODEL_EDGE_DIM"]

        self.n_layers = model_params["MODEL_NUM_LAYERS"]
        self.dropout = model_params["MODEL_DROPOUT_RATE"]

        self.gnn_layers = ModuleList([])

        # CGC block
        self.cgc1 = CGConv(feature_size,
                           dim=edge_dim, batch_norm=True)

        for i in range(self.n_layers - 1):
            self.gnn_layers.append(CGConv(feature_size,
                                          dim=edge_dim, batch_norm=True))

        # Linear layers
        self.linear = Linear(feature_size, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        # Initial CGC
        x = self.cgc1(x, edge_index, edge_attr)

        for i in range(self.n_layers - 1):
            x = self.gnn_layers[i](x, edge_index, edge_attr)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # Pooling
        x = global_mean_pool(x, batch)

        # Output block
        x = self.linear(x)

        return x
