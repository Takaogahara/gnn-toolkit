import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ModuleList

from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp
torch.manual_seed(8)


class GAT(torch.nn.Module):
    def __init__(self, model_params):
        super(GAT, self).__init__()
        feature_size = model_params["MODEL_FEAT_SIZE"]
        edge_dim = model_params["MODEL_EDGE_DIM"]

        embedding_size = model_params["MODEL_EMBEDDING_SIZE"]
        self.n_layers = model_params["MODEL_NUM_LAYERS"]
        dropout_rate = model_params["MODEL_DROPOUT_RATE"]
        dense_neurons = model_params["MODEL_DENSE_NEURONS"]

        n_heads = model_params["MODEL_NUM_HEADS"]

        self.gnn_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        # GAT block
        self.gat1 = GATConv(feature_size,
                            embedding_size,
                            heads=n_heads,
                            dropout=dropout_rate,
                            edge_dim=edge_dim)
        self.transf1 = Linear(embedding_size*n_heads, embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)

        # GAT, Transform, BatchNorm block
        for i in range(self.n_layers):
            self.gnn_layers.append(GATConv(embedding_size,
                                           embedding_size,
                                           heads=n_heads,
                                           dropout=dropout_rate,
                                           edge_dim=edge_dim))

            self.transf_layers.append(
                Linear(embedding_size*n_heads, embedding_size))
            self.bn_layers.append(BatchNorm1d(embedding_size))

        # Linear layers
        self.linear1 = Linear(embedding_size, dense_neurons)
        self.bn2 = BatchNorm1d(dense_neurons)
        self.linear2 = Linear(dense_neurons, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        # Initial GAT
        x = self.gat1(x, edge_index, edge_attr)
        x = torch.relu(self.transf1(x))
        x = self.bn1(x)

        for i in range(self.n_layers):
            x = self.gnn_layers[i](x, edge_index, edge_attr)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)

        # Pooling
        x = gmp(x, batch)

        # Output block
        x = F.dropout(x, p=0.0, training=self.training)
        x = torch.relu(self.linear1(x))
        x = self.bn2(x)
        x = self.linear2(x)

        if torch.isnan(torch.mean(self.linear2.weight)):
            raise RuntimeError("Exploding gradients. Tune learning rate")

        return x
