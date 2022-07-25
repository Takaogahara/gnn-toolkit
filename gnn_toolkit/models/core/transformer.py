import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList

from torch_geometric.nn import TransformerConv
from torch_geometric.nn import global_mean_pool
torch.manual_seed(8)


class Transformer(torch.nn.Module):
    def __init__(self, model_params):
        super(Transformer, self).__init__()
        feature_size = model_params["MODEL_FEAT_SIZE"]
        edge_dim = model_params["MODEL_EDGE_DIM"]

        embedding_size = model_params["MODEL_EMBEDDING_SIZE"]
        self.n_layers = model_params["MODEL_NUM_LAYERS"]
        self.dropout = model_params["MODEL_DROPOUT_RATE"]

        n_heads = model_params["MODEL_NUM_HEADS"]

        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])

        # Transformation layer
        self.conv1 = TransformerConv(feature_size,
                                     embedding_size,
                                     heads=n_heads,
                                     dropout=self.dropout,
                                     edge_dim=edge_dim,
                                     beta=True)
        self.transf1 = Linear(embedding_size*n_heads, embedding_size)

        # Other layers
        for i in range(self.n_layers - 1):
            self.conv_layers.append(TransformerConv(embedding_size,
                                                    embedding_size,
                                                    heads=n_heads,
                                                    dropout=self.dropout,
                                                    edge_dim=edge_dim,
                                                    beta=True))

            self.transf_layers.append(Linear(embedding_size*n_heads,
                                             embedding_size))

        # Linear layers
        self.linear = Linear(embedding_size, 1)

    def forward(self, x, edge_attr, edge_index, batch):
        # Initial transformation
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(self.transf1(x))

        for i in range(self.n_layers - 1):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(self.transf_layers[i](x))

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)

        # Output block
        x = self.linear(x)

        return x
