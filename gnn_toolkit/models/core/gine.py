import torch
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn.conv import GINEConv, MessagePassing
from .basic_gnn import BasicGNN
torch.manual_seed(8)


class GINE(BasicGNN):

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:

        del kwargs["num_timesteps"]
        del kwargs["heads"]
        del kwargs["v2"]
        del kwargs["improved"]

        mlp = Sequential(Linear(in_channels, out_channels),
                         ReLU(), Linear(out_channels, out_channels))

        return GINEConv(mlp, **kwargs)
