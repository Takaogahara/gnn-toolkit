"""torch_geometric.nn.models adaptation to perform graph classification"""
import copy
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.nn import global_mean_pool

from torch_geometric.typing import Adj
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn.resolver import (activation_resolver,
                                         normalization_resolver)
torch.manual_seed(8)


class BasicGNN(torch.nn.Module):
    """An abstract class for implementing basic GNN models."""

    def __init__(self, model_params):
        super(BasicGNN, self).__init__()
        act: Union[str, Callable, None] = "relu"
        act_first: bool = False
        act_kwargs: Optional[Dict[str, Any]] = None
        norm: Union[str, Callable, None] = None
        norm_kwargs: Optional[Dict[str, Any]] = None
        jk: Optional[str] = None

        in_channels = model_params["MODEL_FEAT_SIZE"]
        hidden_channels = model_params["MODEL_EMBEDDING_SIZE"]
        num_layers = model_params["MODEL_NUM_LAYERS"]
        out_channels = 1
        dropout = model_params["MODEL_DROPOUT_RATE"]

        edge_dim = model_params["MODEL_EDGE_DIM"]
        num_timesteps = model_params["MODEL_NUM_TIMESTEPS"]
        heads = model_params["MODEL_NUM_HEADS"]
        v2 = model_params["MODEL_ATT_V2"]
        improved = model_params["MODEL_GCN_IMPROVED"]
        kwargs = {"edge_dim": edge_dim,
                  "num_timesteps": num_timesteps,
                  "heads": heads,
                  "v2": v2,
                  "improved": improved}

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.dropout = dropout
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.jk_mode = jk
        self.act_first = act_first
        self.norm = norm if isinstance(norm, str) else None
        self.norm_kwargs = norm_kwargs

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels

        self.convs = ModuleList()
        if num_layers > 1:
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))
            in_channels = hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))
            in_channels = hidden_channels
        if out_channels is not None and jk is None:
            self._is_conv_to_out = True
            self.convs.append(
                self.init_conv(in_channels, out_channels, **kwargs))
        else:
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))

        self.norms = None
        if norm is not None:
            norm_layer = normalization_resolver(
                norm,
                hidden_channels,
                **(norm_kwargs or {}),
            )
            self.norms = ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(copy.deepcopy(norm_layer))
            if jk is not None:
                self.norms.append(copy.deepcopy(norm_layer))

        if jk is not None and jk != 'last':
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        if jk is not None:
            if jk == 'cat':
                in_channels = num_layers * hidden_channels
            else:
                in_channels = hidden_channels
            self.lin = Linear(in_channels, self.out_channels)

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        raise NotImplementedError

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, batch, *args, **kwargs):
        """"""
        xs: List[Tensor] = []
        for i in range(self.num_layers):
            try:
                x = self.convs[i](x, edge_index, *args, **kwargs)
            except TypeError:
                del kwargs["edge_attr"]
                x = self.convs[i](x, edge_index, *args, **kwargs)

            if i == self.num_layers - 1 and self.jk_mode is None:
                break
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if hasattr(self, 'jk'):
                xs.append(x)

        x = global_mean_pool(x, batch)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x
        return x

    @torch.no_grad()
    def inference(self, loader: NeighborLoader,
                  device: Optional[torch.device] = None) -> Tensor:
        r"""Performs layer-wise inference on large-graphs using
        :class:`~torch_geometric.loader.NeighborLoader`.
        :class:`~torch_geometric.loader.NeighborLoader` should sample the the
        full neighborhood for only one layer.
        This is an efficient way to compute the output embeddings for all
        nodes in the graph.
        Only applicable in case :obj:`jk=None` or `jk='last'`.
        """
        assert self.jk_mode is None or self.jk_mode == 'last'
        assert isinstance(loader, NeighborLoader)
        assert len(loader.dataset) == loader.data.num_nodes
        assert len(loader.num_neighbors) == 1
        assert not self.training

        x_all = loader.data.x.cpu()
        loader.data.n_id = torch.arange(x_all.size(0))

        for i in range(self.num_layers):
            xs: List[Tensor] = []
            for batch in loader:
                x = x_all[batch.n_id].to(device)
                edge_index = batch.edge_index.to(device)
                x = self.convs[i](x, edge_index)[:batch.batch_size]
                if i == self.num_layers - 1 and self.jk_mode is None:
                    xs.append(x.cpu())
                    continue
                if self.act is not None and self.act_first:
                    x = self.act(x)
                if self.norms is not None:
                    x = self.norms[i](x)
                if self.act is not None and not self.act_first:
                    x = self.act(x)
                if i == self.num_layers - 1 and hasattr(self, 'lin'):
                    x = self.lin(x)
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)

        del loader.data.n_id

        return x_all

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')
