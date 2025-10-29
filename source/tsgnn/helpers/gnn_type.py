from enum import Enum, auto
from typing import Union, Optional
from torch.nn import Linear, Module
from torch import Tensor
from torch_geometric.nn.conv import GraphConv
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
)
from torch.nn.modules.linear import init

from triton_tests.gat import GATConv
from models.triton_nn.mean_gnn import TritonMeanGNN
from models.triton_nn.gat import TritonGAT


class GraphConvWrap(GraphConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Optional[Adj] = None,
                edge_weight: OptTensor = None, size: Size = None, **kwargs) -> Tensor:
        return super().forward(x, edge_index=edge_index, edge_weight=edge_weight, size=size)


def get_fan_in(model: Module) -> float:
    return init._calculate_fan_in_and_fan_out(model.aggr_lin.weight)[0]


class GNNType(Enum):
    """
        an object for the different core
    """
    GAT = auto()
    MEAN_GNN = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return GNNType[s]
        except KeyError:
            raise ValueError()

    def get_module(self, in_channel: int, out_channel: int, triton_on: bool):
        if self is GNNType.GAT:
            if triton_on:
                return TritonGAT(in_channels=in_channel, out_channels=out_channel)
            else:
                return GATConv(in_channels=in_channel, out_channels=out_channel, heads=1, bias=False)
        elif self is GNNType.MEAN_GNN:
            if triton_on:
                return TritonMeanGNN(in_channels=in_channel, out_channels=out_channel)
            else:
                return GraphConvWrap(in_channels=in_channel, out_channels=out_channel,
                                     aggr='mean', node_dim=0, bias=False)
        else:
            raise ValueError(f'model {self.name} not supported')

    def uses_triton(self) -> bool:
        return self in [GNNType.MEAN_GNN, GNNType.GAT]
