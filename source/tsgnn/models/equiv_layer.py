from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch.nn import Module, ModuleList, Parameter, Identity
from typing import Optional, Tuple, Dict
import torch
import math
from torch.nn.modules.linear import init

from helpers.gnn_type import GNNType
from helpers.gnn_type import get_fan_in


class EquivLayer(Module):
    def __init__(self, in_channel: int, out_channel: int, ls_num_layers: int, triton_on: bool, gnn_type: GNNType):
        super(EquivLayer, self).__init__()
        self.out_channel = out_channel

        # x
        self.x_self = gnn_type.get_module(in_channel=in_channel, out_channel=out_channel, triton_on=triton_on)
        self.x_aggr = gnn_type.get_module(in_channel=in_channel, out_channel=out_channel, triton_on=triton_on)
        self.pool2x = gnn_type.get_module(in_channel=in_channel, out_channel=out_channel, triton_on=triton_on)

        self.x_bias = Parameter(torch.empty(out_channel))
        fan_in = get_fan_in(model=self.x_self)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.x_bias, -bound, bound)

        # y
        self.y_self = gnn_type.get_module(in_channel=in_channel, out_channel=out_channel, triton_on=triton_on)
        self.y_aggr = gnn_type.get_module(in_channel=in_channel, out_channel=out_channel, triton_on=triton_on)
        self.pool2y = gnn_type.get_module(in_channel=in_channel, out_channel=out_channel, triton_on=triton_on)

        self.y_bias = Parameter(torch.empty(out_channel))
        fan_in = get_fan_in(model=self.y_self)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.y_bias, -bound, bound)

        # ls
        self.ls_num_layers = ls_num_layers
        if ls_num_layers >= 0:
            self.ls2y_layers = ModuleList([gnn_type.get_module(in_channel=in_channel, out_channel=out_channel,
                                                               triton_on=triton_on)
                                           for _ in range(ls_num_layers + 1)])
            self.ls2x_layers = ModuleList([gnn_type.get_module(in_channel=in_channel, out_channel=out_channel,
                                                               triton_on=triton_on)
                                           for _ in range(ls_num_layers + 1)])
            # hetero
            self.ls2y_hetero_layers = [gnn_type.get_module(in_channel=in_channel, out_channel=out_channel,
                                                           triton_on=triton_on)
                                       for _ in range(1, ls_num_layers + 1)]
            self.ls2y_hetero_layers = ModuleList([Identity()] + self.ls2y_hetero_layers)
            self.ls2x_hetero_layers = [gnn_type.get_module(in_channel=in_channel, out_channel=out_channel,
                                                           triton_on=triton_on)
                                       for _ in range(1, ls_num_layers + 1)]
            self.ls2x_hetero_layers = ModuleList([Identity()] + self.ls2x_hetero_layers)

    def single_forward(self, x: Tensor, y: Tensor, xy_conversions: Dict[str, Tensor], device, is_x: bool,
                       edge_index: Optional[Adj] = None, rowptr: OptTensor = None,
                       indices: OptTensor = None) -> Tensor:
        # x Shape: [num_nodes, in_dim, in_channel]
        # y Shape: [num_nodes, num_classes, in_channel]
        x_str = 'x' if is_x else 'y'

        out_x = getattr(self, f'{x_str}_self')(x, edge_index=edge_index, rowptr=rowptr, indices=indices)
        out_x.add_(getattr(self, f'{x_str}_aggr')(x.mean(dim=1, keepdim=True), edge_index=edge_index,
                                                  rowptr=rowptr, indices=indices))
        w = xy_conversions[f'pool2{x_str}_w'].to(device=device)
        b = xy_conversions[f'pool2{x_str}_b'].to(device=device)
        out_x.add_(getattr(self, f'pool2{x_str}')(torch.einsum("ndk,dp->npk", y.mean(dim=1, keepdim=True), w)
                                                  + b.unsqueeze(dim=-1),
                                                  edge_index=edge_index, rowptr=rowptr, indices=indices))

        for num_layer in range(self.ls_num_layers + 1):
            w = xy_conversions[f'ls2{x_str}_{num_layer}_w'].to(device=device)
            b = xy_conversions[f'ls2{x_str}_{num_layer}_b'].to(device=device)
            out_x.add_(getattr(self, f'ls2{x_str}_layers')[num_layer](torch.einsum("ndk,dp->npk", y, w)
                                                                      + b.unsqueeze(dim=-1),
                                                                      edge_index=edge_index, rowptr=rowptr,
                                                                      indices=indices))
            if num_layer != 0:
                # y2x hetero
                w = xy_conversions[f'ls2{x_str}_{num_layer}_hetero_w'].to(device=device)
                b = xy_conversions[f'ls2{x_str}_{num_layer}_hetero_b'].to(device=device)
                out_x.add_(getattr(self, f'ls2{x_str}_hetero_layers')[num_layer](torch.einsum("ndk,dp->npk", y, w)
                                                                                 + b.unsqueeze(dim=-1),
                                                                                 edge_index=edge_index, rowptr=rowptr,
                                                                                 indices=indices))
        return out_x

    def forward(self, x: Tensor, y: Tensor, xy_conversions: Dict[str, Tensor], device,
                edge_index: Optional[Adj] = None, rowptr: OptTensor = None,
                indices: OptTensor = None) -> Tuple[Tensor, Tensor]:
        out_x = self.single_forward(x=x, y=y, xy_conversions=xy_conversions, device=device, is_x=True,
                                    edge_index=edge_index, rowptr=rowptr, indices=indices)
        y = self.single_forward(x=y, y=x, xy_conversions=xy_conversions, device=device, is_x=False,
                                edge_index=edge_index, rowptr=rowptr, indices=indices)
        return out_x, y
