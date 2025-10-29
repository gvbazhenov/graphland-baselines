import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
import torch_geometric


class TritonGAT(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, add_self_loops: bool = True):
        super().__init__(aggr=None)
        self.add_self_loops = add_self_loops
        self.aggr_lin = Linear(in_channels, out_channels,
                          bias=False, weight_initializer='glorot')
        if add_self_loops:
            self.self_lin = Linear(in_channels, out_channels,
                                   bias=False, weight_initializer='glorot')
        self.att_src = Linear(out_channels, 1,
                              bias=False, weight_initializer='glorot')
        self.att_dst = Linear(out_channels, 1,
                              bias=False, weight_initializer='glorot')

    def forward(self, x: Tensor, rowptr: Tensor, indices: Tensor, **kwargs) -> Tensor:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=rowptr.device)
        out = self.propagate(edge_index=edge_index, x=x, rowptr=rowptr, indices=indices)
        if self.add_self_loops:
            out = out + self.self_lin(x)
        return out

    def propagate(self, edge_index, size=None, **kwargs):
        kwargs['x'] = self.aggr_lin(kwargs['x'])

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        # in newer PyG,
        # __check_input__ -> _check_input()
        # __collect__ -> _collect()
        # __fused_user_args__ -> _fuser_user_args
        if size is None:
            size = [2, kwargs['indices'].shape[0]]
        # size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._fused_user_args, edge_index, size, kwargs)
        # use from packaging.version import parse as parse_version as by default 2.4 > 2.14 which is wrong
        pyg_version = [int(i) for i in torch_geometric.__version__.split(".")]
        col_fn = self.inspector.distribute if pyg_version[1] <= 4 else self.inspector.collect_param_data
        msg_aggr_kwargs = col_fn("message_and_aggregate", coll_dict)
        for hook in self._message_and_aggregate_forward_pre_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs))
            if res is not None:
                edge_index, msg_aggr_kwargs = res
        out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
        for hook in self._message_and_aggregate_forward_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs), out)
            if res is not None:
                out = res
        update_kwargs = col_fn("update", coll_dict)
        out = self.update(out, **update_kwargs)
        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, size, kwargs), out)
            if res is not None:
                out = res
        return out

    def message_and_aggregate(self, edge_index, x, index, dim_size, rowptr, indices):
        # fused computation of message and aggregate steps with the custom rspmm cuda kernel
        # speed up computation by several times
        from models.triton_kernels.gat import ConvGATAggr
        from models.triton_kernels.gat3d import ConvGAT3dAggr
        num_node = x.shape[0]

        if x.ndim == 3:
            num_nodes, num_feat, hid_dim = x.shape
            x = x.view(num_nodes, num_feat * hid_dim)
            out = ConvGAT3dAggr.apply(x, rowptr, indices,
                                      self.att_src.weight / num_feat, self.att_dst.weight / num_feat,
                                      num_node, 0)
            return out.view(num_nodes, num_feat, hid_dim)
        elif x.ndim == 2:
            return ConvGATAggr.apply(x, rowptr, indices,
                                     self.att_src.weight, self.att_dst.weight,
                                     num_node, 0)
        else:
            raise ValueError(f'MeanGNN x.ndim={x.ndim} not supported')
