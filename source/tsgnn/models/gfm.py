import math

from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch.nn import Module, Dropout, ModuleList
import torch.nn.functional as F
from typing import NamedTuple, Tuple, Dict, Optional
from helpers.gnn_type import GNNType
from models.equiv_layer import EquivLayer
import torch
import copy


class GFMArgs(NamedTuple):
    hid_channel: int
    num_layers: int
    ls_num_layers: int
    gnn_type: GNNType
    lp_ratio: float


@torch.no_grad()  # CPU function
def sample_k_nodes_per_label(label, visible_nodes, lp_ratio, num_class):
    train_size = visible_nodes.sum().item()
    k = math.floor(train_size * lp_ratio / num_class)
    node_idx_by_cls = [
        (label[visible_nodes] == lbl).nonzero().view(-1) for lbl in range(num_class)
    ]
    rnd_node_idx_by_cls = [
        label_indices[torch.randperm(len(label_indices))]
        for label_indices in node_idx_by_cls
    ]
    lp_indices = [
        label_indices[:k]
        for label_indices in rnd_node_idx_by_cls
    ]
    gt_indices = [
        label_indices[k:]
        for label_indices in rnd_node_idx_by_cls
    ]
    lp_mask = torch.zeros_like(visible_nodes, dtype=torch.bool).cpu()
    lp_mask[torch.cat(lp_indices)] = True
    gt_mask = torch.zeros_like(visible_nodes, dtype=torch.bool).cpu()
    gt_mask[torch.cat(gt_indices)] = True
    return lp_mask, gt_mask


class GFM(Module):
    def __init__(self, gfm_args: GFMArgs):
        super(GFM, self).__init__()
        # ls xy_conversions
        self.lp_ratio = gfm_args.lp_ratio

        # encoder
        triton_on = gfm_args.gnn_type.uses_triton()
        dim_list = [1] + [gfm_args.hid_channel] * (gfm_args.num_layers - 1) + [1]
        self.equiv_layers = [EquivLayer(in_channel=in_channel, out_channel=out_channel,
                                        ls_num_layers=gfm_args.ls_num_layers, gnn_type=gfm_args.gnn_type,
                                        triton_on=triton_on)
                             for layer, (in_channel, out_channel) in enumerate(zip(dim_list[:-1], dim_list[1:]))]
        self.num_layers = gfm_args.num_layers
        self.equiv_layers = ModuleList(self.equiv_layers)

    def forward(self, x: Tensor, train_y: Tensor, xy_conversions: Dict[str, Tensor], is_batch: bool, device,
                edge_index: Optional[Adj], rowptr: OptTensor = None,
                indices: OptTensor = None) -> Tuple[Tensor, OptTensor]:
        # x Shape: [num_edges, in_dim]
        # out Shape: [num_nodes, num_classes]
        x = copy.deepcopy(x)
        train_y = copy.deepcopy(train_y)

        with torch.no_grad():
            # sampling prep
            train_mask = train_y.sum(dim=1).to(torch.bool)
            if is_batch:
                lp_mask, gt_mask =\
                    sample_k_nodes_per_label(label=train_y.argmax(dim=1), visible_nodes=train_mask,
                                             lp_ratio=self.lp_ratio, num_class=train_y.shape[1])
            else:
                lp_mask = train_mask
                gt_mask = None

        # x encoding
        x = x.unsqueeze(dim=-1).to(device=device)
        train_y[~lp_mask] = 0
        y = train_y.unsqueeze(dim=-1).to(device=device)
        if isinstance(edge_index, Tensor):
            edge_index = edge_index.to(device=device)
        if isinstance(rowptr, Tensor):
            rowptr = rowptr.to(device=device)
        if isinstance(indices, Tensor):
            indices = indices.to(device=device)
        xy_conversions = {key: value.to(device=device) if isinstance(value, Tensor) else value
                          for key, value in xy_conversions.items()}
        for idx, layer in enumerate(self.equiv_layers):
            x, y = layer(x=x, y=y, xy_conversions=xy_conversions, device=device,
                         edge_index=edge_index, rowptr=rowptr, indices=indices)

            if idx != self.num_layers - 1:
                x = F.gelu(x)
                y = F.gelu(y)
        return y.squeeze(dim=-1), gt_mask
