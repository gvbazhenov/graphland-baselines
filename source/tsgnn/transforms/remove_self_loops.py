import torch_geometric.transforms as T
from torch_geometric.data import Data


class RemoveSelfLoops(T.BaseTransform):
    def __call__(self, data: Data) -> Data:
        edge_index = data.edge_index
        mask = edge_index[0] != edge_index[1]  # Keep only edges where src != dst
        data.edge_index = edge_index[:, mask]
        return data