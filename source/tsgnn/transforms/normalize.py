from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch import Tensor
import torch

EPS = 1E-8


def standardize(x: Tensor) -> Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + EPS)


def normalize(x: Tensor) -> Tensor:
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + EPS)


class NormalizeTransform(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data: Data) -> Data:
        data.x = normalize(x=data.x)
        return data
