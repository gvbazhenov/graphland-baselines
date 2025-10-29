import torch
import random
import numpy as np
import os
from torch_geometric.utils.sparse import index2ptr
from torch_geometric.utils import index_sort
from typing import Optional
from torch import Tensor

from helpers.metrics import LossAndMetric
from helpers.constants import DECIMAL


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False


def coo_to_csr(row, col, num_nodes=None):
    if num_nodes is None:
        num_nodes = int(row.max()) + 1
    row, perm = index_sort(row, max_value=num_nodes)
    col = col[perm]
    rowptr = index2ptr(row, num_nodes)
    return rowptr, col


def str_print(train_test_name: Optional[str] = None, single_dataset_name: Optional[str] = None,
              seed: Optional[int] = None, losses_n_metric: Optional[LossAndMetric] = None,
              metric_mean: Optional[Tensor] = None, metric_std: Optional[Tensor] = None,
              suffix: Optional[str] = None) -> str:
    path = f"Results"
    for folder_name in [train_test_name, single_dataset_name, seed]:
        if folder_name is not None:
            path = os.path.join(path, str(folder_name))
    path += '/'
    if losses_n_metric is not None:
        for name in losses_n_metric._fields:
            path += f"{name}={round(getattr(losses_n_metric, name), DECIMAL)},"
        path = path[:-1]
    elif metric_mean is not None and metric_std is not None:
        path += ' '
        for split_name, mean, std in zip(['train', 'val', 'test'], metric_mean, metric_std):
            path += f'{split_name}={round(mean.item() * 100, DECIMAL)}+-{round(std.item() * 100, DECIMAL)},'
        path = path[:-1]
    if suffix is not None:
        path = os.path.join(path, suffix)
    return path


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute accuracy between predicted logits/probs and ground truth labels.

    Args:
        preds (torch.Tensor): shape [N, C] for logits or [N] for class indices
        targets (torch.Tensor): shape [N] with class indices

    Returns:
        float: accuracy in [0, 1]
    """
    if preds.dim() == 2:
        pred_labels = preds.argmax(dim=1)
    elif preds.dim() == 1:
        pred_labels = preds
    else:
        raise ValueError(f"Invalid shape for preds: {preds.shape}")

    correct = (pred_labels == targets).sum().item()
    total = targets.numel()

    return correct / total
