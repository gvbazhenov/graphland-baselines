from typing import Tuple
import copy
from torch import Tensor
from sklearn.model_selection import train_test_split
import math
from torch_geometric.data import Data
import torch
import numpy as np

from transforms.least_squares import LeastSquaresTransform


# GraphAny splits: https://github.com/DeepGraphLearning/GraphAny/blob/main/graphany/data.py#L59
def graphany_mask_splits(n_nodes, labels, num_train_nodes, seed=42):
    assert labels.ndim == 1, "y n_dim must equal 1"
    label_idx = np.arange(n_nodes)
    test_rate_in_labeled_nodes = (len(labels) - num_train_nodes) / len(labels)
    train_idx, test_and_valid_idx = train_test_split(
        label_idx,
        test_size=test_rate_in_labeled_nodes,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    valid_idx, test_idx = train_test_split(
        test_and_valid_idx,
        test_size=0.5,
        random_state=seed,
        shuffle=True,
        stratify=labels[test_and_valid_idx],
    )
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[valid_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def get_masks(data: Data, seed: int) -> Tuple[Tensor, Tensor, Tensor]:
    if hasattr(data, "train_mask") and hasattr(data, "val_mask") and hasattr(data, "test_mask"):
        train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    else:
        n_nodes = data.x.shape[0]
        label = data.y
        num_class = label.max().item() + 1
        train_mask, val_mask, test_mask = graphany_mask_splits(
            n_nodes, label, num_train_nodes=20 * num_class, seed=seed
        )
    return train_mask, val_mask, test_mask


def split_data_per_fold(data: Data, seed: int, ls_num_layers: int, dataset_name: str) -> Data:
    train_mask, val_mask, test_mask = get_masks(data=data, seed=seed)
    # create a data split using the masks
    if train_mask.ndim == 2:
        # ! Multiple splits
        # Modified: Use the ${seed} split if not specified!
        split_index = seed
        # Avoid invalid split index
        split_index = (split_index % train_mask.ndim)
        train_mask = train_mask[:, split_index].squeeze()
        val_mask = val_mask[:, split_index].squeeze()
        if test_mask.ndim == 2:
            test_mask = test_mask[:, split_index].squeeze()
    dataset_copy = copy.deepcopy(data)
    setattr(dataset_copy, 'train_mask', train_mask)
    setattr(dataset_copy, 'val_mask', val_mask)
    setattr(dataset_copy, 'test_mask', test_mask)
    ls_pre_transform = LeastSquaresTransform(ls_num_layers=ls_num_layers, dataset_name=dataset_name,
                                             num_feat=data.x.shape[1], seed=seed)
    dataset_copy = ls_pre_transform(dataset_copy)
    return dataset_copy
