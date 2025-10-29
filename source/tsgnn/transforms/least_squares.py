import copy
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.nn.conv import SimpleConv
from torch_scatter import scatter
from torch import Tensor
from typing import List, Dict
import pickle
import os
from typing import Tuple
from sklearn.model_selection import KFold

from helpers.constants import ROOT_DIR


def solve_ls_with_bias_cv(x: Tensor, y: Tensor, lambdas: List[float] = None, k_folds: int = 10) -> Tuple[Tensor, Tensor]:
    if lambdas is None:
        lambdas = [1e-15, 1e-8, 1e-6, 1e-4, 1e-2]  # log scale

    num_nodes, num_feat = x.shape
    num_labels = y.shape[1]

    # Add bias term
    x_with_bias = torch.cat([x, torch.ones(num_nodes, 1)], dim=1)

    best_lambda = None
    best_loss = float('inf')
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    for lam in lambdas:
        fold_losses = []
        for train_idx, val_idx in kf.split(x_with_bias):
            x_train, x_val = x_with_bias[train_idx], x_with_bias[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Augment system for ridge regularization
            x_aug = torch.cat([x_train, lam * torch.eye(num_feat + 1)], dim=0)
            y_aug = torch.cat([y_train, torch.zeros((num_feat + 1, num_labels))], dim=0)

            w = torch.linalg.lstsq(x_aug, y_aug, driver="gelss")[0]
            y_pred = x_val @ w
            fold_losses.append(torch.mean((y_pred - y_val) ** 2).item())

        avg_loss = sum(fold_losses) / len(fold_losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_lambda = lam

    # Retrain on full data with best lambda
    x_aug_full = torch.cat([x_with_bias, best_lambda * torch.eye(num_feat + 1)], dim=0)
    y_aug_full = torch.cat([y, torch.zeros((num_feat + 1, num_labels))], dim=0)
    best_weights = torch.linalg.lstsq(x_aug_full, y_aug_full, driver="gelss")[0]
    return best_weights[:-1], best_weights[-1]


class LeastSquaresTransform(BaseTransform):
    def __init__(self, ls_num_layers: int, dataset_name: str, num_feat: int, seed: int):
        """
        Computes the least squares solution with L2 regularization
        between node features (data.x) and labels (data.y),
        using mean aggregation based on the graph structure.

        Args:
            ls_num_layers (int): Number of layers to apply mean aggregation before solving least squares.
        """
        super().__init__()
        self.ls_num_layers = ls_num_layers
        self.mean_aggr = SimpleConv(aggr='mean')
        self.cache_path = os.path.join(ROOT_DIR, 'datasets', dataset_name, f'{num_feat}feat', f'seed{seed}')

    def _save_cache(self, xy_conversions: Dict[str, Tensor], new_key_list: List[str]):
        if len(new_key_list) == 0:
            return
        os.makedirs(self.cache_path, exist_ok=True)

        # Save each key-value pair separately
        for key in new_key_list:
            with open(os.path.join(self.cache_path, f"{key}.pkl"), "wb") as f:
                pickle.dump(xy_conversions[key], f)

    def _load_cache(self) -> Dict[str, Tensor]:
        if not os.path.exists(self.cache_path):
            return {}

        # Load all ".pkl" file names in the self.cache_path folder (excluding "keys.pkl")
        key_files = os.listdir(self.cache_path)
        keys = {file[:-4] for file in key_files if file.endswith(".pkl")}

        # Load each transformation separately
        xy_conversions = {}
        for key in keys:
            key_path = os.path.join(self.cache_path, f"{key}.pkl")
            if os.path.exists(key_path):
                with open(key_path, "rb") as f:
                    xy_conversions[key] = pickle.load(f)

        return xy_conversions

    def __call__(self, data):
        xy_conversions = self._load_cache()
        xy_keys = list(xy_conversions.keys())

        # pool calc rescale
        pool_x = data.x.mean(dim=1, keepdim=True)
        pool_y = data.y_mat.mean(dim=1, keepdim=True)

        if 'pool2y_w' not in xy_keys:
            xy_conversions[f'pool2y_w'], xy_conversions[f'pool2y_b'] = \
                solve_ls_with_bias_cv(x=pool_x[data.train_mask], y=data.y_mat[data.train_mask])

        if 'pool2x_w' not in xy_keys:
            xy_conversions[f'pool2x_w'], xy_conversions[f'pool2x_b'] = \
                solve_ls_with_bias_cv(x=pool_y[data.train_mask], y=data.x[data.train_mask])

        if self.ls_num_layers >= 0:
            for is_hetero in [False, True]:
                x_l = copy.deepcopy(data.x)
                str_suffix = '_hetero' if is_hetero else ''
                for num_layer in range(self.ls_num_layers + 1):
                    if num_layer == 0 and is_hetero:
                        continue

                    x_l_train = x_l[data.train_mask]
                    y_train = data.y_mat[data.train_mask]
                    name = f'ls2y_{num_layer}{str_suffix}'
                    if name + '_w' not in xy_keys:
                        xy_conversions[name + '_w'], xy_conversions[name + '_b'] = \
                            solve_ls_with_bias_cv(x=x_l_train, y=y_train)

                    name = f'ls2x_{num_layer}{str_suffix}'
                    if name + '_w' not in xy_keys:
                        xy_conversions[name + '_w'], xy_conversions[name + '_b'] = \
                            solve_ls_with_bias_cv(x=y_train, y=x_l_train)

                    # Apply mean aggregation to features
                    if is_hetero:
                        x_l = (x_l - self.mean_aggr(x_l, edge_index=data.edge_index))
                    else:
                        x_l = self.mean_aggr(x_l, edge_index=data.edge_index)

        self._save_cache(xy_conversions, new_key_list=list(set(xy_conversions.keys()) - set(xy_keys)))
        setattr(data, 'xy_conversions', xy_conversions)
        return data
