import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)

import yaml
import numpy as np
import pandas as pd
import torch
import dgl

from pathlib import Path
from typing import Literal
from sklearn.preprocessing import OneHotEncoder


GRAPHLAND_DATASETS = [
    # 'avazu-ctr',
    # 'hm-categories',
    # 'hm-prices',
    'tolokers-2',
    # 'artnet-exp',
    # 'artnet-views',
    # 'pokec-regions',
    # 'twitch-views',
    'city-reviews',
    # 'city-roads-M',
    # 'city-roads-L',
    # 'web-fraud',
    # 'web-traffic',
    # 'web-topics',
]
GRAPHLAND_DATA_ROOT = Path('./datasets')


def _save_yaml(path: str | Path, data: dict) -> None:
    path = Path(path)
    with open(path, 'w') as f:
        data = yaml.dump(data, f)


def _load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def _load_graphland_dataset(dataset_name: str, split_name: str) -> dict:
    path = GRAPHLAND_DATA_ROOT / dataset_name

    df_features = pd.read_csv(path / 'features.csv', index_col=0)
    df_targets = pd.read_csv(path / 'targets.csv', index_col=0)
    df_data_split = pd.read_csv(path / f'split_masks_{split_name}.csv')
    df_edges = pd.read_csv(path / 'edgelist.csv')
    info = _load_yaml(path / 'info.yaml')

    num_features_names = [
        feature_name for feature_name in info['numerical_features_names']
        if feature_name not in info['fraction_features_names']
    ]
    num_features = (
        df_features[num_features_names].values.astype(np.float32)
        if num_features_names else None
    )

    cat_features_names = info['categorical_features_names']
    cat_features = (
        df_features[cat_features_names].values.astype(np.int32).astype(np.str_)
        if cat_features_names else None
    )

    frac_features_names = info['fraction_features_names']
    frac_features = (
        df_features[frac_features_names].values.astype(np.float32)
        if frac_features_names else None
    )

    targets = df_targets.values.reshape(-1)
    edges = df_edges.values[:, :2]

    masks = {
        part: df_data_split[part].values
        for part in df_data_split.columns
    }

    return {
        'num_features': num_features,
        'cat_features': cat_features,
        'frac_features': frac_features,
        'targets': targets,
        'edges': edges,
        'masks': masks,
        'info': info,
    }


def _nfa_reduce(
    g: dgl.DGLGraph,
    x: np.ndarray,
    *,
    mode: Literal['mean', 'max', 'min'],
) -> np.ndarray:
    x = torch.tensor(x)

    not_nan_mask = ~torch.isnan(x)
    not_nan_size = dgl.ops.copy_u_sum(g, not_nan_mask.float())

    neutral_value = {
        'mean': 0.0,
        'max': -torch.inf,
        'min': +torch.inf,
    }[mode]

    operation = {
        'mean': dgl.ops.copy_u_sum,
        'max': dgl.ops.copy_u_max,
        'min': dgl.ops.copy_u_min,
    }[mode]

    denominator = {
        'mean': not_nan_size,
        'max': (not_nan_size > 0.0).float(),
        'min': (not_nan_size > 0.0).float(),
    }[mode]

    x_imputed = torch.where(not_nan_mask, x, neutral_value)
    numerator = operation(g, x_imputed)
    x_reduced = torch.where(denominator != 0.0, numerator / denominator, torch.nan)
    return x_reduced.numpy()


def _make_transductive_nfa() -> None:
    print('>>> transductive NFA\n')

    for dataset_name in GRAPHLAND_DATASETS:
        print(f'{dataset_name=}')

        data = _load_graphland_dataset(dataset_name, split_name='RL')
        edges = torch.from_numpy(data['edges']).T
        graph = dgl.graph(data=(edges[0], edges[1]))
        graph = dgl.to_bidirected(graph)
        graph = dgl.remove_self_loop(graph)

        nfa_features = []

        if data['num_features'] is not None:
            for mode in ['mean', 'max', 'min']:
                _nfa_features = _nfa_reduce(graph, data['num_features'], mode=mode)
                nfa_features.append(_nfa_features)

        if data['cat_features'] is not None:
            _ohe = OneHotEncoder(
                drop='if_binary',
                sparse_output=False,
                dtype=np.float32,
            )
            _ohe_features = _ohe.fit_transform(data['cat_features'])
            for mode in ['mean']:
                _nfa_features = _nfa_reduce(graph, _ohe_features, mode=mode)
                nfa_features.append(_nfa_features)
            del _ohe

        if data['frac_features'] is not None:
            for mode in ['mean', 'max', 'min']:
                _nfa_features = _nfa_reduce(graph, data['num_features'], mode=mode)
                nfa_features.append(_nfa_features)

        nfa_features = np.concatenate(nfa_features, axis=1)
        nfa_features_path = GRAPHLAND_DATA_ROOT / dataset_name / 'nfa_transductive_features.csv'
        df_nfa_features = pd.DataFrame(
            data=nfa_features,
            columns=[f'feature_{idx}' for idx in range(nfa_features.shape[1])],
        )
        df_nfa_features.to_csv(nfa_features_path)
        print('> done\n')


def _make_inductive_nfa() -> None:
    print('>>> inductive NFA\n')

    for dataset_name in GRAPHLAND_DATASETS:
        print(f'{dataset_name=}')

        try:
            data = _load_graphland_dataset(dataset_name, split_name='TH')
        except:
            print('> skip\n')
            continue

        edges = torch.from_numpy(data['edges']).T
        graph = dgl.graph(data=(edges[0], edges[1]))
        graph = dgl.to_bidirected(graph)
        graph = dgl.remove_self_loop(graph)

        if data['cat_features'] is not None:
            _ohe = OneHotEncoder(
                drop='if_binary',
                sparse_output=False,
                dtype=np.float32,
                handle_unknown='ignore',
            )
            _ohe_mask = data['masks']['train']
            _ohe.fit(data['cat_features'][_ohe_mask])
            _ohe_features = _ohe.transform(data['cat_features'])
            del _ohe

        mask_seen = False
        nfa_features = dict()

        for part in ['train', 'val', 'test']:
            mask_part = data['masks'][part]
            mask_seen = mask_seen | mask_part
            subgraph = dgl.subgraph.node_subgraph(
                graph,
                torch.from_numpy(mask_seen),
                relabel_nodes=False,
                store_ids=False,
            )

            nfa_features_part = []
            if data['num_features'] is not None:
                for mode in ['mean', 'max', 'min']:
                    _nfa_features = _nfa_reduce(subgraph, data['num_features'], mode=mode)
                    nfa_features_part.append(_nfa_features)

            if data['cat_features'] is not None:
                for mode in ['mean']:
                    _nfa_features = _nfa_reduce(graph, _ohe_features, mode=mode)
                    nfa_features_part.append(_nfa_features)

            if data['frac_features'] is not None:
                for mode in ['mean', 'max', 'min']:
                    _nfa_features = _nfa_reduce(graph, data['num_features'], mode=mode)
                    nfa_features_part.append(_nfa_features)

            nfa_features_part = np.concatenate(nfa_features_part, axis=1)[mask_part]
            nfa_features[part] = nfa_features_part

        nfa_features_joined = np.zeros(
            shape=(graph.num_nodes(), nfa_features_part.shape[1]),
        )
        for part in ['train', 'val', 'test']:
            mask_part = data['masks'][part]
            nfa_features_joined[mask_part] = nfa_features[part]

        nfa_features_path = GRAPHLAND_DATA_ROOT / dataset_name / 'nfa_inductive_features.csv'
        df_nfa_features = pd.DataFrame(
            data=nfa_features_joined,
            columns=[f'feature_{idx}' for idx in range(nfa_features_part.shape[1])],
        )
        df_nfa_features.to_csv(nfa_features_path)
        print('> done\n')


if __name__ == '__main__':
    assert Path.cwd().joinpath('pixi.lock').exists(), \
        'The scripts must be run from the project root!'

    _make_transductive_nfa()
    _make_inductive_nfa()
