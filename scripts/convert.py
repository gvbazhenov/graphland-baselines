import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)

import yaml
import json
import numpy as np
import scipy.sparse as sp
import pandas as pd
import pickle
import torch
import dgl

from itertools import product
from pathlib import Path
from typing import Any
from sklearn.impute import SimpleImputer


GRAPHLAND_DATASETS = [
    'avazu-ctr',
    'hm-categories',
    'hm-prices',
    'tolokers-2',
    'artnet-exp',
    'artnet-views',
    'pokec-regions',
    'twitch-views',
    'city-reviews',
    'city-roads-M',
    'city-roads-L',
    'web-fraud',
    'web-traffic',
    'web-topics',
]
GRAPHLAND_CLASSIFICATION_DATASETS = [
    'hm-categories',
    'tolokers-2',
    'artnet-exp',
    'pokec-regions',
    'city-reviews',
    'web-fraud',
    'web-topics',
]
GRAPHLAND_DATA_ROOT = Path('./datasets')
GRAPHLAND_DATA_SPLITS = ['RH', 'RL', 'TH', 'THI']


def _load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def _save_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    with open(path, 'w') as f:
        json.dump(data, f)


def _save_ndarray(path: str | Path, data: np.ndarray) -> None:
    np.save(path, data)


def _save_pickle(path: str | Path, data: Any) -> None:
    path = Path(path)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


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
        df_features[cat_features_names].values.astype(np.int32)
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


def _load_nfa_features(dataset_name: str, split_name: str) -> np.ndarray:
    path = GRAPHLAND_DATA_ROOT / dataset_name
    setting_name = 'inductive' if split_name == 'THI' else 'transductive'
    df_nfa_features = pd.read_csv(path / f'nfa_{setting_name}_features.csv', index_col=0)
    nfa_features = df_nfa_features.values.astype(np.float32)
    return nfa_features


def _convert_for_gbdt() -> None:
    print('>>> GBDT\n')
    BASELINE_DATA_ROOT = Path('./source/gbdt/data')
    BASELINE_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    for dataset_name, split_name, use_nfa in product(
        GRAPHLAND_DATASETS,
        GRAPHLAND_DATA_SPLITS,
        [False, True],
    ):
        if split_name == 'THI' and not use_nfa:
            continue

        print(f'{dataset_name=}')
        print(f'{split_name=}')
        print(f'{use_nfa=}')

        try:
            data = _load_graphland_dataset(dataset_name, split_name[:2])
        except:
            print('> skip\n')
            continue

        num_features: None | np.ndarray = data['num_features']
        cat_features: None | np.ndarray = data['cat_features']
        frac_features: None | np.ndarray = data['frac_features']
        targets: np.ndarray = data['targets']
        masks: dict[str, np.ndarray] = data['masks']
        info: dict = data['info']

        baseline_dataset_info = {
            'task_type': 'regression' if info['task'] == 'regression' 
                else 'binclass' if info['task'] == 'binary_classification' 
                else 'multiclass',
            'name': dataset_name,
        }

        if use_nfa:
            try:
                nfa_features = _load_nfa_features(dataset_name, split_name)
            except:
                continue

            num_features = (
                np.concatenate([num_features, nfa_features], axis=1) 
                if num_features is not None else nfa_features
            )

        baseline_dataset_path = BASELINE_DATA_ROOT / '-'.join(
            [dataset_name, split_name] + (['nfa'] if use_nfa else [])
        )
        baseline_dataset_path.mkdir(parents=True, exist_ok=True)
        _save_json(baseline_dataset_path / 'info.json', baseline_dataset_info)

        mask_labeled = ~np.isnan(targets)
        for part_name in ['train', 'val', 'test']:
            mask = masks[part_name] & mask_labeled

            if num_features is not None:
                _save_ndarray(
                    baseline_dataset_path / f'X_num_{part_name}.npy',
                    num_features[mask],
                )
            
            if cat_features is not None:
                _save_ndarray(
                    baseline_dataset_path / f'X_cat_{part_name}.npy',
                    cat_features[mask].astype(np.str_),
                )

            if frac_features is not None:
                _save_ndarray(
                    baseline_dataset_path / f'X_bin_{part_name}.npy',
                    frac_features[mask],
                )

            targets_masked = targets[mask]
            targets_casted = (
                targets_masked.astype(np.int32) if info['task'] != 'regression' 
                else targets_masked.astype(np.float32)
            )
            _save_ndarray(baseline_dataset_path / f'Y_{part_name}.npy', targets_casted)

        open(baseline_dataset_path / 'READY', 'a').close()
        print('> done\n')


def _convert_for_anygraph() -> None:
    print('>>> AnyGraph\n')
    BASELINE_DATA_ROOT = Path('./source/anygraph/datasets/graphland')
    BASELINE_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    for dataset_name, split_name in product(
        GRAPHLAND_CLASSIFICATION_DATASETS,
        GRAPHLAND_DATA_SPLITS[:-1],
    ):
        print(f'{dataset_name=}')
        print(f'{split_name=}')

        try:
            data = _load_graphland_dataset(dataset_name, split_name[:2])
        except:
            print('> skip\n')
            continue

        num_nodes = len(data['num_features'])
        num_classes = pd.Series(data['targets']).nunique()
        mask_labeled = ~np.isnan(data['targets'])

        features = np.concatenate(
            [data[key] for key in ['num_features', 'cat_features', 'frac_features']],
            axis=1
        )
        gfm_features = SimpleImputer(strategy='most_frequent').fit_transform(features)
        gfm_features = np.concatenate(
            [
                gfm_features,
                np.zeros(
                    shape=(num_classes, gfm_features.shape[1]), dtype=gfm_features.dtype
                ),
            ],
            axis=0,
        )

        indices_test = np.where(data['masks']['test'] & mask_labeled)[0]
        gfm_labels_data = (
            np.ones(len(indices_test)),
            (indices_test, data['targets'][indices_test]),
        )
        gfm_labels = sp.coo_matrix(gfm_labels_data, shape=(num_nodes, num_classes))

        indices_train = np.where(data['masks']['train'] & mask_labeled)[0]
        source_idx = np.concatenate(
            [
                data['edges'][:, 0],
                indices_train,
                data['targets'][indices_train] + num_nodes,
            ]
        )
        target_idx = np.concatenate(
            [
                data['edges'][:, 1],
                data['targets'][indices_train] + num_nodes,
                indices_train,
            ]
        )
        gfm_adjacency_data = (np.ones(len(source_idx)), (source_idx, target_idx))
        gfm_adjacency = sp.coo_matrix(
            gfm_adjacency_data,
            shape=(num_nodes + num_classes, num_nodes + num_classes),
        )

        gfm_dataset_path = BASELINE_DATA_ROOT / '-'.join([dataset_name, split_name])
        gfm_dataset_path.mkdir(parents=True, exist_ok=True)

        _save_pickle(gfm_dataset_path / 'feats.pkl', gfm_features)
        _save_pickle(gfm_dataset_path / 'trn_mat.pkl', gfm_adjacency)
        _save_pickle(gfm_dataset_path / 'tst_mat.pkl', gfm_labels)

        print('> done\n')


def _convert_for_opengraph() -> None:
    print('>>> OpenGraph\n')
    BASELINE_DATA_ROOT = Path('./source/opengraph/datasets/graphland')
    BASELINE_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    for dataset_name, split_name in product(
        GRAPHLAND_CLASSIFICATION_DATASETS,
        GRAPHLAND_DATA_SPLITS[:-1],
    ):
        print(f'{dataset_name=}')
        print(f'{split_name=}')

        try:
            data = _load_graphland_dataset(dataset_name, split_name[:2])
        except:
            print('> skip\n')
            continue

        num_nodes = len(data['num_features'])
        num_classes = pd.Series(data['targets']).nunique()
        mask_labeled = ~np.isnan(data['targets'])

        gfm_mask = {
            (part if part != 'val' else 'valid'): data['masks'][part] & mask_labeled
            for part in ['train', 'val', 'test']
        }
        indices_train = np.where(gfm_mask['train'])[0]

        features = np.concatenate(
            [data[key] for key in ['num_features', 'cat_features', 'frac_features']],
            axis=1
        )
        gfm_features = SimpleImputer(strategy='most_frequent').fit_transform(features)
        gfm_labels = np.nan_to_num(data['targets'], nan=0.0)

        source_idx = np.concatenate(
            [
                data['edges'][:, 0],
                indices_train,
                data['targets'][indices_train] + num_nodes,
            ]
        )
        target_idx = np.concatenate(
            [
                data['edges'][:, 1],
                data['targets'][indices_train] + num_nodes,
                indices_train,
            ]
        )
        gfm_adjacency_data = (np.ones(len(source_idx)), (source_idx, target_idx))
        gfm_adjacency = sp.coo_matrix(
            gfm_adjacency_data,
            shape=(num_nodes + num_classes, num_nodes + num_classes),
        )

        gfm_dataset_path = BASELINE_DATA_ROOT / '-'.join([dataset_name, split_name])
        gfm_dataset_path.mkdir(parents=True, exist_ok=True)

        _save_pickle(gfm_dataset_path / 'feats.pkl', gfm_features)
        _save_pickle(gfm_dataset_path / 'adj_-1.pkl', gfm_adjacency)
        _save_pickle(gfm_dataset_path / 'label.pkl', gfm_labels)
        _save_pickle(gfm_dataset_path / 'mask_-1.pkl', gfm_mask)

        print('> done\n')


def _convert_for_tsgnn() -> None:
    print('>>> TS-GNN\n')
    BASELINE_DATA_ROOT = Path('./source/tsgnn/datasets/graphland')
    BASELINE_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    for dataset_name, split_name in product(
        GRAPHLAND_CLASSIFICATION_DATASETS,
        GRAPHLAND_DATA_SPLITS[:-1],
    ):
        print(f'{dataset_name=}')
        print(f'{split_name=}')

        try:
            data = _load_graphland_dataset(dataset_name, split_name[:2])
        except:
            print('> skip\n')
            continue

        features = np.concatenate(
            [data[key] for key in ['num_features', 'cat_features', 'frac_features']],
            axis=1
        )
        gfm_features = SimpleImputer(strategy='most_frequent').fit_transform(features)
        gfm_labels = np.nan_to_num(data['targets'], nan=0.0)

        gfm_data = dict()
        gfm_data['features'] = gfm_features
        gfm_data['labels'] = gfm_labels
        gfm_data['edge_index'] = data['edges'].T

        mask_labeled = ~np.isnan(data['targets'])
        for part in ['train', 'val', 'test']:
            gfm_data[f'{part}_mask'] = data['masks'][part] & mask_labeled

        gfm_dataset_path = BASELINE_DATA_ROOT / '-'.join([dataset_name, split_name])
        gfm_dataset_path.mkdir(parents=True, exist_ok=True)
        np.savez(gfm_dataset_path / 'data.npz', **gfm_data)

        print('> done\n')


def _convert_for_gcope() -> None:
    print('>>> GCOPE\n')
    BASELINE_DATA_ROOT = Path('./source/gcope/datasets/graphland')
    BASELINE_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    for dataset_name, split_name in product(
        GRAPHLAND_CLASSIFICATION_DATASETS,
        GRAPHLAND_DATA_SPLITS,
    ):
        print(f'{dataset_name=}')
        print(f'{split_name=}')

        try:
            data = _load_graphland_dataset(dataset_name, split_name[:2])
        except:
            print('> skip\n')
            continue

        features = np.concatenate(
            [data[key] for key in ['num_features', 'cat_features', 'frac_features']],
            axis=1
        )
        gfm_features = SimpleImputer(strategy='most_frequent').fit_transform(features)
        gfm_labels = np.nan_to_num(data['targets'], nan=0.0)

        gfm_data = dict()
        gfm_data['features'] = gfm_features
        gfm_data['labels'] = gfm_labels

        graph = dgl.graph(data=(data['edges'][:, 0], data['edges'][:, 1]))
        mask_seen = (split_name != 'THI')

        mask_labeled = ~np.isnan(data['targets'])
        for part in ['train', 'val', 'test']:
            mask_seen |= data['masks'][part]
            node_indices = np.where(mask_seen)[0]

            subgraph = dgl.node_subgraph(graph, node_indices, relabel_nodes=False, store_ids=False)
            edge_index = torch.stack(subgraph.edges()).numpy()

            gfm_data[f'{part}_mask'] = data['masks'][part] & mask_labeled
            gfm_data[f'{part}_edge_index'] = edge_index

        gfm_dataset_path = BASELINE_DATA_ROOT / '-'.join([dataset_name, split_name])
        gfm_dataset_path.mkdir(parents=True, exist_ok=True)
        np.savez(gfm_dataset_path / 'data.npz', **gfm_data)

        print('> done\n')


if __name__ == '__main__':
    assert Path.cwd().joinpath('pixi.lock').exists(), \
    'The scripts must be run from the project root!'

    _convert_for_gbdt()
    _convert_for_anygraph()
    _convert_for_opengraph()
    _convert_for_tsgnn()
    _convert_for_gcope()
