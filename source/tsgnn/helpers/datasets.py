import os
import os.path as osp
from enum import Enum, auto
import torch
from typing import Optional
from torch_geometric.data import Data
import torch_geometric.transforms as T
import ssl
import urllib
import sys
from torch_geometric.datasets import (HeterophilousGraphDataset, Planetoid, WikipediaNetwork, WebKB, CitationFull,
                                      AttributedGraphDataset, WikiCS, Coauthor, Airports, Actor, Amazon,
                                      LastFMAsia, DeezerEurope)
from torch_geometric.data import Data
import numpy as np
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset

from helpers.constants import ROOT_DIR
from transforms.remove_self_loops import RemoveSelfLoops
from transforms.pca import PCATransform
from transforms.normalize import normalize, NormalizeTransform


def download_url(url: str, folder: str, log: bool = True, filename=None):
    r"""Modified from torch_geometric.data.download_url

    Downloads the content of an URL to a specific folder.

    Args:
        url (str): The URL.
        folder (str): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    if filename is None:
        filename = url.rpartition("/")[2]
        filename = filename if filename[0] == "?" else filename.split("?")[0]

    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log and "pytest" not in sys.modules:
            print(f"Using existing file {filename}", file=sys.stderr)
        return path

    if log and "pytest" not in sys.modules:
        print(f"Downloading {url}", file=sys.stderr)

    os.makedirs(osp.expanduser(osp.normpath(folder)), exist_ok=True)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, "wb") as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


def load_texas_dataset(raw_dir):
    # Wrap Heterophilous to DGL Graph Dataset format https://arxiv.org/pdf/2302.11640.pdf
    url = f"https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/texas_4_classes.npz"
    download_path = download_url(url, raw_dir)
    data = np.load(download_path)
    node_features = torch.tensor(data["node_features"])
    labels = torch.tensor(data["node_labels"])
    edges = torch.tensor(data["edges"])
    train_masks = torch.tensor(data["train_masks"]).T
    val_masks = torch.tensor(data["val_masks"]).T
    test_masks = torch.tensor(data["test_masks"]).T

    data = Data(x=node_features, y=labels, edge_index=edges.T,
                train_mask=train_masks, val_mask=val_masks, test_mask=test_masks)
    return data


GRAPHLAND_DATA_ROOT = './datasets/graphland'


def _load_graphland_dataset(name: str) -> dict:
    dataset_path = f'{GRAPHLAND_DATA_ROOT}/{name}'
    dataset = dict(np.load(f'{dataset_path}/data.npz', allow_pickle=True))
    return dataset


class DataSetFamily(Enum):
    heterophilic = auto()
    homophilic = auto()
    wiki_net = auto()
    web = auto()
    texas = auto()
    full = auto()
    attr = auto()
    cs = auto()
    coauthor = auto()
    airport = auto()
    actor = auto()
    amazon = auto()
    last_fm_asia = auto()
    deezer = auto()
    arxiv = auto()

    # graphland project
    graphland = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return DataSetFamily[s]
        except KeyError:
            raise ValueError()


class DataSet(Enum):
    """
        an object for the different datasets
    """
    # heterophilic
    roman_empire = auto()
    amazon_ratings = auto()
    minesweeper = auto()
    tolokers = auto()
    questions = auto()

    # homophilic
    cora = auto()
    pubmed = auto()
    citeseer = auto()

    # wiki_net
    chameleon = auto()
    squirrel = auto()

    # web
    cornell = auto()
    wisconsin = auto()

    # texas
    texas = auto()

    # full
    full_cora = auto()
    full_DBLP = auto()

    # attr
    wiki_attr = auto()
    blogcatalog = auto()

    # cs
    wiki_cs = auto()

    # coauthor
    co_cs = auto()
    co_physics = auto()

    # airport
    brazil = auto()
    usa = auto()
    europe = auto()

    # actor
    actor = auto()

    # amazon
    computers = auto()
    photo = auto()

    # last_fm_asia
    last_fm_asia = auto()

    # deezer
    deezer = auto()

    # arxiv
    arxiv = auto()

    # graphland project
    graphland_hm_categories_RL = auto()
    graphland_pokec_regions_RL = auto()
    graphland_web_topics_RL = auto()
    graphland_tolokers_2_RL = auto()
    graphland_city_reviews_RL = auto()
    graphland_artnet_exp_RL = auto()
    graphland_web_fraud_RL = auto()

    graphland_hm_categories_RH = auto()
    graphland_pokec_regions_RH = auto()
    graphland_web_topics_RH = auto()
    graphland_tolokers_2_RH = auto()
    graphland_city_reviews_RH = auto()
    graphland_artnet_exp_RH = auto()
    graphland_web_fraud_RH = auto()

    graphland_hm_categories_TH = auto()
    graphland_pokec_regions_TH = auto()
    graphland_web_topics_TH = auto()
    graphland_tolokers_2_TH = auto()
    graphland_city_reviews_TH = auto()
    graphland_artnet_exp_TH = auto()
    graphland_web_fraud_TH = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return DataSet[s]
        except KeyError:
            raise ValueError()

    def get_family(self) -> DataSetFamily:
        if self in [DataSet.roman_empire, DataSet.amazon_ratings, DataSet.minesweeper,
                    DataSet.tolokers, DataSet.questions]:
            return DataSetFamily.heterophilic
        elif self in [DataSet.cora, DataSet.pubmed, DataSet.citeseer]:
            return DataSetFamily.homophilic
        elif self in [DataSet.chameleon, DataSet.squirrel]:
            return DataSetFamily.wiki_net
        elif self in [DataSet.cornell, DataSet.wisconsin]:
            return DataSetFamily.web
        elif self is DataSet.texas:
            return DataSetFamily.texas
        elif self in [DataSet.full_cora, DataSet.full_DBLP]:
            return DataSetFamily.full
        elif self in [DataSet.wiki_attr, DataSet.blogcatalog]:
            return DataSetFamily.attr
        elif self is DataSet.wiki_cs:
            return DataSetFamily.cs
        elif self in [DataSet.co_cs, DataSet.co_physics]:
            return DataSetFamily.coauthor
        elif self in [DataSet.brazil, DataSet.usa, DataSet.europe]:
            return DataSetFamily.airport
        elif self is DataSet.actor:
            return DataSetFamily.actor
        elif self in [DataSet.computers, DataSet.photo]:
            return DataSetFamily.amazon
        elif self is DataSet.last_fm_asia:
            return DataSetFamily.last_fm_asia
        elif self is DataSet.deezer:
            return DataSetFamily.deezer
        elif self is DataSet.arxiv:
            return DataSetFamily.arxiv
        elif self in [
            DataSet.graphland_hm_categories_RL,
            DataSet.graphland_pokec_regions_RL,
            DataSet.graphland_web_topics_RL,
            DataSet.graphland_tolokers_2_RL,
            DataSet.graphland_city_reviews_RL,
            DataSet.graphland_artnet_exp_RL,
            DataSet.graphland_web_fraud_RL,

            DataSet.graphland_hm_categories_RH,
            DataSet.graphland_pokec_regions_RH,
            DataSet.graphland_web_topics_RH,
            DataSet.graphland_tolokers_2_RH,
            DataSet.graphland_city_reviews_RH,
            DataSet.graphland_artnet_exp_RH,
            DataSet.graphland_web_fraud_RH,

            DataSet.graphland_hm_categories_TH,
            DataSet.graphland_pokec_regions_TH,
            DataSet.graphland_web_topics_TH,
            DataSet.graphland_tolokers_2_TH,
            DataSet.graphland_city_reviews_TH,
            DataSet.graphland_artnet_exp_TH,
            DataSet.graphland_web_fraud_TH,
        ]:
            return DataSetFamily.graphland
        else:
            raise ValueError(f'DataSet {self.name} not supported in dataloader')

    def get_pca_components(self) -> Optional[int]:
        if self in [DataSet.full_cora,  # reduced from 8710
                    DataSet.co_cs,  # reduced from 6805
                    ]:
            return 2048
        elif self is DataSet.co_physics:  # 8415
            return 1024
        else:
            return None

    def load(self) -> Data:
        root = osp.join(ROOT_DIR, 'datasets', self.name)
        os.makedirs(root, exist_ok=True)
        n_components = self.get_pca_components()
        if n_components is None:
            transform = T.Compose([T.ToUndirected(), RemoveSelfLoops(), T.RemoveDuplicatedEdges(),
                                   NormalizeTransform()])
        else:
            transform = T.Compose([T.ToUndirected(), RemoveSelfLoops(), T.RemoveDuplicatedEdges(),
                                   PCATransform(n_components=n_components),
                                   NormalizeTransform()])
        dataset_family = self.get_family()
        if dataset_family is DataSetFamily.heterophilic:
            name = self.name.replace('_', '-').capitalize()
            data = HeterophilousGraphDataset(root=root, name=name, transform=transform)[0]
        elif dataset_family is DataSetFamily.homophilic:
            data = Planetoid(root=root, name=self.name, transform=transform)[0]
        elif dataset_family is DataSetFamily.wiki_net:
            data = WikipediaNetwork(root=root, name=self.name, geom_gcn_preprocess=True, transform=transform)[0]
        elif dataset_family is DataSetFamily.texas:
            data = load_texas_dataset(raw_dir=root)
            data = transform(data)
        elif dataset_family is DataSetFamily.web:
            data = WebKB(root=root, name=self.name, transform=transform)[0]
        elif dataset_family is DataSetFamily.full:
            data = CitationFull(root=root, name=self.name[5:], transform=transform)[0]
        elif dataset_family is DataSetFamily.attr:
            name = 'wiki' if 'wiki' in self.name else 'blogcatalog'
            data = AttributedGraphDataset(root=root, name=name, transform=transform)[0]
        elif dataset_family is DataSetFamily.cs:
            data = WikiCS(root=root, transform=transform)[0]
        elif dataset_family is DataSetFamily.coauthor:
            data = Coauthor(root=root, name=self.name[3:], transform=transform)[0]
        elif dataset_family is DataSetFamily.airport:
            data = Airports(root=root, name=self.name, transform=transform)[0]
        elif dataset_family is DataSetFamily.actor:
            data = Actor(root=root, transform=transform)[0]
        elif dataset_family is DataSetFamily.amazon:
            data = Amazon(root=root, name=self.name, transform=transform)[0]
        elif dataset_family is DataSetFamily.last_fm_asia:
            data = LastFMAsia(root=root, transform=transform)[0]
        elif dataset_family is DataSetFamily.deezer:
            data = DeezerEurope(root=root, transform=transform)[0]
        elif dataset_family is DataSetFamily.arxiv:
            data = PygNodePropPredDataset(root=root, name='ogbn-arxiv', transform=transform)
            split_idx = data.get_idx_split()
            data = data[0]
            train_indices, valid_indices, test_indices = split_idx["train"], split_idx["valid"], split_idx["test"]

            def to_mask(indices):
                mask = torch.BoolTensor(data.x.shape[0]).fill_(False)
                mask[indices] = 1
                return mask

            train_mask, val_mask, test_mask = map(to_mask, (train_indices, valid_indices, test_indices))
            setattr(data, 'train_mask', train_mask)
            setattr(data, 'val_mask', val_mask)
            setattr(data, 'test_mask', test_mask)
        elif dataset_family is DataSetFamily.graphland:
            dataset = _load_graphland_dataset(self.name.removeprefix('graphland_').replace('_', '-'))
            data = Data(
                x=torch.FloatTensor(dataset['features']),
                y=torch.LongTensor(dataset['labels']),
                edge_index=torch.LongTensor(dataset['edge_index']),
            )
            for part in ['train', 'val', 'test']:
                setattr(data, f'{part}_mask', torch.BoolTensor(dataset[f'{part}_mask']))
        else:
            raise ValueError(f'DataSet {self.name} not supported')
        
        # Y preprocessing
        data.y = data.y.squeeze(dim=-1)
        y_mat = F.one_hot(data.y).float()
        y_mat = normalize(x=y_mat)
        setattr(data, 'y_mat', y_mat)
        return data
