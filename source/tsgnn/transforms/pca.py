import torch
from torch_geometric.data import Data
from sklearn.decomposition import PCA
from torch_geometric.transforms import BaseTransform

from transforms.normalize import standardize


class PCATransform(BaseTransform):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def __call__(self, data: Data) -> Data:
        if data.x.shape[1] > self.n_components:
            data.x = standardize(x=data.x)
            x = data.x.numpy()
            data.x = torch.from_numpy(self.pca.fit_transform(x))
        return data
