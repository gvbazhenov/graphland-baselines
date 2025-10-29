from enum import Enum, auto
from torch.nn import CrossEntropyLoss, MSELoss
import torch
from typing import NamedTuple
from torch import Tensor


class LossAndMetric(NamedTuple):
    val_loss: float
    test_loss: float
    val_metric: float
    test_metric: float

    def get_fold_metrics(self) -> Tensor:
        return torch.tensor([self.val_metric, self.test_metric])
