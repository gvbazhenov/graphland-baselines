from enum import Enum, auto
from typing import List
import math

from helpers.datasets import DataSet

TRAIN_MULTISETS = [
        [],
        [DataSet.texas, DataSet.tolokers],
        [DataSet.photo, DataSet.texas, DataSet.roman_empire, DataSet.tolokers],
        [DataSet.photo, DataSet.texas, DataSet.usa, DataSet.actor, DataSet.roman_empire, DataSet.tolokers],
        [DataSet.computers, DataSet.photo, DataSet.texas, DataSet.usa, DataSet.europe, DataSet.actor, DataSet.roman_empire, DataSet.tolokers],
    ]


class TrainTestSetup(Enum):
    trainset1 = auto()
    inc_trainset = auto()

    _all_datasets = set(DataSet)

    @staticmethod
    def from_string(s: str):
        try:
            return TrainTestSetup[s]
        except KeyError:
            raise ValueError(f"Unknown setup name: {s}")

    def get_train_datasets(self, train_size: int) -> List[DataSet]:
        if self is TrainTestSetup.trainset1:
            return [DataSet.cora]

        assert train_size in [1, 3, 5, 7, 9], "Invalid train size"
        train_idx = math.floor(train_size/2)
        trainset_list = [DataSet.cora] + TRAIN_MULTISETS[train_idx]
        return sorted(trainset_list, key=lambda x: x.value)

    def get_test_datasets(self) -> List[DataSet]:
        testset_list = [
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
            DataSet.graphland_artnet_exp_TH,
            DataSet.graphland_web_fraud_TH,
        ]
        return sorted(testset_list, key=lambda x: x.value)
