import numpy as np

from typing import Union
from .base import BaseSelector
from ..forest.tree import DecisionTreeClassifier
from ._libs.mepsi import CMEPSISelector


class MEPSISelector(BaseSelector):
    def __init__(
        self,
        node_count,
        Xtrain: np.ndarray,
        Ytrain: np.ndarray,
        predictions: Union[np.ndarray, list],
        node_counts: np.ndarray,
        edit_dists: np.ndarray,
    ):
        super(MEPSISelector, self).__init__(node_count)

        predictions = np.array([np.arange(np.max(Ytrain) + 1) == x[:, None] for x in predictions])
        self.optimizer = CMEPSISelector(
            node_count=node_count,
            features=Xtrain.astype(np.float),
            labels=Ytrain.astype(np.int32),
            predictions=predictions.astype(np.int32),
            node_counts=node_counts,
            edit_dists=edit_dists,
        )

    def pruning(self, size):
        return self.optimizer.cal_max_diversity_subsets(size)
