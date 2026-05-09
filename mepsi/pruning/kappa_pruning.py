import numpy as np

from .base import BaseSelector

from ._libs.kappa_pruning import CKappaPruning


class KappaPruning(BaseSelector):
    def __init__(self, node_count):
        super(KappaPruning, self).__init__(node_count)

    def pruning(self, dist, size):
        kappa_pruning = CKappaPruning(self.node_count, dist.astype(np.float))
        return kappa_pruning.cal_max_diversity_subsets(size)
