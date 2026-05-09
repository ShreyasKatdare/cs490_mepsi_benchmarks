import numpy as np

from ._libs.tree_edit import TreeWithTED
from ..base import PairWiseMetric
from ...forest.tree import BaseDecisionTree


class TreeEditMetric(PairWiseMetric):
    def __init__(self, inds):
        super(TreeEditMetric, self).__init__(inds)
        self.sign = 1

    @staticmethod
    def cal_pair_ind_measure(ind1: BaseDecisionTree, ind2: BaseDecisionTree):
        distree1 = TreeWithTED(tree=ind1.tree_)
        distree2 = TreeWithTED(tree=ind2.tree_)
        return distree1.cal_distance(distree2)
