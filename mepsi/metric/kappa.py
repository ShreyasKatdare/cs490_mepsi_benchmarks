import numpy as np

from .base import PairWiseMetric


class KappaStatisticsMetric(PairWiseMetric):
    def __init__(self, labels: np.ndarray, inds_predict: np.ndarray):
        super(KappaStatisticsMetric, self).__init__(inds_predict)
        self.m = len(labels)
        self.labels = labels
        self.sign = -1

    def cal_pair_ind_measure(self, ind1: np.ndarray, ind2: np.ndarray):
        Theta_1 = np.sum(ind1 == ind2) / self.m

        unique_class1, count1 = np.unique(ind1, return_counts=True)
        unique_class2, count2 = np.unique(ind2, return_counts=True)

        if count1.shape[0] == count2.shape[0]:
            Theta_2 = np.dot(count1, count2) / self.m**2
        else:
            new_count1 = [_count1 for (_class1, _count1) in zip(unique_class1, count1) if _class1 in unique_class2]
            new_count2 = [_count2 for (_class2, _count2) in zip(unique_class2, count2) if _class2 in unique_class1]
            Theta_2 = np.dot(new_count1, new_count2) / self.m**2

        if 1 - Theta_2 == 0:
            return 1
        return (Theta_1 - Theta_2) / (1 - Theta_2)
