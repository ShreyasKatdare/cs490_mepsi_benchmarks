import numpy as np
from joblib import Parallel, delayed


class PairWiseMetric:
    def __init__(self, inds):
        self.inds = inds
        self.inds_size = len(self.inds)

    @staticmethod
    def cal_pair_ind_measure(ind1, ind2):
        pass

    def cal_pairwirse_measures(
        self,
        return_array=False,
        symmetry=True,
        return_norm=True,
        n_jobs=-1,
        backend="multiprocessing",
        **kwargs,
    ):
        return_sign = self.sign if return_norm is True else 1
        iter_list1 = [(i, j) for i in range(self.inds_size) for j in range(i, self.inds_size)]
        ret_list = Parallel(n_jobs=n_jobs, verbose=False, backend=backend)(
            delayed(self.cal_pair_ind_measure)(self.inds[i], self.inds[j], **kwargs) for i, j in iter_list1
        )
        if not return_array:
            return return_sign * ret_list
        ret_array = np.zeros((self.inds_size, self.inds_size), dtype=np.float)
        if symmetry:
            for _id, (i, j) in enumerate(iter_list1):
                ret_array[i, j] = ret_array[j, i] = ret_list[_id]
        else:
            for _id, (i, j) in enumerate(iter_list1):
                ret_array[i, j] = ret_list[_id]

            iter_list2 = [(i, j) for i in range(self.inds_size) for j in range(i + 1, self.inds_size)]
            ret_list = Parallel(n_jobs=n_jobs, verbose=False, backend=backend)(
                delayed(self.cal_pair_ind_measure)(self.inds[j], self.inds[i], **kwargs) for i, j in iter_list2
            )
            for _id, (i, j) in enumerate(iter_list2):
                ret_array[j, i] = ret_list[_id]
        # print("RET:", ret_array)
        return return_sign * ret_array
