# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3
# distutils: language=c++

cimport cython

import numpy as np

np.import_array() # cython numpy has been cimported in .pxd file

cdef INT32_t INT32_INF = 0x3f3f3f3f

cdef class CMEPSISelector:
    def __init__(self, INT32_t node_count, np.ndarray[DOUBLE_t, ndim=2] features, np.ndarray[INT32_t, ndim=1] labels, np.ndarray[INT32_t, ndim=3] predictions, DOUBLE_t[:] node_counts, DOUBLE_t [:, :] edit_dists):
        self.node_count = node_count
        self.features = features
        self.labels = labels
        self.edit_dists = edit_dists
        self.predictions = predictions
        self.node_counts = node_counts
    
    cpdef np.ndarray cal_max_diversity_subsets(self, INT32_t size):
        return self.select_subset_inds(size).base

    cdef INT32_t[:] select_subset_inds(self, INT32_t size):
        cdef INT32_t [:] visited = np.zeros((self.node_count, ), dtype=np.int32)
        cdef DOUBLE_t [:] min_edit_dists = np.zeros((self.node_count, ), dtype=np.float)
        cdef INT32_t [:] ret_array = np.zeros((size, ), dtype=np.int32)
        cdef np.ndarray[DOUBLE_t, ndim=2] _features = self.features.base
        cdef np.ndarray[INT32_t, ndim=3] _predictions = self.predictions.base
        cdef np.ndarray[INT32_t, ndim=1] _labels = self.labels.base
        cdef np.ndarray[DOUBLE_t, ndim=1] _node_counts = self.node_counts.base
        cdef np.ndarray[DOUBLE_t, ndim=2] _edit_dists = self.edit_dists.base

        cdef DOUBLE_t mean_nodes_count = np.mean(_node_counts)
        cdef DOUBLE_t tradeoff_weight = 1.0 / (mean_nodes_count*size)
        cdef DOUBLE_t class_weight = _predictions[0].shape[1]/ (_predictions[0].shape[1] - 1.0)

        cdef np.ndarray[INT32_t, ndim=2] total_prediction =  np.zeros_like(self.predictions.base[0], dtype=np.int32)
        cdef np.ndarray[INT32_t, ndim=2] now_prediction

        cdef INT32_t _i, _j, min_i, min_j, min_id

        cdef DOUBLE_t _mdl, min_mdl = INT32_INF

        for _i in xrange(self.node_count):
            for _j in xrange(self.node_count):
                if _i <= _j:
                    continue
                now_prediction = _predictions[_i] + _predictions[_j]
                _mdl = class_weight*np.sum(np.argmax(now_prediction, axis=1) != _labels) / len(_labels) + tradeoff_weight*(_edit_dists[_i, _j] - _node_counts[_i] - _node_counts[_j])
                if _mdl < min_mdl:
                    min_mdl = _mdl
                    min_i = _i
                    min_j = _j
                    
        visited[min_i] = 1
        visited[min_j] = 1
        ret_array[0] = min_i
        ret_array[1] = min_j
        total_prediction += _predictions[min_i]
        total_prediction += _predictions[min_j]

        for _i in xrange(self.node_count):
            if visited[_i] == 0:
                min_edit_dists[_i] = min(_edit_dists[min_i, _i], _edit_dists[min_j, _i])

        for _i in xrange(size - 2):
            min_mdl = INT32_INF
            _mdl = 0
            for _j in xrange(self.node_count):
                if visited[_j] == 0:
                    now_prediction = total_prediction + _predictions[_j]
                    _mdl = class_weight*np.sum(np.argmax(now_prediction, axis=1) != _labels) / len(_labels) +  tradeoff_weight*(min_edit_dists[_j] - _node_counts[_j])
                    if _mdl < min_mdl:
                        min_mdl = _mdl
                        min_id = _j
                        
            total_prediction += _predictions[min_id]
            visited[min_id] = 1
            ret_array[_i + 2] = min_id
        
            for _j in xrange(self.node_count):
                if visited[_j] == 0 and _edit_dists[min_id, _j] < min_edit_dists[_j]:
                    min_edit_dists[_j] = _edit_dists[min_id, _j]
            
        return ret_array