# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3
# distutils: language=c++

cimport cython

import numpy as np

np.import_array() # cython numpy has been cimported in .pxd file

from libc.string cimport memset

cdef INT32_t INT32_INF = 0x3f3f3f3f

cdef class CKappaPruning:
    def __init__(self, INT32_t node_count, DOUBLE_t [:, ::1] dist):
        self.node_count = node_count
        self.dist = dist
    
    cpdef np.ndarray cal_max_diversity_subsets(self, INT32_t size):
        return self.select_subset_inds(size).base


    cdef DOUBLE_t _get_diversity_measure(self, INT32_t [:] nodes, INT32_t nodes_count) except? -1:
        cdef INT32_t _i, nodes_id = nodes[nodes_count - 1]
        cdef DOUBLE_t _now_measure = 0
        for _i in xrange(nodes_count - 1):
            _now_measure += self.dist[nodes_id, nodes[_i]]
        return _now_measure

    cdef INT32_t[:] select_subset_inds(self, INT32_t size):
        cdef INT32_t [:] visited = np.zeros((self.node_count, ), dtype=np.int32)
        cdef INT32_t [:] ret_array = np.empty((size, ), dtype=np.int32)
        cdef INT32_t _i, _j, _k

        cdef DOUBLE_t _measure, _max_measure = -INT32_INF
        cdef INT32_t now_count = 0, _max_i, _max_j

        for _i in xrange(self.node_count):
            for _j in xrange(_i + 1, self.node_count):
                ret_array[0] = _i
                ret_array[1] = _j
                _measure = self._get_diversity_measure(nodes=ret_array, nodes_count=2)
                if _measure > _max_measure:
                    _max_measure = _measure
                    _max_i = _i
                    _max_j = _j
        
        now_count = 2
        ret_array[0] =  _max_i
        ret_array[1] =  _max_j
        visited[...] = 0
        visited[_max_i] = 1
        visited[_max_j] = 1
        
        cdef INT32_t _max_node = -1
        for _i in xrange(size - 2):
            _max_measure = -INT32_INF
            _max_node = -1
            for _j in xrange(self.node_count):
                if visited[_j] == 0:
                    ret_array[now_count] = _j
                    _measure = self._get_diversity_measure(nodes=ret_array, nodes_count=now_count + 1)
                    if _measure > _max_measure:
                        _max_measure = _measure
                        _max_node = _j
            ret_array[now_count] = _max_node
            visited[_max_node] = 1
            now_count = now_count + 1
        return ret_array
