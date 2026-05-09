cimport numpy as np

ctypedef np.npy_int32 INT32_t
ctypedef np.npy_float64 DOUBLE_t

cdef class CKappaPruning:
    cdef INT32_t node_count
    cdef DOUBLE_t [:, ::1] dist
    
    cdef DOUBLE_t _get_diversity_measure(self, INT32_t [:] nodes, INT32_t nodes_count) except? -1
    cdef INT32_t[:] select_subset_inds(self, INT32_t size)
    cpdef np.ndarray cal_max_diversity_subsets(self, INT32_t size)