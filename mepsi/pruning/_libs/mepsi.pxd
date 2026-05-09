cimport numpy as np

from ...forest.tree._libs._tree cimport (
    Tree,
    INT32_t,
    DOUBLE_t,
)

ctypedef np.npy_bool BOOL_t

cdef class CMEPSISelector:
    cdef INT32_t node_count
    cdef DOUBLE_t [:, :] features
    cdef INT32_t [:] labels
    cdef DOUBLE_t [:] node_counts
    cdef DOUBLE_t [:, :] edit_dists
    cdef INT32_t [:, :, :] predictions
    
    cdef INT32_t[:] select_subset_inds(self, INT32_t size)

    cpdef np.ndarray cal_max_diversity_subsets(self, INT32_t size)