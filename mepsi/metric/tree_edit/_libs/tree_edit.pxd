
from ....forest.tree._libs._tree cimport (
    SIZE_t,
    INT32_t,
    DOUBLE_t,
    Tree
)

from ....forest.tree._libs._utils cimport sizet_ptr_to_ndarray

cdef class TreeWithTED:
    cdef SIZE_t[:] lmds
    cdef SIZE_t[:] keyroots
    cdef SIZE_t keyroots_count
    cdef SIZE_t[:] postorder_nodes
    cdef SIZE_t travel_count

    cdef SIZE_t[:] feature
    cdef SIZE_t[:] children_left
    cdef SIZE_t[:] children_right
    cdef SIZE_t node_count

    cdef SIZE_t _TREE_LEAF
    cdef SIZE_t _TREE_UNDEFINED

    cdef bint check_skip_node(self, SIZE_t node_id) nogil
    cdef bint check_leaf_node(self, SIZE_t node_id) nogil
    cdef void pre_process(self)


    #cdef DOUBLE_t _delete_cost(self, const SIZE_t node_id, Tree tree)
    #cdef DOUBLE_t _add_cost(self, const SIZE_t node_id, Tree tree)
    #cdef DOUBLE_t _update_cost(self, const SIZE_t node_id1, const SIZE_t node_id2, Tree tree_A, Tree tree_B)


    cdef void treedist(
        self,
        const SIZE_t id1,
        const SIZE_t id2,
        const SIZE_t [:] A_lmds,
        const SIZE_t [:] B_lmds,
        const SIZE_t [:] A_postorder_nodes,
        const SIZE_t [:] B_postorder_nodes,
        const SIZE_t [:] A_feature,
        const SIZE_t [:] B_feature,
        INT32_t [:, ::1] treedists,
        INT32_t [:, ::1] fd,
    )

    cpdef DOUBLE_t cal_distance(self, TreeWithTED distree_B)
