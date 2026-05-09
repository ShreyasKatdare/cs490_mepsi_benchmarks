# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3
# distutils: language=c++

cimport cython

import numpy as np
cimport numpy as np
np.import_array()

from cpython cimport array
import array

from libcpp.queue cimport queue
from libcpp.stack cimport stack
from libc.string cimport memset
from libc.math cimport fmax, fmin, sqrt, pow
from libcpp.algorithm cimport sort

cdef class TreeWithTED:

    def __init__(self, Tree tree):
        self._TREE_LEAF = -1
        self._TREE_UNDEFINED = -2

        self.feature = tree.feature
        self.children_left = tree.children_left
        self.children_right = tree.children_right
        self.node_count = tree.node_count
        self.keyroots_count = 0
        self.travel_count = 0

        #print(self.node_count)
        #print(self.feature.base)
        #print(self.children_left.base)
        #print(self.children_right.base)
        self.pre_process()

        

    cdef bint check_skip_node(self, SIZE_t node_id) nogil:
        return self.feature[node_id] == self._TREE_UNDEFINED

    cdef bint check_leaf_node(self, SIZE_t node_id) nogil:
        return self.feature[node_id] != self._TREE_UNDEFINED and self.feature[self.children_left[node_id]] == self._TREE_UNDEFINED and self.feature[self.children_right[node_id]] == self._TREE_UNDEFINED

    cdef void pre_process(self):

        cdef SIZE_t node_count = self.node_count
        cdef SIZE_t [:] feature = self.feature
        cdef SIZE_t [:] children_left = self.children_left
        cdef SIZE_t [:] children_right = self.children_right

        cdef SIZE_t root_id = 0
        self.postorder_nodes = np.empty((node_count, ), dtype=np.intp)
        cdef SIZE_t[:] preorder_nodes = np.empty((node_count, ), dtype=np.intp)
        cdef SIZE_t[:] nodes_fa = np.empty((node_count, ), dtype=np.intp)
        cdef SIZE_t[:] temp_nodes_fa = np.empty((node_count, ), dtype=np.intp)
        

        cdef stack [SIZE_t] now_stack = stack [SIZE_t] ()
        cdef stack [SIZE_t] fa_stack = stack [SIZE_t] ()
        
        cdef SIZE_t _i
        cdef SIZE_t new_node, new_fa
        cdef SIZE_t left_node, right_node
        
        with nogil:
            now_stack.push(root_id)
            fa_stack.push(self._TREE_UNDEFINED)
            self.travel_count = 0

            while now_stack.empty() == False:
                new_node, new_fa = now_stack.top(), fa_stack.top()
                now_stack.pop()
                fa_stack.pop()
                if self.check_skip_node(new_node):
                    continue
                preorder_nodes[self.travel_count] = new_node
                temp_nodes_fa[self.travel_count] = new_fa
    
                right_node = children_right[new_node]
                if right_node != self._TREE_LEAF:
                    now_stack.push(right_node)
                    fa_stack.push(self.travel_count)
                left_node = children_left[new_node]
                if left_node != self._TREE_LEAF:
                    now_stack.push(left_node)
                    fa_stack.push(self.travel_count)
                self.travel_count = self.travel_count + 1

            # reverse the internal nodes array
            # modify the nodes fa array with reversed node index
            for _i in xrange(self.travel_count):
                self.postorder_nodes[_i] = preorder_nodes[self.travel_count - _i - 1]
                if temp_nodes_fa[self.travel_count - _i - 1] == self._TREE_UNDEFINED:
                    nodes_fa[_i] = self._TREE_UNDEFINED
                else:
                    nodes_fa[_i] = self.travel_count - temp_nodes_fa[self.travel_count - _i - 1] - 1

        cdef SIZE_t lmd = -1
        cdef SIZE_t _idx, _rev_idx, temp_fa, _node
        
        self.lmds = np.empty((self.travel_count, ), dtype=np.intp)
        #self.lmds = array.clone(array.array('l'), self.travel_count, zero=False)
        memset(&self.lmds[0], -1, self.travel_count*sizeof(np.intp))
        self.keyroots = np.empty((self.travel_count, ), dtype=np.intp)
        #self.keyroots = array.clone(array.array('l'), self.travel_count, zero=False)
        #self.keyroots = cython.view.array(shape=(self.travel_count,), itemsize=sizeof(np.intp), format="i")

        with nogil:
        #if True:
            for _idx in xrange(self.travel_count):
                _node = self.postorder_nodes[_idx]
                #print(_node_id, self.postorder_nodes[_node_id], nodes_fa[_node_id])
                if self.check_leaf_node(_node):
                    self.lmds[_idx] = _idx
                    self.keyroots[self.keyroots_count] = _idx
                    temp_fa = nodes_fa[_idx]
                    while temp_fa != self._TREE_UNDEFINED and self.lmds[temp_fa] == -1:
                        self.lmds[temp_fa] = _idx
                        self.keyroots[self.keyroots_count] = temp_fa
                        temp_fa = nodes_fa[temp_fa]
                    self.keyroots_count = self.keyroots_count + 1
            sort(&self.keyroots[0], (&self.keyroots[0]) + self.keyroots_count)
            # print("=====")
            # print(self.tree.node_count, self.travel_count)
            # print("=====")
            # for i, x in enumerate(self.lmds):
            #     print(i, self.tree.threshold[self.postorder_nodes[i]], x, self.tree.threshold[self.postorder_nodes[x]])
            # print("=====")
            # for x in self.keyroots[:self.keyroots_count]:
        #     print(x, self.tree.threshold[self.postorder_nodes[x]])
        # print(self.lmds.base)
        return
    
    
    #cdef DOUBLE_t _delete_cost(self, const SIZE_t node_id, Tree tree):  
    #    return 1.0
    
    #cdef DOUBLE_t _add_cost(self, const SIZE_t node_id, Tree tree):  
    #    return 1.0

    #cdef DOUBLE_t _update_cost(self, const SIZE_t node_id1, const SIZE_t node_id2, Tree tree_A, Tree tree_B):
    #    return <DOUBLE_t> (tree_A.feature[node_id1] != tree_B.feature[node_id2])

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
        #Tree tree_A,
        #Tree tree_B,
        INT32_t [:, ::1] treedists,
        INT32_t [:, ::1] fd,
    ):

        cdef INT32_t m = id1 - A_lmds[id1] + 2
        cdef INT32_t n = id2 - B_lmds[id2] + 2
        # print(id1, A_lmds[id1])
        # print(id2, B_lmds[id2])
        #memset(&temp_fd[0], 0, m*n*sizeof(np.int32))

        cdef INT32_t off1 = A_lmds[id1] - 1
        cdef INT32_t off2 = B_lmds[id2] - 1
        
        cdef INT32_t x, y, p, q

        cdef INT32_t min_cost, update_cost


        fd[0, 0] = 0
        for x in xrange(1, m):
            fd[x, 0] = fd[x - 1, 0] + 1
        for y in xrange(1, n):
            fd[0, y] = fd[0, y - 1] + 1
        
        for x in xrange(1, m):
            for y in xrange(1, n):
                #print("x", x, off1, x+off1)
                #print("y", y, off2, y+off2)
                #print(fd[x - 1, y - 1])
                #delete_cost = self._delete_cost(A_postorder_nodes[x + off1], tree_A)
                #add_cost = self._add_cost(B_postorder_nodes[y + off2], tree_B)

                if A_lmds[id1] == A_lmds[x + off1] and B_lmds[id2] == B_lmds[y + off2]:
                    #update_cost = self._update_cost(A_postorder_nodes[x + off1], B_postorder_nodes[y + off2], tree_A, tree_B)
                    update_cost = <INT32_t> (A_feature[A_postorder_nodes[x + off1]] != B_feature[B_postorder_nodes[y + off2]])
                    fd[x, y] = min(
                        min(fd[x - 1, y] + 1, fd[x, y - 1] + 1),
                        fd[x - 1, y - 1] + update_cost
                    )
                    treedists[x + off1, y + off2] = fd[x, y]
                else:
                    p = A_lmds[x + off1] - 1 - off1
                    q = B_lmds[y + off2] - 1 - off2
                    #if p < 0 or q < 0:
                    #    print(p, q)
                    # print(p, q)
                    fd[x, y] = min(
                        min(fd[x - 1, y] + 1, fd[x, y - 1] + 1),
                        fd[p, q] + treedists[x + off1, y + off2]
                    )
        return

    cpdef DOUBLE_t cal_distance(self, TreeWithTED distree_B):
        cdef SIZE_t size_a = self.travel_count
        cdef SIZE_t size_b = distree_B.travel_count
        #print("SIZE:", size_a, size_b)
        cdef INT32_t [:, ::1] treedists = np.zeros((size_a, size_b), dtype=np.int32)
        cdef INT32_t [:, ::1] fd = np.zeros((size_a + 2, size_b + 2), dtype=np.int32)

        cdef INT32_t id_i, id_j, root1, root2
        #print("feature:", self.feature.base, distree_B.feature.base)
        #print("internals", size_a, size_b, self.postorder_nodes.base[:size_a], distree_B.postorder_nodes.base[:size_b])

        #cdef INT32_t m
        #cdef INT32_t n

        for id_i in xrange(self.keyroots_count):
            for id_j in xrange(distree_B.keyroots_count):
                root1 = self.keyroots[id_i]
                root2 = distree_B.keyroots[id_j]
                #m = root1 - self.lmds[root1] + 2
                #n = root2 - distree_B.lmds[root2] + 2
                #if m > size_a + 1 or n > size_b + 1:
                #    print(root1, self.lmds[root1], root2, distree_B.lmds[root2])

                self.treedist(
                    root1, 
                    root2, 
                    self.lmds,
                    distree_B.lmds,
                    self.postorder_nodes,
                    distree_B.postorder_nodes,
                    self.feature,
                    distree_B.feature,
                    treedists, 
                    fd,
                )
        #print(treedists.base[:20,:])
        #print(self.keyroots.base[:self.keyroots_count])
        #print(distree_B.keyroots.base[:distree_B.keyroots_count])
        #print(self.lmds.base)
        return <DOUBLE_t> treedists[size_a - 1, size_b - 1]
        