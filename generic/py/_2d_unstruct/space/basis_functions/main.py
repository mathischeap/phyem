# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from scipy.sparse import dia_array
from tools.frozen import Frozen
from generic.py._2d_unstruct.space.basis_functions.Lambda import BasisFunctionsLambda
from src.spaces.main import _degree_str_maker
from tools.miscellaneous.ndarray_cache import add_to_ndarray_cache, ndarray_key_comparer


class BasisFunctions(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._Lambda = None
        self._csm_cache = {}
        self._csm_1inner_cache = {}
        self._csm_1outer_cache = {}
        self._bfs_cache = {}
        self._freeze()

    def ___raw_bfs___(self, degree, xi, et):
        """"""
        indicator = self._space.abstract.indicator
        if indicator == 'Lambda':
            return self.Lambda(degree, xi, et)
        else:
            raise NotImplementedError()

    @property
    def Lambda(self):
        if self._Lambda is None:
            self._Lambda = BasisFunctionsLambda(self._space)
        return self._Lambda

    def __call__(self, degree, xi, et):
        """"""
        key = _degree_str_maker(degree)
        cached, xi_et_bfs = ndarray_key_comparer(self._bfs_cache, [xi, et], check_str=key)
        if cached:
            return xi_et_bfs
        else:
            xi_et, bfs = self.___raw_bfs___(degree, xi, et)
            csm = self.csm(degree)
            bfs = _BFWrapper(self._space.mesh, bfs, csm)
            add_to_ndarray_cache(self._bfs_cache, [xi, et], [xi_et, bfs], check_str=key)
            return xi_et, bfs

    def csm(self, degree):
        """"""
        key = _degree_str_maker(degree)
        if key in self._csm_cache:
            return self._csm_cache[key]
        else:
            pass

        k = self._space.abstract.k
        indicator = self._space.abstract.indicator

        if indicator == 'Lambda':
            if k != 1:
                return {}
            else:
                orientation = self._space.abstract.orientation
                if orientation == 'inner':
                    self._csm_cache[key] = self._csm_Lambda_k1_inner(degree)
                else:
                    self._csm_cache[key] = self._csm_Lambda_k1_outer(degree)
        else:
            raise NotImplementedError()

        return self._csm_cache[key]

    def _csm_Lambda_k1_outer(self, degree):
        """"""
        key = _degree_str_maker(degree)
        if key in self._csm_1outer_cache:
            return self._csm_1outer_cache[key]
        else:
            pass
        mesh = self._space.mesh
        opposite_pairs = mesh.opposite_outer_orientation_pairs
        p = self._space[degree].p
        csm = dict()
        num_local_dofs = self._space.num_local_dofs.Lambda._k1(p)
        for switch_position in opposite_pairs:
            element, edge_index = switch_position
            ele_type = mesh[element].type
            local_dofs = self._space.find.local_dofs._Lambda_k1_outer(ele_type, edge_index, degree)
            num_dofs = num_local_dofs[ele_type]
            if element not in csm:
                csm[element] = np.ones(num_dofs)
            else:
                pass
            csm[element][local_dofs] = -1
        for element in csm:
            csm[element] = dia_array(np.diag(csm[element])).tocsc()
        self._csm_1outer_cache[key] = csm
        return csm

    def _csm_Lambda_k1_inner(self, degree):
        """"""
        key = _degree_str_maker(degree)
        if key in self._csm_1inner_cache:
            return self._csm_1inner_cache[key]
        else:
            pass
        mesh = self._space.mesh
        opposite_pairs = mesh.opposite_inner_orientation_pairs
        p = self._space[degree].p
        csm = dict()
        num_local_dofs = self._space.num_local_dofs.Lambda._k1(p)
        for switch_position in opposite_pairs:
            element, edge_index = switch_position
            ele_type = mesh[element].type
            local_dofs = self._space.find.local_dofs._Lambda_k1_inner(ele_type, edge_index, degree)
            num_dofs = num_local_dofs[ele_type]
            if element not in csm:
                csm[element] = np.ones(num_dofs)
            else:
                pass
            csm[element][local_dofs] = -1
        for element in csm:
            csm[element] = dia_array(np.diag(csm[element])).tocsc()
        self._csm_1inner_cache = csm
        return csm


class _BFWrapper(Frozen):
    """"""
    def __init__(self, mesh, bfs, csm):
        """"""
        self._mesh = mesh
        self._bfs = bfs
        self._csm = csm
        self._freeze()

    def __getitem__(self, index):
        """"""
        element = self._mesh[index]
        ele_type = element.type
        bf = self._bfs[ele_type]
        if index in self._csm:
            csm = self._csm[index]
            if len(bf) == 2:
                bf0, bf1 = bf
                num_dof_0 = len(bf0)
                csm0 = csm[:num_dof_0, :num_dof_0]
                csm1 = csm[num_dof_0:, num_dof_0:]
                bf0 = csm0 @ bf0
                bf1 = csm1 @ bf1
                return bf0, bf1
            else:
                raise Exception()
        else:
            return bf
