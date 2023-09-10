# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from typing import Dict
from scipy.sparse import dia_array
from tools.frozen import Frozen
from msehy.py2.space.basis_functions.Lambda import MseHyPy2BasisFunctionsLambda
from msehy.py2.space.basis_functions.wrapper import MseHyPy2BasisFunctionWrapper

from tools.miscellaneous.ndarray_cache import add_to_ndarray_cache, ndarray_key_comparer

_global_csm_cache = {}
_global_array_cache = {}


class MseHyPy2BasisFunctions(Frozen):
    """"""

    def __init__(self, space):
        """Generation in-dependent."""
        self._space = space
        self._Lambda_cache = {}
        self._freeze()

    def __raw_bfs__(self, degree):
        """Generation in-dependent."""
        indicator = self._space.abstract.indicator
        key = str(degree)
        if indicator in ('Lambda', ):
            if key in self._Lambda_cache:
                Lambda_bf = self._Lambda_cache[key]
            else:
                Lambda_bf = MseHyPy2BasisFunctionsLambda(self._space, degree)
                self._Lambda_cache[key] = Lambda_bf
            return Lambda_bf

        else:
            raise NotImplementedError()

    def __call__(self, degree, g, xi, et):
        """"""
        g = self._space._pg(g)
        meshgrid_xi_et, bf_qt = self.__raw_bfs__(degree)(g, xi, et)
        csm = self.cochain_switch_matrix(degree, g=g)
        return meshgrid_xi_et, MseHyPy2BasisFunctionWrapper(bf_qt, csm)

    def cochain_switch_matrix(self, degree, g=None):
        """"""
        indicator = self._space.abstract.indicator
        if indicator == 'Lambda':
            k = self._space.abstract.k
            if k != 1:
                return {}
            else:
                g = self._space._pg(g)
                orientation = self._space.abstract.orientation
                p = self._space[degree].p
                key = f"g{g}-p{p}-{orientation}"

                if key in _global_csm_cache:
                    return _global_csm_cache[key]
                else:
                    pass

                # make csm below ---------------------------
                csm = self._make_csm(g, p, orientation)

                # -- split csm into components and make them into sparse matrices
                num_dofs_components = self._space.num_local_dof_components(degree)

                for index in csm:
                    if isinstance(index, str):
                        components = num_dofs_components['t']
                    else:
                        components = num_dofs_components['q']

                    c0, c1 = components
                    array = csm[index]
                    array0 = array[:c0]
                    array1 = array[-c1:]

                    cached, data = ndarray_key_comparer(_global_array_cache, [array0])
                    if cached:
                        csm0 = data
                    else:
                        csm0 = dia_array(np.diag(array0))
                        add_to_ndarray_cache(_global_array_cache, [array0], csm0)

                    cached, data = ndarray_key_comparer(_global_array_cache, [array1])
                    if cached:
                        csm1 = data
                    else:
                        csm1 = dia_array(np.diag(array1))
                        add_to_ndarray_cache(_global_array_cache, [array1], csm1)

                    csm[index] = (csm0, csm1)

                _global_csm_cache[key] = csm
                return csm
        else:
            raise NotImplementedError()

    def _make_csm(self, g, p, orientation):
        """"""
        px, py = p
        assert px == py, 'must be!'
        representative = self._space.mesh[g]
        if orientation == 'inner':
            opposite_pairs = representative.opposite_pairs_inner
        elif orientation == 'outer':
            opposite_pairs = representative.opposite_pairs_outer
        else:
            raise Exception()
        num_dofs = self._space.num_local_dofs.Lambda._k1(p)
        csm: Dict = dict()
        for switch_location in opposite_pairs:
            fc_index = switch_location[0]
            fc = representative[fc_index]

            if fc_index in csm:
                pass
            else:
                csm[fc_index] = np.ones(num_dofs[fc._type], dtype=int)

            if isinstance(fc_index, str):
                edge_index = switch_location[1]

            else:
                m, n = switch_location[1:]
                edge_index = m * 2 + n

            edge_dofs = self._space.find.local_dofs(fc._type, edge_index, p)
            array = csm[fc_index]
            array[edge_dofs] = -1
            csm[fc_index] = array

        return csm
