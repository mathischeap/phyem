# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 3:18 PM on 7/17/2023
"""
import numpy as np
from tools.frozen import Frozen


class MsePyMeshElementReconstructMatrixLambda(Frozen):
    """"""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._mesh = rf.mesh
        self._degree = rf._degree
        self._e2c = rf.mesh.elements._index_mapping._e2c
        self._freeze()

    def __call__(self, *args, element_range=None):
        """"""

        abs_sp = self._f.space.abstract
        m = abs_sp.m
        n = abs_sp.n
        k = abs_sp.k
        orientation = abs_sp.orientation

        if element_range is None:
            element_range = range(self._mesh.elements._num)
        elif element_range.__class__.__name__ in ('float', 'int', 'int32', 'int64'):
            element_range = [element_range, ]
        else:
            pass

        if m == n == 2 and k == 1:
            return getattr(self, f'_m{m}_n{n}_k{k}_{orientation}')(*args, element_range=element_range)
        else:
            return getattr(self, f'_m{m}_n{n}_k{k}')(*args, element_range=element_range)

    def _m2_n2_k1_outer(self, *meshgrid_xi_et, element_range=None):
        """"""
        xi, et = meshgrid_xi_et
        assert np.ndim(xi) == np.ndim(et) == 1, f"I need 1d xi and et"
        xi_et, bf = self._f.space.basis_functions[self._degree](*meshgrid_xi_et)

        u, v = bf

        iJ = self._mesh.elements.ct.inverse_Jacobian_matrix(*xi_et, element_range=element_range)

        rm_cache = dict()
        rm_dict = dict()
        for e in element_range:
            assert e in iJ, f"element #{e} is out of range."
            iJ0, iJ1 = iJ[e]
            iJ00, iJ01 = iJ0
            iJ10, iJ11 = iJ1

            cache_index = self._e2c[e]

            if cache_index in rm_cache:
                pass

            else:
                x0 = + u * iJ11
                x1 = - v * iJ01
                rm_e_x = np.vstack((x0, x1)).T

                y0 = - u * iJ10
                y1 = + v * iJ00
                rm_e_y = np.vstack((y0, y1)).T

                rm_cache[cache_index] = (rm_e_x, rm_e_y)

            rm_dict[e] = rm_cache[cache_index]

        return rm_dict
