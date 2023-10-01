# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen


class MseHyPy2ReconstructMatrixLambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._mesh = space.mesh
        self._freeze()

    def __call__(self, degree, g, *meshgrid_xi_et, fc_range=None):
        """"""
        g = self._space._pg(g)
        representative = self._mesh[g]

        abs_sp = self._space.abstract
        m = abs_sp.m
        n = abs_sp.n
        k = abs_sp.k
        orientation = abs_sp.orientation

        if fc_range is None:
            fc_range = representative._fundamental_cells.keys()
        elif fc_range.__class__.__name__ in ('float', 'int', 'int32', 'int64'):
            fc_range = [fc_range, ]
        else:
            pass

        if m == n == 2 and k == 1:
            return getattr(self, f'_n{n}_k{k}_{orientation}')(
                degree, g, fc_range, *meshgrid_xi_et
            )
        else:
            return getattr(self, f'_n{n}_k{k}')(
                degree, g, fc_range, *meshgrid_xi_et
            )

    def _n2_k0(self, degree, g, fc_range, *meshgrid_xi_et):
        """"""
        g = self._mesh._pg(g)
        representative = self._mesh[g]

        xi, et = meshgrid_xi_et
        assert np.ndim(xi) == np.ndim(et) == 1, f"I need 1d xi and et"
        _, BF = self._space.basis_functions(degree, g, *meshgrid_xi_et)

        rm_cache = dict()
        rm_dict = dict()

        for i in fc_range:
            bf = BF[i]
            cache_index = representative[i].metric_signature

            if cache_index in rm_cache:
                pass

            else:
                x0 = bf[0].T
                rm_cache[cache_index] = (x0, )

            rm_dict[i] = rm_cache[cache_index]

        return rm_dict

    def _n2_k1_outer(self, degree, g, fc_range, *meshgrid_xi_et):
        """"""
        g = self._mesh._pg(g)
        representative = self._mesh[g]

        xi, et = meshgrid_xi_et
        assert np.ndim(xi) == np.ndim(et) == 1, f"I need 1d xi and et"
        xi_et, BF = self._space.basis_functions(degree, g, *meshgrid_xi_et)

        iJ = representative.ct.inverse_Jacobian_matrix(*xi_et, fc_range=fc_range)
        rm_cache = dict()
        rm_dict = dict()
        for i in fc_range:

            cache_index = representative[i].metric_signature

            if cache_index in rm_cache:
                pass

            else:
                bf = BF[i]
                u, v = bf

                assert i in iJ, f"cell #{i} is out of range."
                iJ0, iJ1 = iJ[i]
                iJ00, iJ01 = iJ0
                iJ10, iJ11 = iJ1

                x0 = + u * iJ11
                x1 = - v * iJ01
                rm_e_x = np.vstack((x0, x1)).T

                y0 = - u * iJ10
                y1 = + v * iJ00
                rm_e_y = np.vstack((y0, y1)).T

                rm_cache[cache_index] = (rm_e_x, rm_e_y)

            rm_dict[i] = rm_cache[cache_index]

        return rm_dict

    def _n2_k1_inner(self, degree, g, fc_range, *meshgrid_xi_et):
        """"""
        g = self._mesh._pg(g)
        representative = self._mesh[g]

        xi, et = meshgrid_xi_et
        assert np.ndim(xi) == np.ndim(et) == 1, f"I need 1d xi and et"
        xi_et, BF = self._space.basis_functions(degree, g, *meshgrid_xi_et)

        iJ = representative.ct.inverse_Jacobian_matrix(*xi_et, fc_range=fc_range)
        rm_cache = dict()
        rm_dict = dict()
        for i in fc_range:

            cache_index = representative[i].metric_signature

            if cache_index in rm_cache:
                pass

            else:
                bf = BF[i]
                u, v = bf
                assert i in iJ, f"cell #{i} is out of range."
                iJ0, iJ1 = iJ[i]
                iJ00, iJ01 = iJ0
                iJ10, iJ11 = iJ1
                x0 = u * iJ00
                x1 = v * iJ10
                rm_e_x = np.vstack((x0, x1)).T

                y0 = u * iJ01
                y1 = v * iJ11
                rm_e_y = np.vstack((y0, y1)).T

                rm_cache[cache_index] = (rm_e_x, rm_e_y)

            rm_dict[i] = rm_cache[cache_index]

        return rm_dict

    def _n2_k2(self, degree, g, fc_range, *meshgrid_xi_et):
        """"""
        g = self._mesh._pg(g)
        representative = self._mesh[g]

        xi, et = meshgrid_xi_et
        assert np.ndim(xi) == np.ndim(et) == 1, f"I need 1d xi and et"
        xi_et, BF = self._space.basis_functions(degree, g, *meshgrid_xi_et)

        iJ = representative.ct.inverse_Jacobian(*xi_et, fc_range=fc_range)
        rm_cache = dict()
        rm_dict = dict()
        for i in fc_range:
            cache_index = representative[i].metric_signature

            if cache_index in rm_cache:
                pass

            else:
                bf = BF[i]
                ij = iJ[i]
                x0 = bf[0] * ij
                rm_cache[cache_index] = (x0.T, )

            rm_dict[i] = rm_cache[cache_index]

        return rm_dict
