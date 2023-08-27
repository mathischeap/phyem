# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen


class MsePyReconstructMatrixBundle(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._mesh = space.mesh
        self._e2c = space.mesh.elements._index_mapping._e2c
        self._freeze()

    def __call__(self, degree, *meshgrid_xi_et_sg, element_range=None):
        """"""

        abs_sp = self._space.abstract
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
            return getattr(self, f'_m{m}_n{n}_k{k}_{orientation}')(
                degree, element_range, *meshgrid_xi_et_sg
            )
        else:
            return getattr(self, f'_m{m}_n{n}_k{k}')(
                degree, element_range, *meshgrid_xi_et_sg
            )

    def _m2_n2_k0(self, degree, element_range, *meshgrid_xi_et):
        """"""
        xi, et = meshgrid_xi_et
        assert np.ndim(xi) == np.ndim(et) == 1, f"I need 1d xi and et"
        _, bf = self._space.basis_functions[degree](*meshgrid_xi_et)
        rm_cache = dict()
        rm_dict = dict()

        for e in element_range:

            cache_index = self._e2c[e]

            if cache_index in rm_cache:
                pass

            else:
                x0 = bf[0].T
                x1 = bf[1].T

                zero0 = np.zeros_like(x0)
                zero1 = np.zeros_like(x1)

                x0 = np.hstack([x0, zero1])
                x1 = np.hstack([zero0, x1])

                rm_cache[cache_index] = (x0, x1)

            rm_dict[e] = rm_cache[cache_index]

        return rm_dict

    def _m2_n2_k2(self, degree, element_range, *meshgrid_xi_et):
        """"""
        xi, et = meshgrid_xi_et
        assert np.ndim(xi) == np.ndim(et) == 1, f"I need 1d xi and et"
        xi_et, bf = self._space.basis_functions[degree](*meshgrid_xi_et)
        iJ = self._mesh.elements.ct.inverse_Jacobian(*xi_et, element_range=element_range)
        rm_cache = dict()
        rm_dict = dict()

        for e in element_range:
            ij = iJ[e]
            cache_index = self._e2c[e]

            if cache_index in rm_cache:
                pass

            else:
                x0 = (bf[0] * ij).T
                x1 = (bf[1] * ij).T

                zero0 = np.zeros_like(x0)
                zero1 = np.zeros_like(x1)

                x0 = np.hstack([x0, zero1])
                x1 = np.hstack([zero0, x1])

                rm_cache[cache_index] = (x0, x1)

            rm_dict[e] = rm_cache[cache_index]

        return rm_dict

    def _m2_n2_k1_inner(self, degree, element_range, *meshgrid_xi_et):
        """"""
        xi, et = meshgrid_xi_et
        assert np.ndim(xi) == np.ndim(et) == 1, f"I need 1d xi and et"
        xi_et, BF = self._space.basis_functions[degree](*meshgrid_xi_et)

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
                u, v = BF[0]
                x0 = u * iJ00
                x1 = v * iJ10
                rm_00 = np.vstack((x0, x1)).T

                y0 = u * iJ01
                y1 = v * iJ11
                rm_01 = np.vstack((y0, y1)).T

                u, v = BF[1]
                x0 = u * iJ00
                x1 = v * iJ10
                rm_10 = np.vstack((x0, x1)).T

                y0 = u * iJ01
                y1 = v * iJ11
                rm_11 = np.vstack((y0, y1)).T

                zero0 = np.zeros_like(rm_00)
                zero1 = np.zeros_like(rm_10)

                rm_00 = np.hstack([rm_00, zero1])
                rm_01 = np.hstack([rm_01, zero1])
                rm_10 = np.hstack([zero0, rm_10])
                rm_11 = np.hstack([zero0, rm_11])

                rm_cache[cache_index] = ((rm_00, rm_01), (rm_10, rm_11))

            rm_dict[e] = rm_cache[cache_index]

        return rm_dict

    def _m2_n2_k1_outer(self, degree, element_range, *meshgrid_xi_et):
        """"""
        xi, et = meshgrid_xi_et
        assert np.ndim(xi) == np.ndim(et) == 1, f"I need 1d xi and et"
        xi_et, BF = self._space.basis_functions[degree](*meshgrid_xi_et)

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
                u, v = BF[0]
                x0 = + u * iJ11
                x1 = - v * iJ01
                rm_00 = np.vstack((x0, x1)).T

                y0 = - u * iJ10
                y1 = + v * iJ00
                rm_01 = np.vstack((y0, y1)).T

                u, v = BF[1]
                x0 = + u * iJ11
                x1 = - v * iJ01
                rm_10 = np.vstack((x0, x1)).T

                y0 = - u * iJ10
                y1 = + v * iJ00
                rm_11 = np.vstack((y0, y1)).T

                zero0 = np.zeros_like(rm_00)
                zero1 = np.zeros_like(rm_10)

                rm_00 = np.hstack([rm_00, zero1])
                rm_01 = np.hstack([rm_01, zero1])
                rm_10 = np.hstack([zero0, rm_10])
                rm_11 = np.hstack([zero0, rm_11])

                rm_cache[cache_index] = ((rm_00, rm_01), (rm_10, rm_11))

            rm_dict[e] = rm_cache[cache_index]

        return rm_dict
