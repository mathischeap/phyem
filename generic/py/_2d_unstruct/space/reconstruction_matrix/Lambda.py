# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen


class ReconstructMatrixLambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._mesh = space.mesh
        self._freeze()

    def __call__(self, degree, xi, et, element_range=None):
        """"""
        abs_sp = self._space.abstract
        k = abs_sp.k
        orientation = abs_sp.orientation

        if element_range is None:
            element_range = list(self._mesh._elements_dict.keys())
        else:
            pass

        if k == 1:
            return getattr(self, f'_k{k}_{orientation}')(
                degree, xi, et, element_range
            )
        else:
            return getattr(self, f'_k{k}')(
                degree, xi, et, element_range
            )

    def _k0(self, degree, xi, et, element_range):
        """"""
        assert np.ndim(xi) == np.ndim(et) == 1, f"I need 1d xi and et"
        _, BF = self._space.basis_functions(degree, xi, et)
        rm_dict = dict()
        _global_cache_rm0_ = dict()
        for e in element_range:

            metric_signature = self._mesh[e].metric_signature

            if metric_signature in _global_cache_rm0_:
                pass

            else:
                bf = BF[e]
                x0 = bf[0].T
                _global_cache_rm0_[metric_signature] = (x0, )

            rm_dict[e] = _global_cache_rm0_[metric_signature]

        return rm_dict

    def _k1_outer(self, degree, xi, et, element_range):
        """"""
        assert np.ndim(xi) == np.ndim(et) == 1, f"I need 1d xi and et"
        xi_et, BF = self._space.basis_functions(degree, xi, et)
        iJ = self._mesh.ct.inverse_Jacobian_matrix(*xi_et, element_range=element_range)
        rm_cache = dict()
        rm_dict = dict()
        csm = self._space.basis_functions.csm(degree)
        for e in element_range:

            if e in csm:
                cache_index = e
            else:
                cache_index = self._mesh[e].metric_signature

            if cache_index in rm_cache:
                pass

            else:
                bf = BF[e]
                u, v = bf
                assert e in iJ, f"element #{e} is out of range."
                iJ0, iJ1 = iJ[e]
                iJ00, iJ01 = iJ0
                iJ10, iJ11 = iJ1

                x0 = + u * iJ11
                x1 = - v * iJ01
                rm_e_x = np.vstack((x0, x1)).T

                y0 = - u * iJ10
                y1 = + v * iJ00
                rm_e_y = np.vstack((y0, y1)).T

                rm = (rm_e_x, rm_e_y)

                rm_cache[cache_index] = rm

            rm_dict[e] = rm_cache[cache_index]

        return rm_dict

    def _k1_inner(self, degree, xi, et, element_range):
        """"""
        assert np.ndim(xi) == np.ndim(et) == 1, f"I need 1d xi and et"
        xi_et, BF = self._space.basis_functions(degree, xi, et)
        iJ = self._mesh.ct.inverse_Jacobian_matrix(*xi_et, element_range=element_range)
        rm_cache = dict()
        rm_dict = dict()
        csm = self._space.basis_functions.csm(degree)
        for e in element_range:
            if e in csm:
                cache_index = e
            else:
                cache_index = self._mesh[e].metric_signature

            if cache_index in rm_cache:
                pass

            else:
                bf = BF[e]
                u, v = bf
                assert e in iJ, f"element #{e} is out of range."
                iJ0, iJ1 = iJ[e]
                iJ00, iJ01 = iJ0
                iJ10, iJ11 = iJ1

                x0 = u * iJ00
                x1 = v * iJ10
                rm_e_x = np.vstack((x0, x1)).T

                y0 = u * iJ01
                y1 = v * iJ11
                rm_e_y = np.vstack((y0, y1)).T

                rm = (rm_e_x, rm_e_y)

                rm_cache[cache_index] = rm

            rm_dict[e] = rm_cache[cache_index]

        return rm_dict

    def _k2(self, degree, xi, et, element_range):
        """"""
        assert np.ndim(xi) == np.ndim(et) == 1, f"I need 1d xi and et"
        xi_et, BF = self._space.basis_functions(degree, xi, et)
        iJ = self._mesh.ct.inverse_Jacobian(*xi_et, element_range=element_range)
        rm_dict = dict()
        _global_cache_rm2_ = dict()
        for e in element_range:

            metric_signature = self._mesh[e].metric_signature

            if metric_signature in _global_cache_rm2_:
                pass

            else:
                bf = BF[e]
                ij = iJ[e]
                x0 = bf[0] * ij
                _global_cache_rm2_[metric_signature] = (x0.T, )

            rm_dict[e] = _global_cache_rm2_[metric_signature]

        return rm_dict
