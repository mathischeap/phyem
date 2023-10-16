# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from src.spaces.main import _degree_str_maker

from tools.frozen import Frozen
from tools.miscellaneous.ndarray_cache import add_to_ndarray_cache, ndarray_key_comparer
_global_cache_1_outer_ = {}
_global_cache_1_inner_ = {}
_global_cache_rm2_ = dict()
_global_cache_rm0_ = dict()


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
        for e in element_range:

            metric_signature = self._mesh[e].metric_signature
            cached, rm = ndarray_key_comparer(
                _global_cache_rm0_, [xi, et], check_str=metric_signature)

            if cached:
                pass

            else:
                rm = (BF[e][0].T, )
                add_to_ndarray_cache(
                    _global_cache_rm0_, [xi, et], rm, check_str=metric_signature,
                    maximum=16,
                )

            rm_dict[e] = rm

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
                use_global_cache = False
                check_str = None
            else:
                cache_index = self._mesh[e].metric_signature
                use_global_cache = True
                check_str = cache_index + _degree_str_maker(degree)

            if cache_index in rm_cache:
                pass

            else:
                if use_global_cache:
                    cached, rm = ndarray_key_comparer(
                        _global_cache_1_outer_, [xi, et], check_str=check_str)
                else:
                    cached = False
                    rm = None

                if cached:
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

                    if use_global_cache:
                        add_to_ndarray_cache(
                            _global_cache_1_outer_, [xi, et], rm, check_str=check_str,
                            maximum=16,
                        )
                    else:
                        pass

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
                use_global_cache = False
                check_str = None
            else:
                cache_index = self._mesh[e].metric_signature
                use_global_cache = True
                check_str = cache_index + _degree_str_maker(degree)

            if cache_index in rm_cache:
                pass

            else:
                if use_global_cache:
                    cached, rm = ndarray_key_comparer(
                        _global_cache_1_inner_, [xi, et], check_str=check_str)
                else:
                    cached = False
                    rm = None

                if cached:
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

                    if use_global_cache:
                        add_to_ndarray_cache(
                            _global_cache_1_inner_, [xi, et], rm, check_str=check_str,
                            maximum=16,
                        )
                    else:
                        pass

                rm_cache[cache_index] = rm

            rm_dict[e] = rm_cache[cache_index]

        return rm_dict

    def _k2(self, degree, xi, et, element_range):
        """"""
        assert np.ndim(xi) == np.ndim(et) == 1, f"I need 1d xi and et"
        xi_et, BF = self._space.basis_functions(degree, xi, et)
        iJ = self._mesh.ct.inverse_Jacobian(*xi_et, element_range=element_range)
        rm_dict = dict()
        for e in element_range:

            metric_signature = self._mesh[e].metric_signature
            cached, rm = ndarray_key_comparer(
                _global_cache_rm2_, [xi, et], check_str=metric_signature)

            if cached:
                pass

            else:
                x0 = BF[e][0] * iJ[e]
                rm = (x0.T, )
                add_to_ndarray_cache(
                    _global_cache_rm2_, [xi, et], rm, check_str=metric_signature,
                    maximum=16,
                )

            rm_dict[e] = rm

        return rm_dict
