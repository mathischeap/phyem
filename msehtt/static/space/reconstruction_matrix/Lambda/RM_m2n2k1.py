# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from src.spaces.main import _degree_str_maker
from tools.miscellaneous.ndarray_cache import add_to_ndarray_cache, ndarray_key_comparer


# ---------------- OUTER ------------------------------------------------------------------------------

def rm__m2n2k1_outer(tpm, degree, xi, et):
    """"""
    assert isinstance(xi, np.ndarray) and xi.ndim == 1, f"xi must be 1d array."
    assert isinstance(et, np.ndarray) and et.ndim == 1, f"eta must be 1d array."
    assert np.min(xi) >= -1 and np.max(xi) <= 1, f"xi must be in [-1, 1]"
    assert np.min(et) >= -1 and np.max(et) <= 1, f"eta must be in [-1, 1]"
    elements = tpm.composition
    RM_u, RM_v = {}, {}
    for e in elements:
        element = elements[e]
        etype = element.etype
        if etype in ("orthogonal rectangle", "unique msepy curvilinear quadrilateral"):
            RM_u[e], RM_v[e] = ___rm221o_msepy_quadrilateral___(element, degree, xi, et)
        else:
            raise NotImplementedError()

    return RM_u, RM_v


_cache_rm221o_ = {}


def ___rm221o_msepy_quadrilateral___(element, degree, xi, et):
    """"""
    metric_signature = element.metric_signature
    if isinstance(metric_signature, str):
        key = metric_signature + _degree_str_maker(degree)
        cached, data = ndarray_key_comparer(_cache_rm221o_, [xi, et], check_str=key)
        do_cache = True
    else:
        key = None
        cached, data = False, None
        do_cache = False

    if cached:
        return data
    else:
        xi_et, bfs = element.bf('m2n2k1_outer', degree, xi, et)
        u, v = bfs
        iJ = element.ct.inverse_Jacobian_matrix(*xi_et)
        iJ0, iJ1 = iJ
        iJ00, iJ01 = iJ0
        iJ10, iJ11 = iJ1

        x0 = + u * iJ11
        x1 = - v * iJ01
        rm_e_x = np.vstack((x0, x1)).T

        y0 = - u * iJ10
        y1 = + v * iJ00
        rm_e_y = np.vstack((y0, y1)).T

        data = rm_e_x, rm_e_y

        if do_cache:
            add_to_ndarray_cache(_cache_rm221o_, [xi, et], data, check_str=key)
        else:
            pass

        return data


# ---------------- INNER ------------------------------------------------------------------------------

def rm__m2n2k1_inner(tpm, degree, xi, et):
    """"""
    assert isinstance(xi, np.ndarray) and xi.ndim == 1, f"xi must be 1d array."
    assert isinstance(et, np.ndarray) and et.ndim == 1, f"eta must be 1d array."
    assert np.min(xi) >= -1 and np.max(xi) <= 1, f"xi must be in [-1, 1]"
    assert np.min(et) >= -1 and np.max(et) <= 1, f"eta must be in [-1, 1]"
    elements = tpm.composition
    RM_u, RM_v = {}, {}
    for e in elements:
        element = elements[e]
        etype = element.etype
        if etype in ("orthogonal rectangle", "unique msepy curvilinear quadrilateral"):
            RM_u[e], RM_v[e] = ___rm221i_msepy_quadrilateral___(element, degree, xi, et)
        else:
            raise NotImplementedError()

    return RM_u, RM_v


_cache_rm221i_ = {}


def ___rm221i_msepy_quadrilateral___(element, degree, xi, et):
    """"""
    metric_signature = element.metric_signature
    if isinstance(metric_signature, str):
        key = metric_signature + _degree_str_maker(degree)
        cached, data = ndarray_key_comparer(_cache_rm221i_, [xi, et], check_str=key)
        do_cache = True
    else:
        key = None
        cached, data = False, None
        do_cache = False

    if cached:
        return data
    else:
        xi_et, bfs = element.bf('m2n2k1_inner', degree, xi, et)
        u, v = bfs
        iJ = element.ct.inverse_Jacobian_matrix(*xi_et)
        iJ0, iJ1 = iJ
        iJ00, iJ01 = iJ0
        iJ10, iJ11 = iJ1

        x0 = u * iJ00
        x1 = v * iJ10
        rm_e_x = np.vstack((x0, x1)).T

        y0 = u * iJ01
        y1 = v * iJ11
        rm_e_y = np.vstack((y0, y1)).T

        data = rm_e_x, rm_e_y

        if do_cache:
            add_to_ndarray_cache(_cache_rm221i_, [xi, et], data, check_str=key)
        else:
            pass

        return data
