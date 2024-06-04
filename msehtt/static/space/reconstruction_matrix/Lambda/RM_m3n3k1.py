# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from src.spaces.main import _degree_str_maker
from tools.miscellaneous.ndarray_cache import add_to_ndarray_cache, ndarray_key_comparer


def rm__m3n3k1(tpm, degree, xi, et, sg):
    """"""
    assert isinstance(xi, np.ndarray) and xi.ndim == 1, f"xi must be 1d array."
    assert isinstance(et, np.ndarray) and et.ndim == 1, f"eta must be 1d array."
    assert isinstance(sg, np.ndarray) and sg.ndim == 1, f"sg must be 1d array."
    assert np.min(xi) >= -1 and np.max(xi) <= 1, f"xi must be in [-1, 1]"
    assert np.min(et) >= -1 and np.max(et) <= 1, f"eta must be in [-1, 1]"
    assert np.min(sg) >= -1 and np.max(sg) <= 1, f"eta must be in [-1, 1]"
    elements = tpm.composition
    RM_u, RM_v, RM_w = {}, {}, {}
    for e in elements:
        element = elements[e]
        etype = element.etype
        if etype in ("orthogonal hexahedron", ):
            RM_u[e], RM_v[e], RM_w[e] = ___rm331_msepy_hexahedral___(element, degree, xi, et, sg)
        else:
            raise NotImplementedError()

    return RM_u, RM_v, RM_w


_cache_rm331_ = {}


def ___rm331_msepy_hexahedral___(element, degree, xi, et, sg):
    """"""
    metric_signature = element.metric_signature
    if isinstance(metric_signature, str):
        key = metric_signature + _degree_str_maker(degree)
        cached, data = ndarray_key_comparer(_cache_rm331_, [xi, et, sg], check_str=key)
        do_cache = True
    else:
        key = None
        cached, data = False, None
        do_cache = False

    if cached:
        return data
    else:
        xi_et_sg, bfs = element.bf('m3n3k1', degree, xi, et, sg)
        u, v, w = bfs
        iJ = element.ct.inverse_Jacobian_matrix(*xi_et_sg)
        iJ0, iJ1, iJ2 = iJ
        iJ00, iJ01, iJ02 = iJ0
        iJ10, iJ11, iJ12 = iJ1
        iJ20, iJ21, iJ22 = iJ2
        x0 = u * iJ00
        x1 = v * iJ10
        x2 = w * iJ20
        rm_e_x = np.vstack((x0, x1, x2)).T

        y0 = u * iJ01
        y1 = v * iJ11
        y2 = w * iJ21
        rm_e_y = np.vstack((y0, y1, y2)).T

        z0 = u * iJ02
        z1 = v * iJ12
        z2 = w * iJ22
        rm_e_z = np.vstack((z0, z1, z2)).T

        data = (rm_e_x, rm_e_y, rm_e_z)

        if do_cache:
            add_to_ndarray_cache(_cache_rm331_, [xi, et, sg], data, check_str=key)
        else:
            pass

        return data
