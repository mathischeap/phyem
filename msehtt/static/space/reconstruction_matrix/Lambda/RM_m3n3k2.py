# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from src.spaces.main import _degree_str_maker
from tools.miscellaneous.ndarray_cache import add_to_ndarray_cache, ndarray_key_comparer


def rm__m3n3k2(tpm, degree, xi, et, sg):
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
            RM_u[e], RM_v[e], RM_w[e] = ___rm332_msepy_hexahedral___(element, degree, xi, et, sg)
        else:
            raise NotImplementedError()

    return RM_u, RM_v, RM_w


_cache_rm332_ = {}


def ___rm332_msepy_hexahedral___(element, degree, xi, et, sg):
    """"""
    metric_signature = element.metric_signature
    if isinstance(metric_signature, str):
        key = metric_signature + _degree_str_maker(degree)
        cached, data = ndarray_key_comparer(_cache_rm332_, [xi, et, sg], check_str=key)
        do_cache = True
    else:
        key = None
        cached, data = False, None
        do_cache = False

    if cached:
        return data
    else:
        xi_et_sg, bfs = element.bf('m3n3k2', degree, xi, et, sg)
        u, v, w = bfs
        ij = element.ct.inverse_Jacobian_matrix(*xi_et_sg)

        x0 = u * (ij[1][1]*ij[2][2] - ij[1][2]*ij[2][1])
        x1 = v * (ij[2][1]*ij[0][2] - ij[2][2]*ij[0][1])
        x2 = w * (ij[0][1]*ij[1][2] - ij[0][2]*ij[1][1])
        rm_e_x = np.vstack((x0, x1, x2)).T

        y0 = u * (ij[1][2]*ij[2][0] - ij[1][0]*ij[2][2])
        y1 = v * (ij[2][2]*ij[0][0] - ij[2][0]*ij[0][2])
        y2 = w * (ij[0][2]*ij[1][0] - ij[0][0]*ij[1][2])
        rm_e_y = np.vstack((y0, y1, y2)).T

        z0 = u * (ij[1][0]*ij[2][1] - ij[1][1]*ij[2][0])
        z1 = v * (ij[2][0]*ij[0][1] - ij[2][1]*ij[0][0])
        z2 = w * (ij[0][0]*ij[1][1] - ij[0][1]*ij[1][0])
        rm_e_z = np.vstack((z0, z1, z2)).T

        data = (rm_e_x, rm_e_y, rm_e_z)

        if do_cache:
            add_to_ndarray_cache(_cache_rm332_, [xi, et, sg], data, check_str=key)
        else:
            pass

        return data
