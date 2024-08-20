# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from src.spaces.main import _degree_str_maker
from tools.miscellaneous.ndarray_cache import add_to_ndarray_cache, ndarray_key_comparer


def rm__m3n3k0(tpm, degree, xi, et, sg):
    """"""
    assert isinstance(xi, np.ndarray) and xi.ndim == 1, f"xi must be 1d array."
    assert isinstance(et, np.ndarray) and et.ndim == 1, f"eta must be 1d array."
    assert isinstance(sg, np.ndarray) and sg.ndim == 1, f"eta must be 1d array."
    assert np.min(xi) >= -1 and np.max(xi) <= 1, f"xi must be in [-1, 1]"
    assert np.min(et) >= -1 and np.max(et) <= 1, f"eta must be in [-1, 1]"
    assert np.min(sg) >= -1 and np.max(sg) <= 1, f"sg must be in [-1, 1]"
    elements = tpm.composition
    RM_u = {}
    for e in elements:
        element = elements[e]
        etype = element.etype
        if etype in ("orthogonal hexahedron", ):
            RM_u[e] = ___rm330_msepy_hexahedral___(element, degree, xi, et, sg)
        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")

    return (RM_u, )


_cache_rm330_ = {}


def ___rm330_msepy_hexahedral___(element, degree, xi, et, sg):
    """"""
    key = _degree_str_maker(degree)
    cached, data = ndarray_key_comparer(_cache_rm330_, [xi, et, sg], check_str=key)

    if cached:
        return data
    else:
        _, bf = element.bf('m3n3k0', degree, xi, et, sg)
        data = bf[0].T
        add_to_ndarray_cache(_cache_rm330_, [xi, et, sg], data, check_str=key)
        return data
