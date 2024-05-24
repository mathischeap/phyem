# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from src.spaces.main import _degree_str_maker
from tools.miscellaneous.ndarray_cache import add_to_ndarray_cache, ndarray_key_comparer


def rm__m2n2k0(tpm, degree, xi, et):
    """"""
    assert isinstance(xi, np.ndarray) and xi.ndim == 1, f"xi must be 1d array."
    assert isinstance(et, np.ndarray) and et.ndim == 1, f"eta must be 1d array."
    assert np.min(xi) >= -1 and np.max(xi) <= 1, f"xi must be in [-1, 1]"
    assert np.min(et) >= -1 and np.max(et) <= 1, f"eta must be in [-1, 1]"
    elements = tpm.composition
    RM_u = {}
    for e in elements:
        element = elements[e]
        etype = element.etype
        if etype in ("orthogonal rectangle", "unique msepy curvilinear quadrilateral"):
            RM_u[e] = ___rm220_msepy_quadrilateral___(element, degree, xi, et)
        else:
            raise NotImplementedError()

    return (RM_u, )


_cache_rm220_ = {}


def ___rm220_msepy_quadrilateral___(element, degree, xi, et):
    """"""
    key = _degree_str_maker(degree)
    cached, data = ndarray_key_comparer(_cache_rm220_, [xi, et], check_str=key)

    if cached:
        return data
    else:
        _, bf = element.bf('m2n2k0', degree, xi, et)
        data = bf[0].T
        add_to_ndarray_cache(_cache_rm220_, [xi, et], data, check_str=key)
        return data
