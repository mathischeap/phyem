# -*- coding: utf-8 -*-
"""
"""
from numpy import arange


def local_numbering_Lambda__m2n2k1_outer(etype, degree):
    """Do the numbering for the outer 1-form on a 2d mesh in 2d space."""
    if etype in ('unique msepy curvilinear quadrilateral', 'orthogonal rectangle'):
        local_numbering = ___ln_outer_msepy_quadrilateral___(degree)
    else:
        raise NotImplementedError()
    return local_numbering


_cache_1_ = {}   # these cache will not be cleaned.


def ___ln_outer_msepy_quadrilateral___(degree):
    """"""
    if isinstance(degree, int):
        px, py = degree, degree
    else:
        raise NotImplementedError(f"cannot parse degree={degree} for 2d msepy quadrilateral element.")
    if (px, py) in _cache_1_:
        return _cache_1_[(px, py)]
    else:
        pass
    Px = (px + 1) * py
    Py = px * (py + 1)
    # segments perp to x-axis
    local_numbering_dy = arange(0, Px).reshape((px + 1, py), order='F')
    # segments perp to y-axis
    local_numbering_dx = arange(Px, Px + Py).reshape((px, py + 1), order='F')
    _cache_1_[(px, py)] = local_numbering_dy, local_numbering_dx
    return local_numbering_dy, local_numbering_dx
