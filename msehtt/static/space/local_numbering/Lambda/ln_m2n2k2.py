# -*- coding: utf-8 -*-
r"""
"""
from numpy import arange


def local_numbering_Lambda__m2n2k2(etype, p):
    """Do the numbering for the outer 1-form on a 2d mesh in 2d space."""
    if etype in ('unique msepy curvilinear quadrilateral', 'orthogonal rectangle'):
        local_numbering = _ln_m2n2k2_msepy_quadrilateral_(p)
    else:
        raise NotImplementedError()
    return local_numbering


_cache_2_ = {}   # these cache will not be cleaned.


def _ln_m2n2k2_msepy_quadrilateral_(p):
    """"""
    if p in _cache_2_:
        return _cache_2_[p]
    else:
        pass
    px, py = p
    # segments perp to x-axis
    local_numbering = arange(0, px * py).reshape((px, py), order='F')
    _cache_2_[p] = local_numbering
    return local_numbering
