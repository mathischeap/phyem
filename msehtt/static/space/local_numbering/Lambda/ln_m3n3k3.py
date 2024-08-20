# -*- coding: utf-8 -*-
r"""
"""
from numpy import arange


def local_numbering_Lambda__m3n3k3(etype, p):
    """Do the numbering for the outer 1-form on a 2d mesh in 2d space."""
    if etype in ('orthogonal hexahedron', ):
        local_numbering = _ln_m3n3k3_msepy_quadrilateral_(p)
    else:
        raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    return local_numbering


_cache_333_ = {}   # this cache will not be cleaned.


def _ln_m3n3k3_msepy_quadrilateral_(p):
    """"""
    if p in _cache_333_:
        return _cache_333_[p]
    else:
        pass
    px, py, pz = p
    # segments perp to x-axis
    local_numbering = arange(0, px * py * pz).reshape((px, py, pz), order='F')
    _cache_333_[p] = local_numbering
    return local_numbering
