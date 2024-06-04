# -*- coding: utf-8 -*-
r"""
"""
from numpy import arange


def local_numbering_Lambda__m3n3k0(etype, p):
    """Do the numbering for the outer 1-form on a 2d mesh in 2d space."""
    if etype in ('orthogonal hexahedron', ):
        local_numbering = _ln_m3n3k0_msepy_quadrilateral_(p)
    else:
        raise NotImplementedError()
    return local_numbering


_cache_330_ = {}   # this cache will not be cleaned.


def _ln_m3n3k0_msepy_quadrilateral_(p):
    """"""
    if p in _cache_330_:
        return _cache_330_[p]
    else:
        pass
    px, py, pz = p
    # segments perp to x-axis
    local_numbering = arange(0, (px + 1) * (py + 1) * (pz + 1)).reshape((px + 1, py + 1, pz + 1), order='F')
    _cache_330_[p] = local_numbering
    return local_numbering
