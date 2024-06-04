# -*- coding: utf-8 -*-
"""
"""
from numpy import arange


def local_numbering_Lambda__m3n3k2(etype, p):
    """Do the numbering for the outer 1-form on a 2d mesh in 2d space."""
    if etype in ('orthogonal hexahedron', ):
        local_numbering = _ln_m3n3k2_msepy_quadrilateral_(p)
    else:
        raise NotImplementedError()
    return local_numbering


_cache_332_ = {}   # this cache will not be cleaned.


def _ln_m3n3k2_msepy_quadrilateral_(p):
    """"""
    if p in _cache_332_:
        return _cache_332_[p]
    else:
        pass
    px, py, pz = p
    Px = (px + 1) * py * pz
    Py = px * (py + 1) * pz
    Pz = px * py * (pz + 1)

    # segments dx
    local_numbering_dydz = arange(0, Px).reshape((px + 1, py, pz), order='F')
    # segments dy
    local_numbering_dzdx = arange(Px, Px + Py).reshape((px, py + 1, pz), order='F')
    # segments dz
    local_numbering_dxdy = arange(Px + Py, Px + Py + Pz).reshape((px, py, pz + 1), order='F')

    _cache_332_[p] = local_numbering_dydz, local_numbering_dzdx, local_numbering_dxdy

    return local_numbering_dydz, local_numbering_dzdx, local_numbering_dxdy
