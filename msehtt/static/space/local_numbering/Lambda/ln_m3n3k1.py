# -*- coding: utf-8 -*-
"""
"""
from numpy import arange


def local_numbering_Lambda__m3n3k1(etype, p):
    """Do the numbering for the outer 1-form on a 2d mesh in 2d space."""
    if etype in ('orthogonal hexahedron', ):
        local_numbering = _ln_m3n3k1_msepy_quadrilateral_(p)
    else:
        raise NotImplementedError()
    return local_numbering


_cache_331_ = {}   # this cache will not be cleaned.


def _ln_m3n3k1_msepy_quadrilateral_(p):
    """"""
    if p in _cache_331_:
        return _cache_331_[p]
    else:
        px, py, pz = p
        Px = px * (py + 1) * (pz + 1)
        Py = (px + 1) * py * (pz + 1)
        Pz = (px + 1) * (py + 1) * pz

        # segments dx
        local_numbering_dx = arange(0, Px).reshape((px, py + 1, pz + 1), order='F')
        # segments dy
        local_numbering_dy = arange(Px, Px + Py).reshape((px + 1, py, pz + 1), order='F')
        # segments dz
        local_numbering_dz = arange(Px + Py, Px + Py + Pz).reshape((px + 1, py + 1, pz), order='F')

        _cache_331_[p] = local_numbering_dx, local_numbering_dy, local_numbering_dz

        return local_numbering_dx, local_numbering_dy, local_numbering_dz
