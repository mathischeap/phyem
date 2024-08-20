# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from numpy import arange


def local_numbering_Lambda__m2n2k0(etype, p):
    """Do the numbering for the outer 1-form on a 2d mesh in 2d space."""
    if etype in ('unique msepy curvilinear quadrilateral', 'orthogonal rectangle'):
        local_numbering = _ln_m2n2k0_msepy_quadrilateral_(p)
    elif etype == 5:
        local_numbering = _ln_m2n2k0_vtu_5_(p)
    else:
        raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    return local_numbering


_cache_0_ = {}   # these cache will not be cleaned.


def _ln_m2n2k0_msepy_quadrilateral_(p):
    """"""
    if p in _cache_0_:
        return _cache_0_[p]
    else:
        pass
    px, py = p
    # segments perp to x-axis
    local_numbering = arange(0, (px + 1) * (py + 1)).reshape((px + 1, py + 1), order='F')
    _cache_0_[p] = local_numbering
    return local_numbering


_cache_5_ = {}


def _ln_m2n2k0_vtu_5_(p):
    """
    -----------------------> et
    |
    |     0         0         0
    |     ---------------------
    |     |         |         |
    |   1 -----------3--------- 5
    |     |         |         |
    |   2 -----------4--------- 6
    |
    v
     xi

    """
    if p in _cache_5_:
        return _cache_5_[p]
    else:
        pass
    px, py = p
    # segments perp to x-axis
    local_numbering = np.zeros((px+1, py+1), dtype=int)
    local_numbering[1:, :] = np.arange(1, px * (py+1)+1).reshape((px, py+1), order='F')
    _cache_5_[p] = local_numbering
    return local_numbering
