# -*- coding: utf-8 -*-
r"""
"""
from numpy import arange


# -------------- OUTER ------------------------------------------------------------------------


def local_numbering_Lambda__m2n2k1_outer(etype, p):
    r"""Do the numbering for the outer 1-form on a 2d mesh in 2d space."""
    if etype in ('unique msepy curvilinear quadrilateral', 'orthogonal rectangle'):
        local_numbering = _ln_m2n2k1_outer_msepy_quadrilateral_(p)
    elif etype == 5:
        local_numbering = _ln_m2n2k1_outer_vtu5_(p)
    else:
        raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    return local_numbering


_cache_1_ = {}   # these cache will not be cleaned.


def _ln_m2n2k1_outer_msepy_quadrilateral_(p):
    r""""""
    if p in _cache_1_:
        return _cache_1_[p]
    else:
        pass
    px, py = p
    Px = (px + 1) * py
    Py = px * (py + 1)
    # segments perp to x-axis
    local_numbering_dy = arange(0, Px).reshape((px + 1, py), order='F')
    # segments perp to y-axis
    local_numbering_dx = arange(Px, Px + Py).reshape((px, py + 1), order='F')
    _cache_1_[p] = local_numbering_dy, local_numbering_dx
    return local_numbering_dy, local_numbering_dx


_cache_outer_vtu5_ = {}


def _ln_m2n2k1_outer_vtu5_(p):
    r"""
    -----------------------> et
    |
    |
    |     ---------------------
    |     |         |         |
    |     4         6         8
    |     |         |         |
    |     -----0---------2-----
    |     |         |         |
    |     5         7         9
    |     |         |         |
    |     -----1----- ---3-----
    |
    v
     xi

    Parameters
    ----------
    p

    Returns
    -------

    """
    if p in _cache_outer_vtu5_:
        return _cache_outer_vtu5_[p]
    else:
        pass

    px, py = p
    Px = px * py
    Py = px * (py + 1)
    # segments perp to x-axis
    local_numbering_dy = arange(0, Px).reshape((px, py), order='F')
    # segments perp to y-axis
    local_numbering_dx = arange(Px, Px + Py).reshape((px, py + 1), order='F')
    _cache_outer_vtu5_[p] = local_numbering_dy, local_numbering_dx
    return local_numbering_dy, local_numbering_dx


# ------------ INNER -------------------------------------------------------------------------


def local_numbering_Lambda__m2n2k1_inner(etype, p):
    r"""Do the numbering for the outer 1-form on a 2d mesh in 2d space."""
    if etype in ('unique msepy curvilinear quadrilateral', 'orthogonal rectangle'):
        local_numbering = _ln_m2n2k1_inner_msepy_quadrilateral_(p)
    elif etype == 5:
        local_numbering = _ln_m2n2k1_inner_vtu5_(p)
    else:
        raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    return local_numbering


_cache_2_ = {}   # these cache will not be cleaned.


def _ln_m2n2k1_inner_msepy_quadrilateral_(p):
    r""""""
    if p in _cache_2_:
        return _cache_2_[p]
    else:
        pass
    px, py = p
    Px = px * (py + 1)
    Py = (px + 1) * py
    # segments perp to x-axis
    local_numbering_dx = arange(0, Px).reshape((px, py + 1), order='F')
    # segments perp to y-axis
    local_numbering_dy = arange(Px, Px + Py).reshape((px + 1, py), order='F')
    _cache_2_[p] = local_numbering_dx, local_numbering_dy
    return local_numbering_dx, local_numbering_dy


_cache_inner_vtu5_ = {}


def _ln_m2n2k1_inner_vtu5_(p):
    r"""
    -----------------------> et
    |
    |
    |     ---------------------
    |     |         |         |
    |     0         2         4
    |     |         |         |
    |     -----6---------8-----
    |     |         |         |
    |     1         3         5
    |     |         |         |
    |     -----7----- ---9-----
    |
    v
     xi

    """
    if p in _cache_inner_vtu5_:
        return _cache_inner_vtu5_[p]
    else:
        pass
    px, py = p
    Px = px * (py + 1)
    Py = px * py
    # segments perp to x-axis
    local_numbering_dx = arange(0, Px).reshape((px, py + 1), order='F')
    # segments perp to y-axis
    local_numbering_dy = arange(Px, Px + Py).reshape((px, py), order='F')
    _cache_inner_vtu5_[p] = local_numbering_dx, local_numbering_dy
    return local_numbering_dx, local_numbering_dy
