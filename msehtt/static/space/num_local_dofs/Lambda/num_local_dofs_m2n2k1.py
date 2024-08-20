# -*- coding: utf-8 -*-
r"""
"""


def num_local_dofs__Lambda__m2n2k1_inner(etype, p):
    """Number of local dofs for the inner 1-form on a 2d mesh in 2d space."""
    if etype in ("orthogonal rectangle", "unique msepy curvilinear quadrilateral"):
        nom_local_dofs = _num_local_dofs__m2n2k1_inner_msepy_rectangle_(p)
    else:
        raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    return nom_local_dofs


_cache_221i_ = {}   # this cache will not be cleaned.


def _num_local_dofs__m2n2k1_inner_msepy_rectangle_(p):
    """"""
    if p in _cache_221i_:
        pass
    else:
        px, py = p
        Px = px * (py + 1)
        Py = (px + 1) * py

        _cache_221i_[p] = Px + Py, (Px, Py)

    return _cache_221i_[p]


def num_local_dofs__Lambda__m2n2k1_outer(etype, p):
    """Number of local dofs for the outer 1-form on a 2d mesh in 2d space."""
    if etype in ("orthogonal rectangle", "unique msepy curvilinear quadrilateral"):
        nom_local_dofs = _num_local_dofs__m2n2k1_outer_msepy_rectangle_(p)
    else:
        raise NotImplementedError()
    return nom_local_dofs


_cache_221o_ = {}


def _num_local_dofs__m2n2k1_outer_msepy_rectangle_(p):
    """"""
    if p in _cache_221o_:
        pass
    else:
        px, py = p
        Px = px * (py + 1)
        Py = (px + 1) * py

        _cache_221o_[p] = Px + Py, (Px, Py)

    return _cache_221o_[p]
