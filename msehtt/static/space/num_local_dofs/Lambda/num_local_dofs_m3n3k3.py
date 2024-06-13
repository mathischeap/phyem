# -*- coding: utf-8 -*-
r"""
"""


def num_local_dofs__Lambda__m3n3k3(etype, p):
    """Number of local dofs for the 3-form on a 3d mesh in 3d space."""
    if etype in ('orthogonal hexahedron', ):
        local_numbering = _num_local_dofs__m3n3k3_msepy_hexahedral_(p)
    else:
        raise NotImplementedError()
    return local_numbering


_cache_333_ = {}   # this cache will not be cleaned.


def _num_local_dofs__m3n3k3_msepy_hexahedral_(p):
    """"""
    if p in _cache_333_:
        pass
    else:
        px, py, pz = p
        Px = px * py * pz

        _cache_333_[p] = Px, (Px, )

    return _cache_333_[p]
