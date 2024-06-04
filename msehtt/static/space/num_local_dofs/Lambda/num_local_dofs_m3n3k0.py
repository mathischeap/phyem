# -*- coding: utf-8 -*-
"""
"""


def num_local_dofs__Lambda__m3n3k0(etype, p):
    """Number of local dofs for the 0-form on a 3d mesh in 3d space."""
    if etype in ('orthogonal hexahedron', ):
        local_numbering = _num_local_dofs__m3n3k0_msepy_hexahedral_(p)
    else:
        raise NotImplementedError()
    return local_numbering


_cache_330_ = {}   # this cache will not be cleaned.


def _num_local_dofs__m3n3k0_msepy_hexahedral_(p):
    """"""
    if p in _cache_330_:
        pass
    else:
        px, py, pz = p
        Px = (px + 1) * (py + 1) * (pz + 1)

        _cache_330_[p] = Px, (Px, )

    return _cache_330_[p]
