# -*- coding: utf-8 -*-
r"""
"""


def num_local_dofs__Lambda__m3n3k2(etype, p):
    """Number of local dofs for the 2-form on a 3d mesh in 3d space."""
    if etype in ('orthogonal hexahedron', ):
        local_numbering = _num_local_dofs__m3n3k2_msepy_hexahedral_(p)
    else:
        raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    return local_numbering


_cache_332_ = {}   # this cache will not be cleaned.


def _num_local_dofs__m3n3k2_msepy_hexahedral_(p):
    """"""
    if p in _cache_332_:
        pass
    else:
        px, py, pz = p
        Px = (px + 1) * py * pz
        Py = px * (py + 1) * pz
        Pz = px * py * (pz + 1)

        _cache_332_[p] = Px + Py + Pz, (Px, Py, Pz)

    return _cache_332_[p]
