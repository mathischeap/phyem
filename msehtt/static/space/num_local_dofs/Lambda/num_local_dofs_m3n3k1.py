# -*- coding: utf-8 -*-
r"""
"""


def num_local_dofs__Lambda__m3n3k1(etype, p):
    """Number of local dofs for the 1-form on a 3d mesh in 3d space."""
    if etype in ('orthogonal hexahedron', ):
        local_numbering = _num_local_dofs__m3n3k1_msepy_hexahedral_(p)
    else:
        raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    return local_numbering


_cache_331_ = {}   # this cache will not be cleaned.


def _num_local_dofs__m3n3k1_msepy_hexahedral_(p):
    """"""
    if p in _cache_331_:
        pass
    else:
        px, py, pz = p
        Px = px * (py + 1) * (pz + 1)
        Py = (px + 1) * py * (pz + 1)
        Pz = (px + 1) * (py + 1) * pz

        _cache_331_[p] = Px + Py + Pz, (Px, Py, Pz)

    return _cache_331_[p]
