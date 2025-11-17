# -*- coding: utf-8 -*-
r"""
"""


def find_local_dofs_on_face__m2n2k0(etype, p, face_index):
    r""""""
    if etype in (
            9,
            'orthogonal rectangle',
            'unique curvilinear quad',
            'unique msepy curvilinear quadrilateral',
    ):
        local_numbering = __m2n2k0_msepy_quadrilateral_(p, face_index)
    elif etype == 5:
        local_numbering = __m2n2k0_vtu5__(p, face_index)
    else:
        raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    return local_numbering


from phyem.msehtt.static.space.local_numbering.Lambda.ln_m2n2k0 import _ln_m2n2k0_msepy_quadrilateral_
from phyem.msehtt.static.space.local_numbering.Lambda.ln_m2n2k0 import _ln_m2n2k0_vtu_5_
_cache_220_ = {}


def __m2n2k0_msepy_quadrilateral_(p, face_index):
    r""""""
    key = f"{p}{face_index}"
    if key in _cache_220_:
        return _cache_220_[key]
    else:
        ln = _ln_m2n2k0_msepy_quadrilateral_(p)
        if face_index == 0:                    # x- face, dy edges
            local_numbering = ln[0, :].copy()
        elif face_index == 1:                  # x+ face, dy edges
            local_numbering = ln[-1, :].copy()
        elif face_index == 2:                  # y- face, dx edges
            local_numbering = ln[:, 0].copy()
        elif face_index == 3:                  # y+ face, dx edges
            local_numbering = ln[:, -1].copy()
        else:
            raise NotImplementedError()
        _cache_220_[key] = local_numbering
        return _cache_220_[key]


def __m2n2k0_vtu5__(p, face_index):
    r"""
    -----------------------> et
    |
    |
    |      0---------0---------0
    |      |         |         |
    |      |         |         |
    |      |         |         |
    | e0   1---------3---------5  e2
    |      |         |         |
    |      |         |         |
    |      |         |         |
    |      2---------4---------6
    |               e1
    v
     xi

    """
    key = f"vtu5-{p}{face_index}"
    if key in _cache_220_:
        return _cache_220_[key]
    else:
        ln = _ln_m2n2k0_vtu_5_(p)
        if face_index == 1:                    # x+ face, dy edges, South
            local_numbering = ln[-1, :].copy()
        elif face_index == 0:                  # y- face, dx edges, West
            local_numbering = ln[:, 0].copy()
        elif face_index == 2:                  # y+ face, dx edges, East
            local_numbering = ln[:, -1].copy()
        else:
            raise NotImplementedError()
        _cache_220_[key] = local_numbering
        return _cache_220_[key]
