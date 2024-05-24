# -*- coding: utf-8 -*-
r"""
"""


def find_local_dofs_on_face__m2n2k0(etype, p, face_index):
    """"""
    if etype in ('unique msepy curvilinear quadrilateral', 'orthogonal rectangle'):
        local_numbering = __m2n2k0_msepy_quadrilateral_(p, face_index)
    else:
        raise NotImplementedError()
    return local_numbering


from msehtt.static.space.local_numbering.Lambda.ln_m2n2k0 import _ln_m2n2k0_msepy_quadrilateral_
_cache_220_ = {}


def __m2n2k0_msepy_quadrilateral_(p, face_index):
    """"""
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
