# -*- coding: utf-8 -*-
r"""
"""


def find_local_dofs_on_face__m3n3k0(etype, p, face_index):
    """"""
    if etype in ('orthogonal hexahedron',):
        local_numbering = __m3n3k0_msepy_hexahedral_(p, face_index)
    else:
        raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    return local_numbering


from msehtt.static.space.local_numbering.Lambda.ln_m3n3k0 import _ln_m3n3k0_msepy_quadrilateral_
_cache_330_ = {}


def __m3n3k0_msepy_hexahedral_(p, face_index):
    """"""
    key = f"{p}{face_index}"
    if key in _cache_330_:
        return _cache_330_[key]
    else:
        ln = _ln_m3n3k0_msepy_quadrilateral_(p)
        if face_index == 0:                    # x- face
            local_numbering = ln[0, :, :]
        elif face_index == 1:                  # x+ face
            local_numbering = ln[-1, :, :]
        elif face_index == 2:                  # y- face
            local_numbering = ln[:, 0, :]
        elif face_index == 3:                  # y+ face
            local_numbering = ln[:, -1, :]
        elif face_index == 4:                  # z- face,
            local_numbering = ln[:, :, 0]
        elif face_index == 5:                  # z+ face
            local_numbering = ln[:, :, -1]
        else:
            raise NotImplementedError()

        _cache_330_[key] = local_numbering.ravel('F')
        return _cache_330_[key]
