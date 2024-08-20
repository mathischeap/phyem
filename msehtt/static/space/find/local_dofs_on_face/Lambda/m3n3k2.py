# -*- coding: utf-8 -*-
r"""
"""
import numpy as np


def find_local_dofs_on_face__m3n3k2(etype, p, face_index, component_wise=False):
    """"""
    if etype in ('orthogonal hexahedron', ):
        local_numbering = __m3n3k2_msepy_hexahedral_(p, face_index, component_wise=component_wise)
    else:
        raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    return local_numbering


from msehtt.static.space.local_numbering.Lambda.ln_m3n3k2 import _ln_m3n3k2_msepy_quadrilateral_
_cache_332_ = {}


def __m3n3k2_msepy_hexahedral_(p, face_index, component_wise=False):
    """"""
    key = f"{p}{face_index}{component_wise}"
    if key in _cache_332_:
        return _cache_332_[key]
    else:
        ln_dydz, ln_dzdx, ln_dxdy = _ln_m3n3k2_msepy_quadrilateral_(p)

        if face_index == 0:  # x- face
            local_numbering = ln_dydz[0, :, :]
        elif face_index == 1:  # x+ face
            local_numbering = ln_dydz[-1, :, :]
        elif face_index == 2:  # y- face
            local_numbering = ln_dzdx[:, 0, :]
        elif face_index == 3:  # y+ face
            local_numbering = ln_dzdx[:, -1, :]
        elif face_index == 4:  # z- face
            local_numbering = ln_dxdy[:, :, 0]
        elif face_index == 5:  # z+ face
            local_numbering = ln_dxdy[:, :, -1]
        else:
            raise Exception()

        if component_wise:
            if face_index in (0, 1):
                _cache_332_[key] = 0, local_numbering
            elif face_index in (2, 3):
                dydz_max = np.max(ln_dydz) + 1
                local_numbering -= dydz_max
                _cache_332_[key] = 1, local_numbering
            elif face_index in (4, 5):
                dydz_max = np.max(ln_dydz) + 1
                dzdx_max = np.max(ln_dzdx) + 1
                local_numbering -= dydz_max + dzdx_max
                _cache_332_[key] = 2, local_numbering
            else:
                pass

        else:
            _cache_332_[key] = local_numbering.ravel('F')

        return _cache_332_[key]
