# -*- coding: utf-8 -*-
r"""
"""
import numpy as np


def find_local_dofs_on_face__m3n3k1(etype, p, face_index, component_wise=False):
    """"""
    if etype in (
        'orthogonal hexahedron',
        "unique msepy curvilinear hexahedron",
    ):
        local_numbering = __m3n3k1_msepy_hexahedral_(p, face_index, component_wise=component_wise)
    else:
        raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    return local_numbering


from msehtt.static.space.local_numbering.Lambda.ln_m3n3k1 import _ln_m3n3k1_msepy_quadrilateral_
_cache_331_ln_ = {}


def __m3n3k1_msepy_hexahedral_(p, face_index, component_wise=False):
    """"""
    key = f"{p}{face_index}{component_wise}"
    if key in _cache_331_ln_:
        pass
    else:
        ln_dx, ln_dy, ln_dz = _ln_m3n3k1_msepy_quadrilateral_(p)

        if component_wise:
            raise Exception()  # should not use this for m3n3k1-form

        else:

            if face_index == 0:  # x- face
                local_numbering_d0 = ln_dy[0, :, :]
                local_numbering_d1 = ln_dz[0, :, :]
            elif face_index == 1:  # x+ face
                local_numbering_d0 = ln_dy[-1, :, :]
                local_numbering_d1 = ln_dz[-1, :, :]
            elif face_index == 2:  # y- face
                local_numbering_d0 = ln_dz[:, 0, :]
                local_numbering_d1 = ln_dx[:, 0, :]
            elif face_index == 3:  # y+ face
                local_numbering_d0 = ln_dz[:, -1, :]
                local_numbering_d1 = ln_dx[:, -1, :]
            elif face_index == 4:  # z- face
                local_numbering_d0 = ln_dx[:, :, 0]
                local_numbering_d1 = ln_dy[:, :, 0]
            elif face_index == 5:  # z+ face
                local_numbering_d0 = ln_dx[:, :, -1]
                local_numbering_d1 = ln_dy[:, :, -1]
            else:
                raise Exception()

            _cache_331_ln_[key] = np.concatenate((local_numbering_d0.ravel('F'), local_numbering_d1.ravel('F')))

    return _cache_331_ln_[key]
