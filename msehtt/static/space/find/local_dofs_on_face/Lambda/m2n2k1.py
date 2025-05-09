# -*- coding: utf-8 -*-
r"""
"""
import numpy as np


# ---------------- OUTER ------------------------------------------------------------------------------


def find_local_dofs_on_face__m2n2k1_outer(etype, p, face_index, component_wise=False):
    r""""""
    if etype in ('unique msepy curvilinear quadrilateral', 'orthogonal rectangle',
                 9, 'unique curvilinear quad'):
        local_numbering = __m2n2k1_outer_msepy_quadrilateral_(p, face_index, component_wise=component_wise)

    elif etype in (5, "unique msepy curvilinear triangle"):
        local_numbering = __m2n2k1_outer_vtu_5__(p, face_index, component_wise=component_wise)

    else:
        raise NotImplementedError(f"{__name__} not implemented for etype={etype}")

    return local_numbering


from msehtt.static.space.local_numbering.Lambda.ln_m2n2k1 import _ln_m2n2k1_outer_msepy_quadrilateral_
_cache_221o_ = {}


def __m2n2k1_outer_msepy_quadrilateral_(p, face_index, component_wise=False):
    r""""""
    key = f"{p}{face_index}{component_wise}"
    if key in _cache_221o_:
        return _cache_221o_[key]
    else:
        ln_dy, ln_dx = _ln_m2n2k1_outer_msepy_quadrilateral_(p)

        if face_index == 0:                    # x- face, dy edges
            local_numbering = ln_dy[0, :].copy()
        elif face_index == 1:                  # x+ face, dy edges
            local_numbering = ln_dy[-1, :].copy()
        elif face_index == 2:                  # y- face, dx edges
            local_numbering = ln_dx[:, 0].copy()
        elif face_index == 3:                  # y+ face, dx edges
            local_numbering = ln_dx[:, -1].copy()
        else:
            raise NotImplementedError()

        if component_wise:
            if face_index in (2, 3):
                dy_max = np.max(ln_dy) + 1
                local_numbering -= dy_max
                _cache_221o_[key] = 1, local_numbering
            else:
                _cache_221o_[key] = 0, local_numbering
        else:
            _cache_221o_[key] = local_numbering

        return _cache_221o_[key]


from msehtt.static.space.local_numbering.Lambda.ln_m2n2k1 import _ln_m2n2k1_outer_vtu5_
_cache_221o_vtu5_ = {}


def __m2n2k1_outer_vtu_5__(p, face_index, component_wise=False):
    r"""
    -----------------------> et
    |
    |
    |      ---------------------
    |      |         |         |
    |      4         6         8
    |      |         |         |
    |  e0  -----0---------2-----  e2
    |      |         |         |
    |      5         7         9
    |      |         |         |
    |      -----1---------3-----
    |               e1
    v
     xi

    """
    key = f"{p}{face_index}{component_wise}"

    if key in _cache_221o_vtu5_:
        return _cache_221o_vtu5_[key]

    else:
        ln_dy, ln_dx = _ln_m2n2k1_outer_vtu5_(p)

        if face_index == 1:                          # x+ face, dy edges, South face
            local_numbering = ln_dy[-1, :].copy()
        elif face_index == 0:                        # y- face, dx edges, West face
            local_numbering = ln_dx[:, 0].copy()
        elif face_index == 2:                        # y+ face, dx edges, East face
            local_numbering = ln_dx[:, -1].copy()
        else:
            raise NotImplementedError()

        if component_wise:
            if face_index in (0, 2):    # the dofs of the first component
                dy_max = np.max(ln_dy) + 1
                local_numbering -= dy_max
                _cache_221o_vtu5_[key] = 1, local_numbering  # the dofs of the first component
            else:
                _cache_221o_vtu5_[key] = 0, local_numbering  # the dofs of the 0th component
        else:
            _cache_221o_vtu5_[key] = local_numbering

        return _cache_221o_vtu5_[key]


# ---------------- INNER ------------------------------------------------------------------------------


def find_local_dofs_on_face__m2n2k1_inner(etype, p, face_index, component_wise=False):
    r""""""
    if etype in ('unique msepy curvilinear quadrilateral', 'orthogonal rectangle',
                 9, 'unique curvilinear quad'):
        local_numbering = __m2n2k1_inner_msepy_quadrilateral_(p, face_index, component_wise=component_wise)

    elif etype in (5, "unique msepy curvilinear triangle"):
        local_numbering = __m2n2k1_inner_vtu_5__(p, face_index, component_wise=component_wise)

    else:
        raise NotImplementedError(f"{__name__} not implemented for etype={etype}")

    return local_numbering


from msehtt.static.space.local_numbering.Lambda.ln_m2n2k1 import _ln_m2n2k1_inner_msepy_quadrilateral_
_cache_221i_ = {}


def __m2n2k1_inner_msepy_quadrilateral_(p, face_index, component_wise=False):
    r""""""
    key = f"{p}{face_index}{component_wise}"
    if key in _cache_221i_:
        return _cache_221i_[key]
    else:
        ln_dx, ln_dy = _ln_m2n2k1_inner_msepy_quadrilateral_(p)

        if face_index == 0:                         # x- face, dy edges
            local_numbering = ln_dy[0, :].copy()
        elif face_index == 1:                       # x+ face, dy edges
            local_numbering = ln_dy[-1, :].copy()
        elif face_index == 2:                       # y- face, dx edges
            local_numbering = ln_dx[:, 0].copy()
        elif face_index == 3:                       # y+ face, dx edges
            local_numbering = ln_dx[:, -1].copy()
        else:
            raise NotImplementedError()

        if component_wise:
            if face_index in (2, 3):
                _cache_221i_[key] = 0, local_numbering
            else:
                dx_max = np.max(ln_dx) + 1
                local_numbering -= dx_max
                _cache_221i_[key] = 1, local_numbering
        else:
            _cache_221i_[key] = local_numbering

        return _cache_221i_[key]


from msehtt.static.space.local_numbering.Lambda.ln_m2n2k1 import _ln_m2n2k1_inner_vtu5_
_cache_221i_vtu5_ = {}


def __m2n2k1_inner_vtu_5__(p, face_index, component_wise=False):
    r"""

    -----------------------> et
    |
    |
    |     ---------------------
    |     |         |         |
    |     0         2         4
    |     |         |         |
    | e0  -----6---------8-----  e2
    |     |         |         |
    |     1         3         5
    |     |         |         |
    |     -----7---------9-----
    |              e1
    v
     xi

    """
    key = f"{p}{face_index}{component_wise}"
    if key in _cache_221i_vtu5_:
        return _cache_221i_vtu5_[key]
    else:
        ln_dx, ln_dy = _ln_m2n2k1_inner_vtu5_(p)

        if face_index == 1:                         # x+ face, dy edges, South
            local_numbering = ln_dy[-1, :].copy()
        elif face_index == 0:                       # y- face, dx edges, West
            local_numbering = ln_dx[:, 0].copy()
        elif face_index == 2:                       # y+ face, dx edges, East
            local_numbering = ln_dx[:, -1].copy()
        else:
            raise NotImplementedError()

        if component_wise:
            if face_index in (0, 2):
                _cache_221i_vtu5_[key] = 0, local_numbering
            else:
                dx_max = np.max(ln_dx) + 1
                local_numbering -= dx_max
                _cache_221i_vtu5_[key] = 1, local_numbering
        else:
            _cache_221i_vtu5_[key] = local_numbering

        return _cache_221i_vtu5_[key]
