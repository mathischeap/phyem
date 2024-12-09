# -*- coding: utf-8 -*-
r"""
"""
import numpy as np


# ============ INNER ==========================================================
def ___ref_m2n2k1_inner___(degree, element, face_index, xi_or_et):
    r""""""
    etype = element.etype
    if etype in ("orthogonal rectangle", ):
        # for these element types: face index 0: North, 1: South, 2: West, 3: East.
        return ___ref_inner_msepy_quadrilateral___(degree, element, face_index, xi_or_et)
    else:
        raise NotImplementedError()


def ___ref_inner_msepy_quadrilateral___(degree, element, face_index, xi_or_et):
    r""""""
    raise NotImplementedError()


# ============ OUTER =========================================================
def ___ref_m2n2k1_outer___(degree, element, face_index, xi_or_et):
    r""""""
    etype = element.etype
    if etype in ("orthogonal rectangle", ):
        # for these element types: face index 0: North, 1: South, 2: West, 3: East.
        return ___ref_outer_msepy_quadrilateral___(degree, element, face_index, xi_or_et)
    else:
        raise NotImplementedError()


def ___ref_outer_msepy_quadrilateral___(degree, element, face_index, xi_or_et):
    r""""""
    raise NotImplementedError()
