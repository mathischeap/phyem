# -*- coding: utf-8 -*-
r"""
"""
import numpy as np


# ============ INNER ==========================================================
def ___RMef_m2n2k1_inner___(degree, element, face_index, xi_or_et):
    r""""""
    etype = element.etype
    if etype in ("orthogonal rectangle", ):
        # for these element types: face index 0: North, 1: South, 2: West, 3: East.
        return ___rMef_inner_msepy_quadrilateral___(degree, element, face_index, xi_or_et)
    else:
        raise NotImplementedError()


def ___rMef_inner_msepy_quadrilateral___(degree, element, face_index, xi_or_et):
    r""""""
    raise NotImplementedError()


# ============ OUTER =========================================================
def ___RMef_m2n2k1_outer___(degree, element, face_index, xi_or_et):
    r""""""
    etype = element.etype
    if etype in ("orthogonal rectangle", ):
        # for these element types: face index 0: North, 1: South, 2: West, 3: East.
        return ___rMef_outer_msepy_quadrilateral___(degree, element, face_index, xi_or_et)
    elif etype == 9:
        return ___rMef_outer_9___(degree, element, face_index, xi_or_et)
    else:
        raise NotImplementedError(f"___ref_m2n2k1_outer___ not implemented for etype={etype}")


from phyem.msehtt.static.space.reconstruction_matrix.Lambda.RM_m2n2k1 import ___rm221o_quad_9___


def ___rMef_outer_msepy_quadrilateral___(degree, element, face_index, xi_or_et):
    r""""""
    raise NotImplementedError()


def ___rMef_outer_9___(degree, element, face_index, xi_or_et):
    r""""""
    raise Exception()
