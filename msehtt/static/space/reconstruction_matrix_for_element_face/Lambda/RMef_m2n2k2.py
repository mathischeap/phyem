# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

___one___ = np.array([1])
___min___ = np.array([-1])


def ___RMef_m2n2k2___(degree, element, face_index, xi_or_et):
    r""""""
    etype = element.etype
    if etype in ("orthogonal rectangle", ):
        # for these element types: face index 0: North, 1: South, 2: West, 3: East.
        return ___rMef222_msepy_quadrilateral___(degree, element, face_index, xi_or_et)
    else:
        raise NotImplementedError()


from msehtt.static.space.reconstruction_matrix.Lambda.RM_m2n2k2 import ___rm222_msepy_quadrilateral___


def ___rMef222_msepy_quadrilateral___(degree, element, face_index, xi_or_et):
    r""""""
    assert np.ndim(xi_or_et) == 1, f"I must receive 1d coo data."
    if face_index == 0:  # north face
        xi = ___min___
        et = xi_or_et
    elif face_index == 1:  # south face
        xi = ___one___
        et = xi_or_et
    elif face_index == 2:  # west face
        xi = xi_or_et
        et = ___min___
    elif face_index == 3:  # east face
        xi = xi_or_et
        et = ___one___
    else:
        raise Exception

    return ___rm222_msepy_quadrilateral___(element, degree, xi, et)
