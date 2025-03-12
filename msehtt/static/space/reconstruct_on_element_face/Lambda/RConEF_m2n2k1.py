# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

___one___ = np.array([1])
___min___ = np.array([-1])


# =============== INNER ===========================================================

def ___RoF_m2n2k1_inner___(degree, cochain, element, face_index, xi_or_et):
    r""""""
    raise NotImplementedError()


# ==================== OUTER ========================================================
def ___RoF_m2n2k1_outer___(degree, cochain, element, face_index, xi_or_et):
    r""""""
    etype = element.etype
    if etype in (
                9,
                "orthogonal rectangle",
                "unique msepy curvilinear quadrilateral",
                'unique curvilinear quad',
    ):
        # for these element types, face index 0: North, 1: South, 2: West, 3: East.
        return ___RoF_outer_QUAD___(degree, cochain, element, face_index, xi_or_et)
    else:
        raise NotImplementedError(f"___RoF_m2n2k1_outer___ not implemented for etype={etype}")


from msehtt.static.space.reconstruct.Lambda.Rc_m2n2k1 import ___rc221o_msepy_quadrilateral___


def ___RoF_outer_QUAD___(degree, cochain, element, face_index, xi_or_et):
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

    return ___rc221o_msepy_quadrilateral___(element, degree, cochain, xi, et, ravel=True)
