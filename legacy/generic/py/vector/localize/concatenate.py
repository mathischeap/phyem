# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from legacy.generic.py.vector.localize.static import Localize_Static_Vector, Localize_Static_Vector_Cochain


def concatenate(v_1d_list, gm):
    """"""
    shape = len(v_1d_list)
    gms = gm._gms
    assert len(gms) == shape, f"composite wrong."
    v = list()
    for i, vi in enumerate(v_1d_list):
        if vi is None:
            v.append(
                Localize_Static_Vector(0, gms[i])
            )
        else:
            assert vi.__class__ in (Localize_Static_Vector, Localize_Static_Vector_Cochain)
            assert vi._gm == gms[i], 'gm wrong'
            v.append(
                vi
            )

    cv = _Concatenate(v)

    return Localize_Static_Vector(cv, gm)


class _Concatenate(Frozen):
    """"""

    def __init__(self, vs):
        """"""
        self._vs = vs
        self._freeze()

    def __call__(self, i):
        """get the concatenated vector for element #i."""
        v_list = list()
        for v in self._vs:
            v_list.append(
                v[i]  # all adjustments and customizations take effect.
            )
        return np.concatenate(v_list)
