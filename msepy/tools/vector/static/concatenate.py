# -*- coding: utf-8 -*-
""""""
import numpy as np
from tools.frozen import Frozen
from msepy.tools.vector.static.local import MsePyStaticLocalVector


def concatenate(v_1d_list, gm):
    """"""
    shape = len(v_1d_list)
    gms = gm._gms
    assert len(gms) == shape, f"composite wrong."
    v = list()
    for i, vi in enumerate(v_1d_list):
        if vi is None:
            v.append(
                MsePyStaticLocalVector(0, gms[i])
            )
        else:
            assert issubclass(vi.__class__, MsePyStaticLocalVector) and vi._gm is gms[i], f"gm wrong!"
            v.append(
                vi
            )

    cv = _MsePyStaticLocalVectorConcatenate(v)

    return MsePyStaticLocalVector(cv, gm)


class _MsePyStaticLocalVectorConcatenate(Frozen):
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
