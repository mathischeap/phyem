# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from src.config import RANK, MASTER_RANK, COMM
from tools.frozen import Frozen
from tools.dds.region_wise_structured import DDSRegionWiseStructured


class MseHtt_Static_Form_Export_RWS(Frozen):
    """"""

    def __init__(self, f, t):
        """"""
        self._f = f
        self._t = t
        self._freeze()

    def __call__(self, filename, ddf=1):
        """"""
        density = int(9 * ddf)
        if density < 3:
            density = 3
        elif density > 35:
            density = 35
        else:
            pass
        linspace = np.linspace(-1, 1, density)
        xyz, value = self._f[self._t].reconstruct(linspace, linspace, ravel=False)
        XYZ = list()
        VAL = list()
        for _ in xyz:
            XYZ.append(_merge_dict_(_, root=MASTER_RANK))
        for _ in value:
            VAL.append(_merge_dict_(_, root=MASTER_RANK))

        if RANK == MASTER_RANK:
            dds = DDSRegionWiseStructured(XYZ, VAL)
            dds.saveto(filename)
        else:
            pass


def _merge_dict_(data, root=MASTER_RANK):
    """"""
    assert isinstance(data, dict)
    DATA = COMM.gather(data, root=root)
    if RANK == root:
        data = {}
        for _ in DATA:
            data.update(_)
        return data
    else:
        return None
