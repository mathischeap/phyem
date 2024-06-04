# -*- coding: utf-8 -*-
"""
"""
import numpy as np
from tools.frozen import Frozen
from src.config import RANK, MASTER_RANK, COMM

if RANK == MASTER_RANK:
    import pickle


class MseHtt_Form_Numeric_Export(Frozen):
    """"""

    def __init__(self, f):
        """"""
        self._f = f
        self._freeze()

    def scatter(self, filename, t=None, ddf=1):
        """export the form to a file with scatter data of coordinates and values."""

        dtype, coo, val = self._f.numeric._interpolate_(t=t, ddf=ddf, data_only=True)

        coo = COMM.gather(coo, root=MASTER_RANK)
        val = COMM.gather(val, root=MASTER_RANK)

        if RANK != MASTER_RANK:
            return
        else:
            pass

        COO = [[] for _ in range(len(coo[0]))]
        VAL = [[] for _ in range(len(val[0]))]

        for ___ in coo:
            for i, _ in enumerate(___):
                COO[i].extend(_)
        COO = [np.array(_) for _ in COO]
        coo = COO

        for ___ in val:
            for i, _ in enumerate(___):
                VAL[i].extend(_)
        VAL = [np.array(_) for _ in VAL]
        val = VAL

        to_save_dict = {
            'dtype': dtype,
            'coordinates': coo,
            'values': val
        }
        with open(filename, 'wb') as output:
            pickle.dump(to_save_dict, output, pickle.HIGHEST_PROTOCOL)
        output.close()
