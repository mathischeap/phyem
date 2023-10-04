# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen


class MPI_PyGM(Frozen):
    """The mpi (distributed) version of the python gathering matrix."""

    def __init__(self, *gms):
        """"""
        if len(gms) == 1:
            gm = gms[0]

            if gm.__class__ is MPI_PyGM:
                gm = gm._gm
                gms = gms
            else:
                assert isinstance(gm, dict), f"put gm in dict."
                gm = gm
                gms = [self, ]

        else:
            raise NotImplementedError()

        # -----------------------------------------------------------------------
        assert isinstance(gm, dict), f"put raw gathering matrix in a dictionary. {gm.__class__}"
        for gmi in gms:
            assert gmi.__class__ is MPI_PyGM, f"must be"
        for i in gm:
            assert isinstance(gm[i], np.ndarray) and np.ndim(gm[i]) == 1, \
                f"numbering of element #{i} must be a 1d ndarray."
        # -----------------------------------------------------------------------
        self._gm = gm
        self._gms = gms
        self._num_dofs = None
        self._num_local_dofs = dict()
        self._freeze()
