# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from src.config import RANK, MASTER_RANK, COMM


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
                assert isinstance(gm, dict), f"pls put gm in dict."
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

    @property
    def num_dofs(self):
        """the amount of total dofs across all cores."""
        if self._num_dofs is None:
            local_max = list()
            for index in self:
                local_max.append(max(self[index]))
            local_max = max(local_max)
            local_max = COMM.gather(local_max, root=MASTER_RANK)
            if RANK == MASTER_RANK:
                local_max = max(local_max) + 1
            else:
                local_max = None
            self._num_dofs = COMM.bcast(local_max, root=MASTER_RANK)
        return self._num_dofs

    def __iter__(self):
        """Iteration over all local element indices."""
        for index in self._gm:
            yield index

    def __contains__(self, index):
        """If element #index is a local element?"""
        return index in self._gm

    def __getitem__(self, index):
        """The global numbering of dos in element #index."""
        return self._gm[index]

    def num_local_dofs(self, index):
        """Num of local dofs in element #index."""
        if index not in self._num_local_dofs:
            self._num_local_dofs[index] = len(self[index])
        return self._num_local_dofs[index]
