# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.tools.frozen import Frozen
from phyem.msehtt.tools.gathering_matrix import MseHttGatheringMatrix


class MseHttGlobalVectorGathered(Frozen):
    r"""We store the same complete vector in all ranks."""

    def __init__(self, V, gm=None):
        r""""""
        assert isinstance(V, np.ndarray) and np.ndim(V) == 1, f"Gathered vector must be a 1d array."

        if gm is None:
            pass
        else:
            assert gm.__class__ is MseHttGatheringMatrix, f"gathering matrix must be {MseHttGatheringMatrix}."
            assert V.shape == (gm._num_global_dofs, ), f"vector shape does not match the number of global dofs in gm."

        self._V = V
        self._dtype = 'vector-gathered'
        self._gm = gm
        self._freeze()

    @property
    def shape(self):
        r"""it must be 1d, so the shape is like (x,)."""
        return self._V.shape

    @property
    def dtype(self):
        r"""data type"""
        return self._dtype

    @property
    def V(self):
        r"""The gathered vector. A 1d array. Same in all ranks."""
        return self._V

    def __repr__(self):
        r""""""
        super_repr = super().__repr__().split(' at ')[1]
        return rf"<msehtt-Gathered-Global-Vector of shape {self.shape} at " + super_repr

    def split(self):
        r"""split self into multiple gathered global vectors according to `self._gm`.

        These vectors are put into a list. Even when it is split into one vector (so its gm only has one
        basic gm, i.e. composite == 1), this vector is put in a list as [vector, ].

        """
        assert self._gm is not None, f"I have not gathering matrix, set it first."
        if self._gm._composite == 1:
            return [self, ]
        else:
            local_dict = {}
            for e in self._gm:
                dofs = self._gm[e]
                local_dict[e] = self._V[dofs]
            all_local_value_dictS = self._gm.split(local_dict)

            split_1d_arrays = []
            for gm_i, local_dict_i in zip(self._gm._gms, all_local_value_dictS):
                _1d_array_i = gm_i.assemble(local_dict_i, mode='replace')
                split_1d_arrays.append(self.__class__(_1d_array_i, gm=gm_i))

            return split_1d_arrays
