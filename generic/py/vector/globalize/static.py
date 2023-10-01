# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
import numpy as np


class Globalize_Static_Vector(Frozen):
    """"""

    def __init__(self, V, gm):
        """"""
        assert isinstance(V, np.ndarray) and np.ndim(V) == 1, f"V must be a 1-d ndarray."
        assert V.shape == (gm.num_dofs, ), f"vector shape dis-match the gathering matrices"
        self._V = V
        self._gm = gm
        self._freeze()

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        return f"<Globalize_Static_Vector of shape {self.shape}{super_repr}>"

    @property
    def shape(self):
        return self._V.shape

    @property
    def V(self):
        return self._V

    def localize(self):
        """Reshape this vector into localize version."""
        from generic.py.vector.localize.static import Localize_Static_Vector

        data_dict = dict()
        for index in self._gm:
            local_dofs = self._gm[index]
            data_dict[index] = self._V[local_dofs]

        return Localize_Static_Vector(data_dict, self._gm)
