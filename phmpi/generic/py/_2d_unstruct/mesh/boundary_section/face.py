# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class _MPI_PY_2d_Face(Frozen):
    """The element edge as a separate class, and we call it a Face."""

    def __init__(self, mesh, element_index, edge_index):
        """"""
        self._mesh = mesh
        self._element_index = element_index
        self._edge_index = edge_index
        self._freeze()

    def __repr__(self):
        """repr."""
        return f"<Face {self._element_index}: {self._edge_index} on {self._mesh}>"

    @property
    def metric_signature(self):
        """"""
        return self._mesh[self._element_index].metric_signature + f'->{self._edge_index}'

    @property
    def ct(self):
        return self._mesh[self._element_index].edge_ct(self._edge_index)

    def find_local_dofs_of(self, f):
        """find the local dofs of form f on this face."""
        space = f.space
        local_dofs = space.find.local_dofs(self._element_index, self._edge_index, f.degree)
        return local_dofs
