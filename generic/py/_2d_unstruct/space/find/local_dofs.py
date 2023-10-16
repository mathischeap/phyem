# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class FindLocalDofs(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._indicator = space.abstract.indicator
        self._cache1_outer = {}
        self._cache1_inner = {}
        self._freeze()

    def __call__(self, element_index, edge_index, degree):
        """Find the local dofs on edge #edge_index of element.

        Parameters
        ----------
        element_index
        edge_index
        degree

        Returns
        -------

        """
        if self._indicator == 'Lambda':
            k = self._space.abstract.k
            orientation = self._space.abstract.orientation
            ele_type = self._space.mesh[element_index].type

            if k == 1:
                return getattr(self, f'_Lambda_k{k}_{orientation}')(ele_type, edge_index, degree)
            else:
                raise NotImplementedError()

        else:
            raise NotImplementedError

    def _Lambda_k1_outer(self, ele_type, edge_index, p):
        """"""
        key = (ele_type, edge_index, p)

        if key in self._cache1_outer:
            return self._cache1_outer[key]

        else:
            local_numbering = self._space.local_numbering.Lambda._k1_outer(p)[ele_type]
            dy_numbering, dx_numbering = local_numbering
            if edge_index == 0:
                self._cache1_outer[key] = dx_numbering[:, 0]
            elif edge_index == 1:
                self._cache1_outer[key] = dy_numbering[-1, :]
            elif edge_index == 2:
                self._cache1_outer[key] = dx_numbering[:, -1]
            elif edge_index == 3:
                self._cache1_outer[key] = dy_numbering[0, :]
            else:
                raise Exception()

            return self._cache1_outer[key]

    def _Lambda_k1_inner(self, ele_type, edge_index, p):
        """"""
        key = (ele_type, edge_index, p)

        if key in self._cache1_inner:
            return self._cache1_inner[key]

        else:
            local_numbering = self._space.local_numbering.Lambda._k1_inner(p)[ele_type]
            dx_numbering, dy_numbering = local_numbering
            if edge_index == 0:
                self._cache1_inner[key] = dx_numbering[:, 0]
            elif edge_index == 1:
                self._cache1_inner[key] = dy_numbering[-1, :]
            elif edge_index == 2:
                self._cache1_inner[key] = dx_numbering[:, -1]
            elif edge_index == 3:
                self._cache1_inner[key] = dy_numbering[0, :]
            else:
                raise Exception()

            return self._cache1_inner[key]
