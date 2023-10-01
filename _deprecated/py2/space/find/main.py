# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehy.py2.space.find.local_dofs import _MseHyPy2SpaceFindLocalDofs


class MseHyPy2SpaceFind(Frozen):
    """"""

    def __init__(self, space):
        self._space = space
        self._local_dofs = _MseHyPy2SpaceFindLocalDofs(space)
        self._freeze()

    def local_dofs(self, q_or_t, edge_index, degree):
        """"""
        return self._local_dofs(q_or_t, edge_index, degree)
