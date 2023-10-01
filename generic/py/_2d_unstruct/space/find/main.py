# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from generic.py._2d_unstruct.space.find.local_dofs import FindLocalDofs


class Find(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._local_dofs = FindLocalDofs(space)
        self._freeze()

    @property
    def local_dofs(self):
        return self._local_dofs
