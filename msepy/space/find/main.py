# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.msepy.space.find.local_dofs import _MsePySpaceFindLocalDofs


class MsePySpaceFind(Frozen):
    """"""

    def __init__(self, space):
        self._space = space
        self._local_dofs = _MsePySpaceFindLocalDofs(space)
        self._freeze()

    def local_dofs(self, m, n, degree):
        """"""
        return self._local_dofs(m, n, degree)
