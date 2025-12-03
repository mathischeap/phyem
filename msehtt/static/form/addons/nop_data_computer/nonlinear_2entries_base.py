# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen


class NonlinearTwoEntriesBase(Frozen):
    r""""""

    def __init__(self, A, B):
        r""""""
        self._A = A
        self._B = B
        self._gmA = A.cochain.gathering_matrix
        self._gmB = B.cochain.gathering_matrix
        self._cache_key = None
        self._freeze()
