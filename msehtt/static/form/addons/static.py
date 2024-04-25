# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MseHttFormStaticCopy(Frozen):
    """"""

    def __init__(self, f, t):
        """"""
        self._f = f
        self._t = t
        self._field = None
        self._freeze()

    @property
    def cochain(self):
        return self._f.cochain[self._t]

    @cochain.setter
    def cochain(self, cc):
        """"""
        self._f.cochain._set(self._t, cc)

    def reduce(self):
        """"""
        self.cochain = self._f.reduce(self._f.cf, self._t)

    def reconstruct(self, *meshgrid, ravel=False):
        return self._f.reconstruct(self.cochain, *meshgrid, ravel=ravel)

    @property
    def visualize(self):
        return self._f.visualize(self._t)
