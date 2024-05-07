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
        self.cochain = self._f.reduce(self.cf)

    def reconstruct(self, *meshgrid, ravel=False):
        return self._f.reconstruct(self.cochain, *meshgrid, ravel=ravel)

    @property
    def cf(self):
        return self._f.cf[self._t]

    def error(self, error_type='L2'):
        """"""
        return self._f.error(self.cf, self.cochain, error_type=error_type)

    @property
    def visualize(self):
        return self._f.visualize(self._t)

    def norm(self, norm_type='L2'):
        """

        Parameters
        ----------
        norm_type :
            ``L2_norm``: ((self, self)_{tpm})**0.5

        Returns
        -------

        """
        return self._f.norm(self.cochain, norm_type=norm_type)
