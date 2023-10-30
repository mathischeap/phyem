# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class StaticCopy(Frozen):
    """Static copy of f at time `t`"""

    def __init__(self, f, t):
        """"""
        self._f = f
        self._t = t
        self._freeze()

    @property
    def mesh(self):
        """mesh"""
        return self._f.mesh

    def __repr__(self):
        """repr."""
        super_repr = super().__repr__().split('object')[1]
        f_repr = self._f.__repr__()
        return rf"<StaticCopy @{self._t} of " + f_repr + super_repr

    @property
    def cochain(self):
        """cochain."""
        return self._f.cochain[self._t]

    @cochain.setter
    def cochain(self, cochain):
        """setting cochain of this static copy."""
        self._f.cochain._set(self._t, cochain)

    @property
    def gathering_matrix(self):
        """gathering matrix."""
        return self._f.cochain.gathering_matrix

    def reduce(self, update_cochain=True, target=None):
        """reduce."""
        return self._f.reduce(self._t, update_cochain=update_cochain, target=target)

    def reconstruct(self, xi, et, ravel=False, element_range=None):
        """reconstruct."""
        return self._f.reconstruct(self._t, xi, et, ravel=ravel, element_range=element_range)

    @property
    def incidence_matrix(self):
        """incidence matrix."""
        return self._f.incidence_matrix

    @property
    def mass_matrix(self):
        """mass matrix."""
        return self._f.mass_matrix

    def visualize(self, **kwargs):
        """visualize."""
        return self._f.visualize(t=self._t, **kwargs)

    def error(self, d=2):
        return self._f.space.error(self._f.cf, self.cochain, d=d)

    def norm(self, d=2):
        return self._f.space.norm(self.cochain, d=d)
