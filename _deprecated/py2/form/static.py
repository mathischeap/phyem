# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehy.py2.form.numeric.main import MseHyPy2FormNumeric


class MseHyPy2RootFormStaticCopy(Frozen):
    """"""

    def __init__(self, rf, t, g):
        """"""
        assert rf._is_base()  # we make copy only from a base form.
        self._f = rf

        t = rf._pt(t)
        self._t = t

        g = rf._pg(g)
        self._g = g

        self._freeze()

    @staticmethod
    def _is_form_static_copy():
        """A signature."""
        return True

    @property
    def generation(self):
        """The static copy is on this mesh generation."""
        return self._g

    @property
    def current_representative(self):
        return self._f.mesh[self.generation]

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        f_repr = self._f.__repr__()
        return rf"<StaticCopy @ (t{self._t}; G[{self._g}]) of " + f_repr + super_repr

    @property
    def cochain(self):
        """"""
        return self._f.cochain[(self._t, self._g)]

    @cochain.setter
    def cochain(self, cochain):
        """"""
        self._f.cochain._set(self._t, self._g, cochain)

    def reduce(self, update_cochain=True, **kwargs):
        self._f.reduce(self._t, self._g, update_cochain=update_cochain, **kwargs)

    def reconstruct(self, *meshgrid, **kwargs):
        """reconstruct"""
        return self._f.reconstruct(self._t, self._g, *meshgrid, **kwargs)

    @property
    def visualize(self):
        """visualize"""
        return self._f.visualize[(self._t, self._g)]

    def error(self, **kwargs):
        """error"""
        return self._f.error(self._t, self._g, **kwargs)

    def norm(self, **kwargs):
        """norm"""
        return self._f.norm(self._t, self._g, **kwargs)

    @property
    def coboundary(self):
        """coboundary"""
        return self._f.coboundary[(self._t, self._g)]

    def __eq__(self, other):
        """"""
        if other.__class__ is self.__class__:

            return self._t == other._t and self._f == other._f and self._g == other._g

        else:
            return False

    @property
    def numeric(self):
        return MseHyPy2FormNumeric(self._f, self._t, self._g)

    def visualize_difference_to(self, other_t_g, density=100, magnitude=True):
        """visualize reconstruction differences: self-reconstruction - other-reconstruction."""
        self._f.cochain.visualize_difference((self._t, self._g), other_t_g, density=density, magnitude=magnitude)
