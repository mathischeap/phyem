# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msepy.form.cochain.vector.static import MsePyRootFormStaticCochainVector


class MsePyRootFormStaticCopy(Frozen):
    """"""

    def __init__(self, rf, t):
        """"""
        assert rf._is_base()  # we make copy only from a base form.
        self._f = rf
        self._t = t
        self._freeze()

    @property
    def mesh(self):
        return self._f.mesh

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        f_repr = self._f.__repr__()
        return rf"<StaticCopy @{self._t} of " + f_repr + super_repr

    @property
    def cochain(self):
        """"""
        return self._f.cochain[self._t]

    @cochain.setter
    def cochain(self, cochain):
        """"""
        self._f.cochain._set(self._t, cochain)

    @property
    def vec(self):
        """The vector of dofs (cochain) of the form at time `t`, \vec{f}^{t}."""
        gm = self._f.cochain.gathering_matrix
        if self._t in self._f.cochain:
            local = self._f.cochain[self._t].local
            # it is a separate object
            return MsePyRootFormStaticCochainVector(self._f, self._t, local, gm)
        else:
            # it is a separate object
            return MsePyRootFormStaticCochainVector(self._f, self._t, None, gm)

    def reduce(self, update_cochain=True, **kwargs):
        self._f.reduce(self._t, update_cochain=update_cochain, **kwargs)

    def reconstruct(self, *meshgrid, **kwargs):
        """reconstruct"""
        return self._f.reconstruct(self._t, *meshgrid, **kwargs)

    @property
    def visualize(self):
        """visualize"""
        return self._f.visualize[self._t]

    def error(self, **kwargs):
        """error"""
        return self._f.error(self._t, **kwargs)

    def norm(self, **kwargs):
        """norm"""
        return self._f.norm(self._t, **kwargs)

    def inner_product(self, other_form_static_copy):
        """"""
        assert other_form_static_copy.__class__ is self.__class__, f"other must be a msepy form static copy."
        other_f = other_form_static_copy._f
        other_t = other_form_static_copy._t
        return self._f.inner_product(self._t, other_f, other_t)

    @property
    def coboundary(self):
        """coboundary"""
        return self._f.coboundary[self._t]

    def __eq__(self, other):
        """"""
        if other.__class__ is self.__class__:

            return self._t == other._t and self._f == other._f

        else:
            return False
