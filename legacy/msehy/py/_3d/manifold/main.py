# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msepy.main import base as msepy_base


class MseHyPy3Manifold(Frozen):
    """"""

    def __init__(self, abstract_manifold):
        self._abstract = abstract_manifold
        self._freeze()

    @property
    def abstract(self):
        return self._abstract

    @property
    def background(self):
        """We return it in realtime."""
        return msepy_base['manifolds'][self._abstract._sym_repr]

    def __repr__(self):
        super_repr = super().__repr__().split('object')[1]
        return f"<{self.__class__.__name__} " + self._abstract._sym_repr + super_repr

    @property
    def visualize(self):
        """"""
        return self.background.visualize
