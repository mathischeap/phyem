# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.manifold import Manifold


class MseHttManifold(Frozen):
    r""""""
    def __init__(self, abstract_manifold):
        r""""""
        assert abstract_manifold.__class__ is Manifold, f"I need an abstract manifold."
        self._abstract = abstract_manifold
        self._freeze()

    @property
    def abstract(self):
        r""""""
        return self._abstract

    def __repr__(self):
        r""""""
        super_repr = super().__repr__().split('object')[1]
        return f"<{self.__class__.__name__} " + self._abstract._sym_repr + super_repr
