# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msepy.main import base as msepy_base
from src.config import RANK, MASTER_RANK
from tools.void import VoidClass


class MPI_MseHy_Py2_Manifold(Frozen):
    """"""

    def __init__(self, abstract_manifold):
        """"""
        self._abstract = abstract_manifold
        self._freeze()

    @property
    def abstract(self):
        return self._abstract

    @property
    def background(self):
        """We return it in realtime."""
        if RANK == MASTER_RANK:
            return msepy_base['manifolds'][self._abstract._sym_repr]
        else:
            return None

    def __repr__(self):
        super_repr = super().__repr__().split('object')[1]
        return f"<{self.__class__.__name__} " + self._abstract._sym_repr + super_repr

    @property
    def visualize(self):
        """"""
        if RANK == MASTER_RANK:
            return self.background.visualize
        else:
            return VoidClass()
