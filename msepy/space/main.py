# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
"""
import sys
if './' not in sys.path:
    sys.path.append('./')
from src.tools.frozen import Frozen
from msepy.mesh.main import MsePyMesh


class MsePySpace(Frozen):
    """"""

    def __init__(self, abstract_space):
        """"""
        self._abstract = abstract_space
        abstract_mesh = abstract_space.mesh
        mesh = abstract_mesh._objective
        assert mesh.__class__ is MsePyMesh, f"mesh type wrong."
        self._mesh = mesh

        self._freeze()

    @property
    def abstract(self):
        return self._abstract

    @property
    def esd(self):
        return self.abstract.mesh.manifold.esd

    @property
    def ndim(self):
        return self.abstract.mesh.ndim

    @property
    def mesh(self):
        """The mesh"""
        return self._mesh

    @property
    def manifold(self):
        """The manifold."""
        return self._mesh.manifold
