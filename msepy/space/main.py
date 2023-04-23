# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
"""
import sys
if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from src.spaces.finite import SpaceFiniteSetting
from msepy.mesh.main import MsePyMesh


class MsePySpace(Frozen):
    """"""

    def __init__(self, abstract_space):
        """"""
        self._abstract = abstract_space
        abstract_mesh = abstract_space.mesh
        mesh = abstract_mesh._objective
        abstract_space._objective = self   # this is important, we use it for making forms.
        assert mesh.__class__ is MsePyMesh, f"mesh type wrong."
        self._mesh = mesh
        self._finite = SpaceFiniteSetting(self)  # this is a necessary attribute for a particular space.
        self._freeze()

    @property
    def abstract(self):
        return self._abstract

    def __repr__(self):
        """repr"""
        ab_space_repr = self.abstract.__repr__().split(' at ')[0][1:]
        repr = '<MsePy ' + ab_space_repr + super().__repr__().split('object')[1]
        return repr

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

    @property
    def finite(self):
        """The finite setting."""
        return self._finite
