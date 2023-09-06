# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.spaces.finite import SpaceFiniteSetting
from msehy.py2.mesh.main import MseHyPy2Mesh
from msepy.space.degree import PySpaceDegree


class MseHyPy2Space(Frozen):
    """"""

    def __init__(self, abstract_space):
        """"""
        self._abstract = abstract_space
        abstract_mesh = abstract_space.mesh
        mesh = abstract_mesh._objective
        abstract_space._objective = self   # this is important, we use it for making forms.
        assert mesh.__class__ is MseHyPy2Mesh, f"mesh type {mesh} wrong."
        self._mesh = mesh
        self._finite = SpaceFiniteSetting(self)  # this is a necessary attribute for a particular space.
        self._degree_cache = {}
        self._freeze()

    @property
    def abstract(self):
        return self._abstract

    def __repr__(self):
        """repr"""
        ab_space_repr = self.abstract.__repr__().split(' at ')[0][1:]
        return '<MseHy-py2 ' + ab_space_repr + super().__repr__().split('object')[1]

    @property
    def finite(self):
        """The finite setting."""
        return self._finite

    @property
    def mesh(self):
        return self._mesh

    @property
    def esd(self):
        return self.abstract.mesh.manifold.esd

    @property
    def ndim(self):
        return self.abstract.mesh.ndim

    @property
    def m(self):
        return self.esd

    @property
    def n(self):
        return self.ndim

    @property
    def manifold(self):
        """The manifold."""
        return self._mesh.manifold

    def __getitem__(self, degree):
        """"""
        key = str(degree)
        if key in self._degree_cache:
            pass
        else:
            self._degree_cache[key] = PySpaceDegree(self, degree)
        return self._degree_cache[key]
