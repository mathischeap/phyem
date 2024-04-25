# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehtt.static.space.gathering_matrix.main import MseHttSpaceGatheringMatrix
from msehtt.static.space.local_numbering.main import MseHttSpaceLocalNumbering
from msehtt.static.space.reduce.main import MseHttSpaceReduce
from msehtt.static.space.reconstruct.main import MseHttSpaceReconstruct


class MseHttSpace(Frozen):
    """"""

    def __init__(self, abstract_space):
        """"""
        assert abstract_space._is_space(), f"I need a, abstract space"
        self._abstract = abstract_space
        self._gm = None
        self._ln = None
        self._rd = None
        self._rc = None
        self._freeze()

    @property
    def abstract(self):
        """The abstract space of me."""
        return self._abstract

    @property
    def indicator(self):
        """The indicator showing what type of space I am."""
        return self.abstract.indicator

    @property
    def m(self):
        """the dimensions of the space I am living in."""
        return self.abstract.m

    @property
    def n(self):
        """the dimensions of the mesh I am living in."""
        return self.abstract.n

    @property
    def _imn_(self):
        """"""
        return self.indicator, self.m, self.n

    @property
    def orientation(self):
        """The orientation I am."""
        return self.abstract.orientation

    def __repr__(self):
        """repr"""
        ab_space_repr = self.abstract.__repr__().split(' at ')[0][1:]
        return '<MseHtt ' + ab_space_repr + super().__repr__().split('object')[1]

    @property
    def gathering_matrix(self):
        """"""
        if self._gm is None:
            self._gm = MseHttSpaceGatheringMatrix(self)
        return self._gm

    @property
    def local_numbering(self):
        """local numbering property"""
        if self._ln is None:
            self._ln = MseHttSpaceLocalNumbering(self)
        return self._ln

    @property
    def reduce(self):
        if self._rd is None:
            self._rd = MseHttSpaceReduce(self)
        return self._rd

    @property
    def reconstruct(self):
        if self._rc is None:
            self._rc = MseHttSpaceReconstruct(self)
        return self._rc
