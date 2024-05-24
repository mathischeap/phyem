# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehtt.static.space.gathering_matrix.main import MseHttSpaceGatheringMatrix
from msehtt.static.space.reduce.main import MseHttSpaceReduce
from msehtt.static.space.reconstruct.main import MseHttSpaceReconstruct
from msehtt.static.space.mass_matrix.main import MseHttSpaceMassMatrix
from msehtt.static.space.error.main import MseHttSpaceError
from msehtt.static.space.incidence_matrix.main import MseHttSpaceIncidenceMatrix
from msehtt.static.space.norm.main import MseHttSpaceNorm
from msehtt.static.space.reconstruction_matrix.main import MseHttSpaceReconstructionMatrix


class MseHttSpace(Frozen):
    """"""

    def __init__(self, abstract_space):
        """"""
        assert abstract_space._is_space(), f"I need a, abstract space"
        self._abstract = abstract_space
        self._tpm = None
        self._gm = None
        self._rd = None
        self._rc = None
        self._mm = None
        self._im = None
        self._error = None
        self._norm = None
        self._rm = None
        self._freeze()

    @property
    def tpm(self):
        if self._tpm is None:
            raise Exception(f"first set tpm!")
        return self._tpm

    @property
    def tgm(self):
        return self.tpm._tgm

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
    def str_indicator(self):
        """"""
        idc, m, n = self._imn_
        if idc == 'Lambda':
            k = self.abstract.k
            if m == n == 2 and k == 1:
                orientation = self.abstract.orientation
                return f"m{m}n{n}k{k}_{orientation}"
            else:
                return f"m{m}n{n}k{k}"
        else:
            raise NotImplementedError(idc)

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
    def reduce(self):
        """"""
        if self._rd is None:
            self._rd = MseHttSpaceReduce(self)
        return self._rd

    @property
    def reconstruct(self):
        """"""
        if self._rc is None:
            self._rc = MseHttSpaceReconstruct(self)
        return self._rc

    @property
    def mass_matrix(self):
        """"""
        if self._mm is None:
            self._mm = MseHttSpaceMassMatrix(self)
        return self._mm

    @property
    def error(self):
        """"""
        if self._error is None:
            self._error = MseHttSpaceError(self)
        return self._error

    @property
    def incidence_matrix(self):
        """"""
        if self._im is None:
            self._im = MseHttSpaceIncidenceMatrix(self)
        return self._im

    @property
    def norm(self):
        """"""
        if self._norm is None:
            self._norm = MseHttSpaceNorm(self)
        return self._norm

    @property
    def reconstruction_matrix(self):
        """"""
        if self._rm is None:
            self._rm = MseHttSpaceReconstructionMatrix(self)
        return self._rm
