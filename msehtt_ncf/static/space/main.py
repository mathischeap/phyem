# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MseHtt_NCF_Static_Space(Frozen):
    r""""""

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
        self._ref = None
        self._int_mat_over_sub_geo = None
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
    def d_space_str_indicator(self):
        """"""
        _, m, n = self._imn_
        indicator = self.str_indicator

        if m == n == 3:
            if indicator == 'm3n3k0':
                return 'm3n3k1'
            elif indicator == 'm3n3k1':
                return 'm3n3k2'
            elif indicator == 'm3n3k2':
                return 'm3n3k3'
            else:
                raise NotImplementedError(f"m3n3 for what? {indicator}")
        else:
            raise NotImplementedError(indicator)

    @property
    def orientation(self):
        """The orientation I am."""
        return self.abstract.orientation

    def __repr__(self):
        """repr"""
        ab_space_repr = self.abstract.__repr__().split(' at ')[0][1:]
        return '<MseHtt ' + ab_space_repr + super().__repr__().split('object')[1]
