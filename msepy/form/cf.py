# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""
import numpy as np
from tools.frozen import Frozen


class MsePyContinuousForm(Frozen):
    """"""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._field = None
        self._freeze()

    def __getitem__(self, t):
        """Return the partial functions at time `t` in all regions."""
        field_t = dict()
        for i in self._f.mesh.regions:
            field_t[i] = self.field[i][t]
        return MsePyContinuousFormPartialTime(self._f, field_t)

    @property
    def field(self):
        """the cf."""
        return self._field

    @field.setter
    def field(self, _field):
        """"""
        regions = self._f.mesh.regions
        if isinstance(_field, dict):
            self._field = _field
        else:
            _fd = dict()
            for i in regions:
                _fd[i] = _field
            self._field = _fd

    @property
    def _exterior_derivative_vc_operators(self):
        """"""
        space = self._f.space.abstract
        space_indicator = space.indicator
        m, n, k = space.m, space.n, space.k
        ori = space.orientation
        return _d_to_vc(space_indicator, m, n, k, ori)

    @property
    def _codifferential_vc_operators(self):
        """"""
        space = self._f.space.abstract
        space_indicator = space.indicator
        m, n, k = space.m, space.n, space.k
        ori = space.orientation
        return _d_ast_to_vc(space_indicator, m, n, k, ori)


class MsePyContinuousFormPartialTime(Frozen):
    """"""

    def __init__(self, rf, field_t):
        """"""
        self._f = rf
        self._field = field_t
        self._freeze()

    def __call__(self, *xyz, axis=0):
        """No matter what `axis` is, in the results, also `axis` is elements-wise.

        Parameters
        ----------
        xyz
        axis :
            The element-wise data is along this axis.

        Returns
        -------

        """
        values = None
        for ri in self._field:
            elements = self._f.mesh.elements._elements_in_region(ri)
            start, end = elements
            xyz_region_wise = list()
            for coo in xyz:
                xyz_region_wise.append(
                    coo.take(indices=range(start, end), axis=axis)
                )

            func = self._field[ri]

            value_region = func(*xyz_region_wise)
            num_components = len(value_region)

            if values is None:
                values = [[] for _ in range(num_components)]
            else:
                pass

            for c in range(num_components):
                values[c].append(value_region[c])

        for i, val in enumerate(values):
            values[i] = np.concatenate(val, axis=axis)

        return values


def _d_to_vc(space_indicator, *args):
    """"""

    if space_indicator == 'Lambda':  # scalar valued form spaces.
        m, n, k, ori = args
        if m == n == 1 and k == 0:  # 0-form on 1d manifold in 1d space.
            return 'derivative'
        elif m == n == 2 and k == 0:
            if ori == 'inner':
                return 'gradient'
            elif ori == 'outer':
                return 'curl'
            else:
                raise Exception()
        elif m == n == 2 and k == 1:
            if ori == 'inner':
                return 'rot'
            elif ori == 'outer':
                return 'divergence'
            else:
                raise Exception()
        elif m == n == 3:
            if k == 0:
                return 'gradient'
            elif k == 1:
                return 'curl'
            elif k == 2:
                return 'divergence'
            else:
                raise Exception()
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()


def _d_ast_to_vc(space_indicator, *args):
    """"""
    if space_indicator == 'Lambda':  # scalar valued form spaces.
        m, n, k, ori = args
        if m == n == 1 and k == 1:  # 0-form on 1d manifold in 1d space.
            raise NotImplementedError()
        elif m == n == 2 and k == 1:
            if ori == 'inner':
                raise NotImplementedError()
            elif ori == 'outer':
                raise NotImplementedError()
            else:
                raise Exception()
        elif m == n == 2 and k == 2:
            if ori == 'inner':
                raise NotImplementedError()
            elif ori == 'outer':
                raise NotImplementedError()
            else:
                raise Exception()
        elif m == n == 3:
            if k == 1:
                raise NotImplementedError()
            elif k == 2:
                raise NotImplementedError()
            elif k == 3:
                raise NotImplementedError()
            else:
                raise Exception()
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
