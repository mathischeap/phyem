# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""
import numpy as np
from tools.frozen import Frozen
from src.spaces.operators import _d_to_vc, _d_ast_to_vc


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
        _field = self._proceed_field(_field)
        self._field = _FieldWrapper(_field)

    def _proceed_field(self, _field):
        """"""
        regions = self._f.mesh.regions
        if isinstance(_field, dict):
            pass
        elif isinstance(_field, _FieldWrapper):
            pass
        else:
            _fd = dict()
            for i in regions:
                _fd[i] = _field
            _field = _fd
        return _field

    def coboundary(self):
        """alias for exterior derivative"""
        return self.exterior_derivative()

    def exterior_derivative(self):
        """exterior derivative."""

        if self.field is None:
            raise Exception('No cf, set it first!')
        else:
            new_d_cf = _FieldWrapper()
            for i in self.field:
                field_i = self.field[i]
                if hasattr(field_i, "_is_time_space_func") and field_i._is_time_space_func():
                    vc_operator = self._exterior_derivative_vc_operators
                    new_d_cf[i] = getattr(field_i, vc_operator)
                else:
                    raise NotImplementedError(f"exterior_derivative")
            return new_d_cf

    def codifferential(self):
        """"""

        if self.field is None:
            raise Exception('No cf, set it first!')
        else:
            new_cd_cf = _FieldWrapper()
            for i in self.field:
                field_i = self.field[i]
                if hasattr(field_i, "_is_time_space_func") and field_i._is_time_space_func():
                    sign, cd_operator = self._codifferential_vc_operators
                    new_cf_i = getattr(field_i, cd_operator)
                    if sign == '+':
                        pass
                    elif sign == '-':
                        new_cf_i = - new_cf_i
                    else:
                        raise Exception()
                    new_cd_cf[i] = new_cf_i
                else:
                    raise NotImplementedError(f"codifferential")

            return new_cd_cf

    def time_derivative(self):
        """"""

        if self.field is None:
            raise Exception('No cf, set it first!')
        else:
            new_cd_cf = _FieldWrapper()
            for i in self.field:
                field_i = self.field[i]
                if hasattr(field_i, "_is_time_space_func") and field_i._is_time_space_func():
                    new_cd_cf[i] = field_i.time_derivative
                else:
                    raise NotImplementedError(f"codifferential")

            return new_cd_cf

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


class _FieldWrapper(dict):
    """"""

    def __neg__(self):
        new_neg_field = _FieldWrapper()
        for i in self:
            new_neg_field[i] = - self[i]
        return new_neg_field


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
