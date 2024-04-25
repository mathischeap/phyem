# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from tools.functions.region_wise_wrapper import RegionWiseFunctionWrapper
from src.spaces.operators import _d_to_vc, _d_ast_to_vc
from src.spaces.continuous.Lambda import ScalarValuedFormSpace
from src.spaces.continuous.bundle import BundleValuedFormSpace
from src.spaces.continuous.bundle_diagonal import DiagonalBundleValuedFormSpace


class MsePyContinuousForm(Frozen):
    """"""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._regions = self._f.mesh.regions
        self._field = None
        self._shape = None
        self._freeze()

    def __getitem__(self, t):
        """Return the partial functions at time `t` in all regions."""
        field_t = dict()
        for i in self._regions:
            field_t[i] = self.field[i][t]
        return MsePyContinuousFormPartialTime(self._f, field_t)

    @staticmethod
    def _is_cf():
        """A signature."""
        return True

    @property
    def field(self):
        """the cf."""
        return self._field

    @field.setter
    def field(self, _field):
        """"""
        _field = self._proceed_field(_field)
        if _field.__class__ is RegionWiseFunctionWrapper:
            self._field = _field
        else:
            self._field = RegionWiseFunctionWrapper(_field)  # initialization: dict({...})
        self._check_field()

    @property
    def shape(self):
        """The shape of this cf should be. """
        if self._shape is None:
            abstract = self._f.abstract
            space = abstract.space
            if space.__class__ is ScalarValuedFormSpace:
                m, n, k = space.m, space.n, space.k

                if k == 0 or k == n:
                    self._shape = (1, )
                else:
                    self._shape = (n, )

            elif space.__class__ is BundleValuedFormSpace:
                n, k = space.n, space.k

                if k == 0 or k == n:
                    self._shape = (n, )
                else:
                    self._shape = (n, n)
            elif space.__class__ is DiagonalBundleValuedFormSpace:

                self._shape = (1, )   # only accept scalar

            else:
                raise Exception()

        return self._shape

    def _check_field(self):
        """"""
        assert (len(self._field) == len(self._regions) and
                all([region in self._field for region in self._regions])), \
            f"cf does not cover all regions."

        abstract = self._f.abstract
        space = abstract.space
        for region in self._field:
            region_cf = self._field[region]
            assert region_cf.ndim == space.mesh.m, f"cf dimension must be equal to embedding space dimension."
            assert region_cf.shape == self.shape, \
                f"cf shape does not match! For example, set a 0-form or top-form a vector, or k-form a scalar."

    def _proceed_field(self, _field):
        """"""
        regions = self._regions
        if isinstance(_field, dict):
            pass
        elif isinstance(_field, RegionWiseFunctionWrapper):
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
            new_d_cf = RegionWiseFunctionWrapper()
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
            new_cd_cf = RegionWiseFunctionWrapper()
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
            new_cd_cf = RegionWiseFunctionWrapper()
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


class MsePyContinuousFormPartialTime(Frozen):
    """"""

    def __init__(self, rf, field_t):
        """"""
        self._f = rf
        self._field = field_t
        self._freeze()

    # noinspection PyTypeChecker
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

            if len(self._f.cf.shape) == 1:  # scalar or vector form

                num_components = len(value_region)

                if values is None:
                    values = [[] for _ in range(num_components)]
                else:
                    pass

                for c in range(num_components):
                    values[c].append(value_region[c])

            elif len(self._f.cf.shape) == 2:  # tensor

                S0, S1 = self._f.cf.shape

                if values is None:
                    values = dict()
                    for s0 in range(S0):
                        for s1 in range(S1):
                            values[(s0, s1)] = list()
                else:
                    pass

                for s0 in range(S0):
                    for s1 in range(S1):
                        values[(s0, s1)].append(
                            value_region[s0][s1]
                        )

            else:
                raise NotImplementedError('not implemented for tensor form.')

        if len(self._f.cf.shape) == 1:  # scalar or vector form
            for i, val in enumerate(values):
                values[i] = np.concatenate(val, axis=axis)

        elif len(self._f.cf.shape) == 2:  # tensor

            S0, S1 = self._f.cf.shape

            for s0 in range(S0):
                for s1 in range(S1):
                    values[(s0, s1)] = np.concatenate(
                        values[(s0, s1)],
                        axis=axis
                    )

            tensor_values = [[None for _ in range(S1)] for _ in range(S0)]

            for s0 in range(S0):
                for s1 in range(S1):
                    tensor_values[s0][s1] = values[(s0, s1)]

            values = tensor_values

        else:
            raise NotImplementedError()

        return values
