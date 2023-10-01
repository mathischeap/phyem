# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from msehy.tools.vector.static.assemble import IrregularStaticLocalVectorAssemble


class IrregularStaticLocalVector(Frozen):
    """"""
    def __init__(self, irregular_data, igm):
        """"""
        self._gm = igm
        self._set_data(irregular_data)
        self._adjust = _IrregularLocalVectorAdjust(self)
        self._customize = _IrregularLocalVectorCustomize(self)
        self._assemble = None
        self._freeze()

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        return rf"Irregular Static Local Vector: {self._dtype}" + super_repr

    def _set_data(self, irregular_data):
        """"""
        if irregular_data == 0:
            _0_cache_ = dict()
            _irregular_data = dict()
            for i in self._gm:
                gm_i = self._gm[i]

                len_gm_i = len(gm_i)
                if len_gm_i in _0_cache_:
                    _ = _0_cache_[len_gm_i]
                else:
                    _ = np.zeros(len_gm_i)
                    _0_cache_[len_gm_i] = _

                _irregular_data[i] = _

            irregular_data = _irregular_data
        else:
            pass

        if irregular_data is None:   # used to receive data or as unknown vector later.
            self._dtype = 'None'
            self._data = None

        elif isinstance(irregular_data, dict):
            self._dtype = "dict"
            assert len(irregular_data) == len(self._gm)
            for i in irregular_data:
                assert isinstance(irregular_data[i], np.ndarray), \
                    f"irregular_data[{i}]={irregular_data[i].__class__} is not an array"
                assert irregular_data[i].shape == (len(self._gm[i]), ), \
                    (f"shape for cell #{i}: {irregular_data[i].shape} is wrong, "
                     f"must be a ndarray of shape ({len(self._gm[i])}, ).")
            self._data = irregular_data

        elif callable(irregular_data):
            self._dtype = "callable"
            self._data_caller = irregular_data
            self._data = None

        else:
            raise Exception(f"msepy static local vector data type wrong.")

    @property
    def data(self):
        """Let the data ready as a 2-d array.

        All adjustments and customizations take effects.

        Must return a 2-d array data.
        """
        if len(self._adjust) == len(self._customize) == 0:
            # will raise Error when data is callable (generated in realtime) or None.
            if self._dtype == 'callable':
                # collect all callable data here
                raise NotImplementedError(f"callable data is not ready")

            elif self._dtype == 'None':
                raise Exception(f"None type vector has no data, set it first!")

            else:
                return self._data
        else:
            raise NotImplementedError()

    def split(self, data=None):
        """split the data according to `self._gm._gms` (chained gathering matrix).

        """
        if data is None:
            data = self.data
        else:
            pass

        gm = self._gm
        sub_gms = gm._gms
        assert isinstance(data, dict) and len(data) == len(gm), 'data shape wrong!'

        if len(sub_gms) == 1:
            print(1)
            return [data, ]
        else:
            x_individuals = list()
            for _ in sub_gms:
                x_individuals.append(
                    dict()
                )

            for index in gm:

                for x_j in x_individuals:
                    x_j[index] = list()

                data_at_index = data[index]

                start = 0
                for j, x_j in enumerate(x_individuals):

                    num_local_dofs = sub_gms[j].num_local_dofs(index)
                    x_j[index] = np.array(
                        data_at_index[start:(start+num_local_dofs)]
                    )
                    start += num_local_dofs

            return x_individuals

    @property
    def assemble(self):
        if self._assemble is None:
            self._assemble = IrregularStaticLocalVectorAssemble(self)
        return self._assemble

    @property
    def customize(self):
        """customize

        Will not affect dependent matrices. See ``adjust``.
        """
        return self._customize

    @property
    def adjust(self):
        """
        Adjustment will change matrices dependent on me. For example, B = A.T. If I adjust A late on,
        B will also change.

        While if we ``customize`` A, B will not be affected.
        """
        return self._adjust

    def _get_meta_data(self, i):
        """"""
        if self._dtype == 'callable':
            # noinspection PyCallingNonCallable
            return self._data_caller(i)
        elif self._dtype == 'None':
            raise Exception(f"None type vector has no data, set it first!")
        else:
            return self._data[i]  # raise Error when data is None.

    def _get_data_adjusted(self, i):
        """"""
        if len(self._adjust) == 0:
            return self._get_meta_data(i)
        else:
            if i in self.adjust:
                # adjusted data is cached in `adjust.adjustment` anyway!
                data = self.adjust[i]
            else:
                data = self._get_meta_data(i)
            return data

    def __getitem__(self, i):
        """When `self._data` is None, raise Error.

        return the vector (1d array) for element #i.
        """
        if len(self._adjust) == len(self._customize) == 0:
            return self._get_meta_data(i)

        elif len(self._customize) == 0:
            return self._get_data_adjusted(i)

        else:
            if i in self.customize:
                data = self.customize[i]
            else:
                data = self._get_data_adjusted(i)
            return data

    def __iter__(self):
        """Iteration over all cells."""
        if self._dtype == 'dict':
            for index in self._data:
                yield index
        else:
            for index in self._gm:
                yield index

    def __rmul__(self, other):
        """rmul:  other * self"""
        if isinstance(other, (int, float)):
            if self._dtype == 'None':
                raise Exception(f"cannot do * for None type vector")
            elif self._dtype == 'dict':

                _new_data_dict = dict()
                for i in self._gm:
                    _new_data_dict[i] = other * self._get_data_adjusted(i)
                return self.__class__(_new_data_dict, self._gm)

            elif self._dtype == 'callable':
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def __add__(self, other):
        """"""
        if other.__class__ is self.__class__:
            assert other._gm == self._gm, f"gathering matrix does not match."

            if self._dtype == 'None' or other._dtype == 'None':
                raise Exception(f"cannot do + for None type vector")
            elif self._dtype == 'dict' and other._dtype == 'dict':
                _new_data_dict = dict()
                for i in self._gm:
                    _new_data_dict[i] = self._get_data_adjusted(i) + other._get_data_adjusted(i)
                return self.__class__(_new_data_dict, self._gm)
            elif self._dtype == 'callable' or other._dtype == 'callable':
                raise NotImplementedError()

        else:
            raise NotImplementedError(f"{other}")

    def __neg__(self):
        """- self."""
        if self._dtype == 'None':
            raise Exception(f"cannot do * for None type vector")
        elif self._dtype == 'dict':
            _new_data_dict = dict()
            for i in self._gm:
                _new_data_dict[i] = - self._get_data_adjusted(i)
            return self.__class__(_new_data_dict, self._gm)

        elif self._dtype == 'callable':
            raise NotImplementedError()

        else:
            raise NotImplementedError()


class _IrregularLocalVectorAdjust(Frozen):
    """"""

    def __init__(self, v):
        """"""
        self._v = v
        self._adjustments = {}
        self._freeze()

    def __len__(self):
        return len(self._adjustments)

    def __contains__(self, i):
        """"""
        return i in self._adjustments

    def __getitem__(self, i):
        """"""
        return self._adjustments[i]


class _IrregularLocalVectorCustomize(Frozen):
    """"""
    def __init__(self, v):
        """"""
        self._v = v
        self._customizations = {}
        self._freeze()

    def __len__(self):
        return len(self._customizations)

    def __contains__(self, i):
        """"""
        return i in self._customizations

    def __getitem__(self, i):
        """Return the customized data for element #i."""
        return self._customizations[i]

    def set_value(self, i, value):
        """Set global value v[i] to be `value`.

        Parameters
        ----------
        i
        value

        Returns
        -------

        """
        fcs_local_rows = self._v._gm._find_fundamental_cells_and_local_indices_of_dofs(i)
        dof = list(fcs_local_rows.keys())[0]
        fcs, local_rows = fcs_local_rows[dof]
        fc, local_row = fcs[0], local_rows[0]
        data = self._v[fc]
        data[local_row] = value
        self._customizations[fc] = data
        for fc, local_row in zip(fcs[1:], local_rows[1:]):
            data = self._v[fc]
            data[local_row] = 0
            self._customizations[fc] = data

    def set_values(self, global_dofs, cochain):
        """set `v[global_dofs]` to be `cochain`."""
        # first we build up one-2-one relation between dofs and cochain
        if isinstance(cochain, (int, float)):
            cochain = np.ones(len(global_dofs)) * cochain
        else:
            pass

        assert len(global_dofs) == len(cochain), f"len(dofs) != len(cochains)"
        dof_cochain_dict = {}
        for dof, cc in zip(global_dofs, cochain):
            if dof in dof_cochain_dict:
                pass
            else:
                dof_cochain_dict[dof] = cc

        fcs_local_rows = self._v._gm._find_fundamental_cells_and_local_indices_of_dofs(
            list(dof_cochain_dict.keys())
        )

        involved_fcs = list()
        for dof in fcs_local_rows:
            for fc in fcs_local_rows[dof][0]:
                if fc not in involved_fcs:
                    involved_fcs.append(fc)

        involved_data = dict()
        for fc in involved_fcs:
            involved_data[fc] = self._v[fc]

        for dof in dof_cochain_dict:
            cochain = dof_cochain_dict[dof]
            fcs, rows = fcs_local_rows[dof]

            fc, row = fcs[0], rows[0]

            involved_data[fc][row] = cochain

            if len(fcs) == 1:  # this dof only appear in a single place.
                pass
            else:
                fcs, rows = fcs[1:], rows[1:]
                for fc, row in zip(fcs, rows):
                    involved_data[fc][row] = 0  # set other places to be 0.

        for fc in involved_data:  # update customization
            self._customizations[fc] = involved_data[fc]


class IrregularStaticCochainVector(IrregularStaticLocalVector):
    """"""

    def __init__(self, rf, t, g, irregular_data, gathering_matrix):
        """"""
        if irregular_data is None:
            pass
        else:
            assert isinstance(irregular_data, dict), \
                f"{IrregularStaticCochainVector} only accepts dict data."
        self._f = rf
        self._time = t
        self._g = g
        super().__init__(irregular_data, gathering_matrix)
        self._freeze()

    def override(self):
        """override `self._data` to be the cochain of `self._f` at time `self._t`."""
        if len(self.adjust) == 0 and len(self.customize) == 0:
            assert self.data is not None, f"I have no data."
            self._f[(self._time, self._g)].cochain = self.data
        else:
            raise NotImplementedError()
