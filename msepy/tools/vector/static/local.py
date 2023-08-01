# -*- coding: utf-8 -*-
"""
"""
import numpy as np
from tools.frozen import Frozen
from msepy.tools.vector.static.assemble import MsePyStaticLocalVectorAssemble


class MsePyStaticLocalVector(Frozen):
    """"""
    def __init__(self, _2d_data, gathering_matrix):
        """"""
        self._gm = gathering_matrix
        self.data = _2d_data
        self._adjust = MsePyStaticLocalVectorAdjust(self)
        self._customize = MsePyStaticLocalVectorCustomize(self)
        self._assemble = None
        self._freeze()

    @property
    def data(self):
        """Let the data ready as a 2-d array.

        All adjustments and customizations take effects.

        Must return a 2-d array data.
        """
        if len(self._adjust) == len(self._customize) == 0:
            if self._dtype == 'callable':
                raise NotImplementedError(f"callable data is not ready")

            elif self._dtype == 'None':
                raise Exception(f"None type vector has no data, set it first!")

            else:
                return self._data  # will raise Error when data is callable (generated in realtime) or None.
        else:
            raise NotImplementedError()

    @data.setter
    def data(self, data):
        """Do this such that data can be renewed.
        """
        # _2d_data: 2d numpy array or None.
        if data is None:
            self._dtype = 'None'
            self._data = data

        elif isinstance(data, (int, float)):
            self._dtype = "homogeneous"
            self._data = 1. * data * np.ones(self._gm.shape)

        elif isinstance(data, np.ndarray):
            self._dtype = "2d"  # for example, 2d array: rows -> num of elements, cols -> local cochain
            assert data.shape == self._gm.shape
            self._data = data

        elif callable(data):
            self._dtype = "callable"
            self._data_caller = data
            self._data = None

        else:
            raise Exception(f"msepy static local vector data type wrong.")

    def _get_meta_data(self, i):
        """"""
        if self._dtype == 'callable':
            return self._data_caller(i)
        elif self._dtype == 'None':
            raise Exception(f"None type vector has no data, set it first!")
        else:
            return self._data[i]  # raise Error when data is None.

    def _get_data_adjusted(self, i):
        """"""
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
        """"""
        for i in range(self._gm.num_elements):
            yield i

    @staticmethod
    def is_static():
        """static"""
        return True

    def split(self, data=None):
        """split the `data` according to `self._gm._gms` (chained gathering matrix).

        if `data` is `None`, it split the data of itself.
        """
        if data is None:
            data = self.data  # 2d data ready!
        else:
            pass
        gm = self._gm
        sub_gms = gm._gms

        assert data.shape == gm.shape, 'data shape wrong!'

        if len(sub_gms) == 1:
            return [data, ]
        else:
            num_local_dofs = [0, ]
            x_individuals = list()
            for sgm in sub_gms:
                num_local_dofs.append(
                    sum(num_local_dofs) + sgm.num_local_dofs
                )

                x_individuals.append(
                    np.zeros(sgm.shape)
                )

            for i, x_i in enumerate(x_individuals):
                start = num_local_dofs[i]
                end = num_local_dofs[i+1]

                x_individuals[i] = data[:, start:end]

            return x_individuals
    
    @property
    def assemble(self):
        if self._assemble is None:
            self._assemble = MsePyStaticLocalVectorAssemble(self)
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

    def __rmul__(self, other):
        """rmul"""
        if isinstance(other, (int, float)):
            if self._dtype == 'None':
                raise Exception(f"cannot do * for None type vector")
            elif self._dtype in ("homogeneous", "2d"):
                data = other * self.data
                return self.__class__(data, self._gm)
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
            elif self._dtype in ("homogeneous", "2d") and other._dtype in ("homogeneous", "2d"):
                data = self.data + other.data
                return self.__class__(data, self._gm)
            elif self._dtype == 'callable' or other._dtype == 'callable':
                raise NotImplementedError()

        else:
            raise NotImplementedError()

    def __neg__(self):
        """- self."""
        if self._dtype == 'None':
            raise Exception(f"cannot do * for None type vector")
        elif self._dtype in ("homogeneous", "2d"):
            data = - self.data
            return self.__class__(data, self._gm)
        elif self._dtype == 'callable':
            raise NotImplementedError()

        else:
            raise NotImplementedError()


class MsePyStaticLocalVectorAdjust(Frozen):
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


class MsePyStaticLocalVectorCustomize(Frozen):
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
        """Set v[i] to be `value`.

        Parameters
        ----------
        i
        value

        Returns
        -------

        """
        elements_local_rows = self._v._gm._find_elements_and_local_indices_of_dofs(i)
        dof = list(elements_local_rows.keys())[0]
        elements, local_rows = elements_local_rows[dof]
        if len(elements) == 1:  # only found one place.
            element, local_row = elements[0], local_rows[0]
            data = self._v[element]
            data[local_row] = value
            self._customizations[element] = data

        else:
            raise NotImplementedError()

    def set_values(self, global_dofs, cochain):
        """set `v[global_dofs]` to be `cochain`."""
        # first we build up one-2-one relation between dofs and cochain
        assert len(global_dofs) == len(cochain), f"len(dofs) != len(cochains)"
        dof_cochain_dict = {}
        for dof, cc in zip(global_dofs, cochain):
            if dof in dof_cochain_dict:
                pass
            else:
                dof_cochain_dict[dof] = cc

        elements_local_rows = self._v._gm._find_elements_and_local_indices_of_dofs(list(dof_cochain_dict.keys()))

        involved_elements = list()
        for dof in elements_local_rows:
            for element in elements_local_rows[dof][0]:
                if element not in involved_elements:
                    involved_elements.append(element)

        involved_data = dict()
        for element in involved_elements:
            involved_data[element] = self._v[element]

        for dof in dof_cochain_dict:
            cochain = dof_cochain_dict[dof]
            elements, rows = elements_local_rows[dof]

            element, row = elements[0], rows[0]

            involved_data[element][row] = cochain

            if len(elements) == 1:  # this dof only appear in a single place.
                pass
            else:
                elements, rows = elements[1:], rows[1:]
                for element, row in zip(elements, rows):
                    involved_data[element][row] = 0  # set other places to be 0.

        for element in involved_data:  # update customization
            self._customizations[element] = involved_data[element]
