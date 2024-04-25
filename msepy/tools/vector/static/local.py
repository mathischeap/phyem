# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen

from msepy.tools.vector.static.assembled import MsePyStaticAssembledVector
from numpy import zeros


class MsePyStaticLocalVectorAssemble(Frozen):
    """"""

    def __init__(self, v):
        """"""
        self._v = v
        self._freeze()

    def __call__(self):
        """"""
        gm = self._v._gm
        v = zeros(gm.num_dofs)

        for i in self._v:
            Vi = self._v[i]  # all adjustments and customizations take effect.
            v[gm[i]] += Vi  # must do this to be consistent with the matrix assembling.

        return MsePyStaticAssembledVector(v, gm)


class MsePyStaticLocalVector(Frozen):
    """"""
    def __init__(self, _2d_data, gathering_matrix):
        """"""
        self._gm = gathering_matrix
        self.data = _2d_data
        self._customize = MsePyStaticLocalVectorCustomize(self)
        self._assemble = None
        self._freeze()

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        return rf"MsePy Static Local Vector: {self._dtype}" + super_repr

    @property
    def data(self):
        """Let the data ready as a 2-d array.

        All customizations take effects.

        Must return a 2-d array data.
        """
        if len(self._customize) == 0:
            if self._dtype == 'callable':
                # collect all callable data here
                raise NotImplementedError(f"callable data is not ready")

            elif self._dtype == 'None':
                raise Exception(f"None type vector has no data, set it first!")

            else:
                return self._data  # will raise Error when data is callable (generated in realtime) or None.

        else:
            raise NotImplementedError()

    @data.setter
    def data(self, _data):
        """Do this such that data can be renewed.
        """
        # _2d_data: 2d numpy array or None.
        if _data is None:
            self._dtype = 'None'
            self._data = None

        elif isinstance(_data, (int, float)):
            self._dtype = "homogeneous"
            self._data = 1. * _data * np.ones(self._gm.shape)

        elif isinstance(_data, np.ndarray):
            self._dtype = "2d"  # for example, 2d array: rows -> num of elements, cols -> local cochain
            assert _data.shape == self._gm.shape, f"{_data.shape} != {self._gm.shape}"
            self._data = _data

        elif callable(_data):
            self._dtype = "callable"
            self._data_caller = _data
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

    def __getitem__(self, i):
        """When `self._data` is None, raise Error.

        return the vector (1d array) for element #i.
        """
        if len(self._customize) == 0:
            return self._get_meta_data(i)

        else:
            if i in self.customize:
                data = self.customize[i]
            else:
                data = self._get_meta_data(i)
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
                    num_local_dofs[-1] + sgm.num_local_dofs
                )

                x_individuals.append(
                    np.zeros(sgm.shape)
                )

            assert data.shape[1] == num_local_dofs[-1], f"num_local_dofs is built wrongly!"

            end = None
            for i, x_i in enumerate(x_individuals):
                start = num_local_dofs[i]
                end = num_local_dofs[i+1]

                x_individuals[i] = data[:, start:end]

            if end is not None:
                assert end == num_local_dofs[-1], f"make sure all data are used."

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
                def ___add___(i):
                    return self[i] + other[i]

                return self.__class__(___add___, self._gm)

        else:
            raise NotImplementedError(f"{other}")

    def __neg__(self):
        """# all adjustment and customization take effect."""
        if self._dtype == 'None':
            raise Exception(f"cannot do * for None type vector")

        elif self._dtype in ("homogeneous", "2d"):

            if len(self._customize) == 0:
                data = - self.data

                return self.__class__(data, self._gm)

            else:

                def ___neg_data___(i):
                    """"""
                    return - self[i]

                return self.__class__(___neg_data___, self._gm)

        elif self._dtype == 'callable':


            def ___neg_callable___(i):
                return - self[i]

            return self.__class__(___neg_callable___, self._gm)

        else:
            raise NotImplementedError()


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
        """Set global value v[i] to be `value`.

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
        element, local_row = elements[0], local_rows[0]
        data = self._v[element]
        data[local_row] = value
        self._customizations[element] = data
        for element, local_row in zip(elements[1:], local_rows[1:]):
            data = self._v[element]
            data[local_row] = 0
            self._customizations[element] = data

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


def concatenate(v_1d_list, gm):
    """"""
    shape = len(v_1d_list)
    gms = gm._gms
    assert len(gms) == shape, f"composite wrong."
    v = list()
    for i, vi in enumerate(v_1d_list):
        if vi is None:
            v.append(
                MsePyStaticLocalVector(0, gms[i])
            )
        else:
            assert issubclass(vi.__class__, MsePyStaticLocalVector) and vi._gm is gms[i], f"gm wrong!"
            v.append(
                vi
            )

    cv = _MsePyStaticLocalVectorConcatenate(v)

    return MsePyStaticLocalVector(cv, gm)


class _MsePyStaticLocalVectorConcatenate(Frozen):
    """"""

    def __init__(self, vs):
        """"""
        self._vs = vs
        self._freeze()

    def __call__(self, i):
        """get the concatenated vector for element #i."""
        v_list = list()
        for v in self._vs:
            v_list.append(
                v[i]  # all customizations take effect.
            )
        return np.concatenate(v_list)
