# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from src.config import COMM, MPI
from tools.frozen import Frozen
from msehtt.tools.gathering_matrix import MseHttGatheringMatrix

from msehtt.tools.vector.static.global_distributed import MseHttGlobalVectorDistributed


class MseHttStaticLocalVectorAssemble(Frozen):
    """"""

    def __init__(self, v):
        """"""
        self._v = v
        self._freeze()

    def __call__(self):
        """"""
        gm = self._v._gm
        v = np.zeros(gm.num_global_dofs)

        for i in self._v:
            Vi = self._v[i]  # all adjustments and customizations take effect.
            v[gm[i]] += Vi  # must do this to be consistent with the matrix assembling.

        return MseHttGlobalVectorDistributed(v, gm)


class EmptyDataError(Exception):
    """"""


class MseHttStaticLocalVector(Frozen):
    """"""
    def __init__(self, data, gathering_matrix):
        """We do not use data cache for local vector."""
        assert gathering_matrix.__class__ is MseHttGatheringMatrix, f"I need a gathering matrix."
        self._gm = gathering_matrix
        self._receive_data(data)
        self._assemble = None
        self._freeze()

    def _receive_data(self, data):
        """Do this such that data can be renewed.
        """
        # _2d_data: 2d numpy array or None.
        if data is None:
            self._dtype = 'None'
            self._data = None

        elif isinstance(data, (int, float)):
            if data == 0:
                self._dtype = "dict"
                self._data = {}
            else:
                self._dtype = "homogeneous"
                self._data = data

        elif isinstance(data, dict):
            self._dtype = "dict"
            for i in data:
                assert i in self._gm, f"element #{i} is not a local element"
                assert isinstance(data[i], np.ndarray), f"each local vector must be 1d array."
                if data[i].ndim == 1:
                    pass
                elif data[i].shape[1] == 1:
                    data[i] = data[i][:, 0]
                else:
                    raise Exception()
                assert len(data[i]) == self._gm.num_local_dofs(i), f"num values in element #{i} is wrong."
            for i in self._gm:
                if self._gm.num_local_dofs(i) > 0:
                    assert i in data, f"data missing for element #{i}."
                else:
                    pass
            self._data = data

        elif callable(data):
            self._dtype = "realtime"
            self._data = data
        else:
            raise Exception(f"msepy static local vector data type wrong.")
        self._customize = None

    @property
    def gathering_matrix(self):
        return self._gm

    @property
    def gm(self):
        return self._gm

    def _get_meta_data(self, i):
        """Get meta data for element #i"""
        assert i in self._gm, f"element #{i} is not a local element."
        if self._dtype == "None":
            raise EmptyDataError()
        elif self._dtype == 'homogeneous':
            return self._data * np.ones(self._gm.num_local_dofs(i))
        elif self._dtype == 'dict':
            if i in self._data:
                return self._data[i]
            else:
                return np.zeros(self._gm.num_local_dofs(i))
        elif self._dtype == "realtime":
            return self._data(i)

    @property
    def customize(self):
        """customize"""
        if self._customize is None:
            # noinspection PyAttributeOutsideInit
            self._customize = _MseHtt_StaticVector_Customize(self)
        return self._customize

    @property
    def assemble(self):
        """"""
        if self._assemble is None:
            self._assemble = MseHttStaticLocalVectorAssemble(self)
        return self._assemble

    def __getitem__(self, i):
        """Get the data of element #i."""
        assert i in self._gm, f"element #{i} is not local."
        if i in self.customize:
            return self.customize[i]
        else:
            return self._get_meta_data(i)

    def __iter__(self):
        """iteration over all rank elements."""
        for i in self._gm:
            yield i

    def __contains__(self, i):
        """If i is a rank element?"""
        return i in self._gm

    def __len__(self):
        """"""
        return len(self._gm)

    @property
    def data_dict(self):
        """generate all data and put them in a dictionary in real time and all customization will take effect."""
        data_dict = {}
        for i in self:
            data_dict[i] = self[i]
        return data_dict

    @staticmethod
    def is_static():
        """static"""
        return True

    def split(self, data_dict=None):
        """split the `data_dict` according to `self._gm._gms` (chained gathering matrix).

        if `data_dict` is `None`, it split the ``data_dict`` of itself.
        """
        if data_dict is None:
            data_dict = self.data_dict  # 2d data ready!
        else:
            pass
        return self._gm.split(data_dict)


class _MseHtt_StaticVector_Customize(Frozen):
    """"""
    def __init__(self, sv):
        """"""
        self._sv = sv
        self._customizations = {}
        self._freeze()

    def __contains__(self, i):
        """"""
        return i in self._customizations

    def __getitem__(self, i):
        """"""
        return self._customizations[i]

    def __len__(self):
        """"""
        return len(self._customizations)

    def set_value(self, global_dof, value):
        """"""
        assert isinstance(global_dof, (int, float)), f"can just deal with one dof!"
        if global_dof < 0:
            global_dof += self._sv._gm.num_global_dofs
        else:
            pass
        assert global_dof == int(global_dof) and 0 <= global_dof < self._sv._gm.num_global_dofs, \
            f"global_dof = {global_dof} is wrong."
        global_dof = int(global_dof)
        elements_local_rows = self._sv._gm.find_rank_locations_of_global_dofs(global_dof)[global_dof]
        num_rank_locations = len(elements_local_rows)
        num_global_locations = COMM.allreduce(num_rank_locations, op=MPI.SUM)
        if num_global_locations == 1:
            if num_rank_locations == 1:  # in the rank where the place is
                rank_element, local_dof = elements_local_rows[0]
                data = self._sv[rank_element].copy()
                data[local_dof] = value
                self._customizations[rank_element] = data
            else:
                pass
        else:
            raise NotImplementedError()


def concatenate(v_1d_list, gm):
    """"""
    shape = len(v_1d_list)
    gms = gm._gms
    assert len(gms) == shape, f"composite wrong."
    vs = list()
    for i, vi in enumerate(v_1d_list):
        if vi is None:
            vs.append(
                MseHttStaticLocalVector(0, gms[i])
            )
        else:
            assert issubclass(vi.__class__, MseHttStaticLocalVector) and vi._gm is gms[i], f"gm wrong!"
            vs.append(
                vi
            )

    cv = _MsePyStaticLocalVectorConcatenate(vs)

    return MseHttStaticLocalVector(cv, gm)


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
