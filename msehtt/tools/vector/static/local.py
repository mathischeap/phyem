# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.src.config import COMM, RANK, MASTER_RANK
from phyem.tools.frozen import Frozen
from phyem.msehtt.tools.gathering_matrix import MseHttGatheringMatrix

from phyem.msehtt.tools.vector.static.global_distributed import MseHttGlobalVectorDistributed
from phyem.msehtt.tools.vector.static.global_gathered import MseHttGlobalVectorGathered


class MseHttStaticLocalVectorAssemble(Frozen):
    """"""

    def __init__(self, v):
        """"""
        self._v = v
        self._freeze()

    def __getitem__(self, e):
        """"""
        return self._v[e]

    def __iter__(self):
        """"""
        for i in self._v:
            yield i

    def __call__(self, vtype='distributed', mode='sum', customizations=None):
        """"""

        if vtype == 'distributed':
            if mode == 'sum':
                gm = self._v._gm
                v = np.zeros(gm.num_global_dofs)
                for i in self:
                    v[gm[i]] += self[i]  # must do this to be consistent with the matrix assembling.
                RETURN = MseHttGlobalVectorDistributed(v, gm)
            else:
                raise NotImplementedError(f"mode={mode} not implemented for {vtype} assembling")

        elif vtype == 'gathered':
            if mode == 'replace':
                v = {}
                for i in self:
                    v[i] = self[i]
                v = COMM.gather(v, root=MASTER_RANK)
                gm = COMM.gather(self._v._gm._gm, root=MASTER_RANK)
                if RANK == MASTER_RANK:
                    V = {}
                    GM = {}
                    for _, __ in zip(v, gm):
                        V.update(_)
                        GM.update(__)
                    v = np.zeros(self._v._gm.num_global_dofs)
                    for i in V:
                        v[GM[i]] = V[i]
                else:
                    v = np.zeros(self._v._gm.num_global_dofs)
                COMM.Bcast(v, root=MASTER_RANK)
                RETURN = MseHttGlobalVectorGathered(v, self._v._gm)

            else:
                raise NotImplementedError(f"mode={mode} not implemented for {vtype} assembling")

        else:
            raise NotImplementedError()

        if customizations is None:
            return RETURN
        else:
            return self._deal_with_customizations_(RETURN, customizations, vtype, mode)

    def _deal_with_customizations_(self, RETURN, customizations, vtype, mode):
        r""""""
        if len(customizations) == 1:
            cus = customizations[0]
            indicator = cus[0]
            if indicator == 'add_a_value_at_the_end':
                value = cus[1]
                if vtype == 'distributed' and mode == 'sum' and value == 0:
                    new_v = np.append(RETURN.V, [value, ])
                    return MseHttGlobalVectorDistributed(new_v)
                else:
                    raise NotImplementedError(f"vtype={vtype}, mode={mode}, value={value} not implemented.")
            else:
                raise NotImplementedError(
                    f"MseHttStaticLocalVectorAssemble _deal_with_customizations_ cannot do for "
                    f"indicator={indicator}."
                )
        else:
            raise NotImplementedError(
                f"MseHttStaticLocalVectorAssemble cannot deal with multi customizations yet"
            )


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
                elif data[i].ndim == 2 and data[i].shape[1] == 1:
                    data[i] = data[i][:, 0]
                else:
                    raise Exception()
                assert len(data[i]) == self._gm.num_local_dofs(i), f"num values in element #{i} is wrong."
            for i in self._gm:
                if self._gm.num_local_dofs(i) > 0:
                    if i in data:
                        assert data[i].shape == (self._gm.num_local_dofs(i),)
                    else:
                        pass
                else:
                    pass
            self._data = data

        elif callable(data):
            self._dtype = "realtime"
            self._data = data
        else:
            raise Exception(f"msepy static local vector data type wrong: {data.__class__}.")
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
        else:
            raise Exception()

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
        """iteration over all rank (local) element indices."""
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

    def __rmul__(self, other):
        """other * self"""
        if isinstance(other, (int, float)):
            def data_caller(i):
                return other * self[i]
            return self.__class__(data_caller, self._gm)
        else:
            raise NotImplementedError()

    def __add__(self, other):
        """self + other"""
        if (other.__class__ is self.__class__) or issubclass(other.__class__, self.__class__):
            assert self._gm == other._gm, f"gathering matrix does not match."

            data_dict = {}

            for e in self._gm:
                data_dict[e] = self[e] + other[e]

            return self.__class__(data_dict, self._gm)
        else:
            raise NotImplementedError(other.__class__)

    def __sub__(self, other):
        """self - other"""
        if (other.__class__ is self.__class__) or issubclass(other.__class__, self.__class__):
            assert self._gm == other._gm, f"gathering matrix does not match."

            data_dict = {}

            for e in self._gm:
                data_dict[e] = self[e] - other[e]

            return self.__class__(data_dict, self._gm)
        else:
            raise NotImplementedError(other.__class__)

    def __neg__(self):
        """-self"""
        def data_caller(i):
            return - self[i]

        return self.__class__(data_caller, self._gm)


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
        gm = self._sv._gm
        if isinstance(global_dof, (int, float)):
            pass
        elif global_dof.__class__.__name__ in ('int32', 'int64'):
            pass
        else:
            raise Exception(f"cannot deal with dof = {global_dof} of class {global_dof.__class__}!")

        if global_dof < 0:
            global_dof += gm.num_global_dofs
        else:
            pass
        assert global_dof == int(global_dof) and 0 <= global_dof < gm.num_global_dofs, \
            f"global_dof = {global_dof} is wrong."
        global_dof = int(global_dof)
        elements_local_rows = gm.find_rank_locations_of_global_dofs(global_dof)[global_dof]
        num_global_locations = gm.num_global_locations(global_dof)
        if num_global_locations == 1:
            if len(elements_local_rows) == 1:  # in the rank where the place is
                rank_element, local_dof = elements_local_rows[0]
                data = self._sv[rank_element].copy()
                data[local_dof] = value
                self._customizations[rank_element] = data
            else:
                pass
        else:
            representative_rank, element = gm.find_representative_location(global_dof)
            if RANK == representative_rank:
                representative_local_dof = 1
                for element_local_dof in elements_local_rows:
                    _element, _local_dof = element_local_dof
                    if (_element == element) and representative_local_dof:
                        representative_local_dof = 0
                        data = self._sv[_element].copy()
                        data[_local_dof] = value
                        self._customizations[_element] = data
                    else:
                        data = self._sv[_element].copy()
                        data[_local_dof] = 0
                        self._customizations[_element] = data
            else:
                for element_local_dof in elements_local_rows:
                    _element, _local_dof = element_local_dof
                    data = self._sv[_element].copy()
                    data[_local_dof] = 0
                    self._customizations[_element] = data

    def set_values(self, global_dofs, global_cochain):
        """"""
        if isinstance(global_cochain, (int, float)):
            for global_dof in global_dofs:
                self.set_value(global_dof, global_cochain)

        else:
            assert len(global_dofs) == len(global_cochain), f"dofs, cochain length dis-match."
            for global_dof, cochain in zip(global_dofs, global_cochain):
                self.set_value(global_dof, cochain)

    def set_value_through_local_dof(self, element_index, local_dof_index, value):
        """We first try to find the global dof of this local dof, then apply the `set_value` method.

        So, all positions that share this dof will be taken into consideration.

        Returns
        -------

        """
        raise NotImplementedError()

    def set_local_value(self, element_index, local_dof_index, value):
        """Unlike the `set_value_through_local_dof` method, this method only change the value in the
        element #`element_index`. This is useful when we try to solve a problem element-wise.

        Parameters
        ----------
        element_index
        local_dof_index
        value

        Returns
        -------

        """
        if element_index in self._sv:
            data = self._sv[element_index].copy()
            data[local_dof_index] = value
            self._customizations[element_index] = data
        else:
            pass


def concatenate(v_1d_list, gm):
    """"""
    if gm.__class__ is MseHttGatheringMatrix:
        pass
    elif isinstance(gm, (list, tuple)) and all([_.__class__ is MseHttGatheringMatrix for _ in gm]):
        gm = MseHttGatheringMatrix(gm)
    else:
        raise NotImplementedError()

    gms = gm._gms
    shape = len(v_1d_list)
    assert len(gms) == shape, f"composite wrong."
    vs = list()
    for i, vi in enumerate(v_1d_list):
        if vi is None:
            vs.append(
                MseHttStaticLocalVector(0, gms[i])
            )
        else:
            assert vi._gm is gms[i], f"vi wrong: {vi.__class__}"
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
