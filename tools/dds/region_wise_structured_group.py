r"""For some dds-rws data that use the same coo_dict_list, we can save them all together to
save some space.

"""
import pickle
from tools.frozen import Frozen
from src.config import RANK, MASTER_RANK

from tools.dds.region_wise_structured import _find_shape
from tools.dds.region_wise_structured import DDSRegionWiseStructured


class DDS_RegionWiseStructured_Group(Frozen):
    """"""
    def __init__(self, coo_dict_list, val_dict_list_group):
        """"""
        assert RANK == MASTER_RANK, f"only use this class in the master rank please."
        space_dim = len(coo_dict_list)
        assert isinstance(coo_dict_list[0], dict), f"put coordinates in a dict please."
        regions = coo_dict_list[0].keys()

        for i, coo_dict in enumerate(coo_dict_list):
            assert coo_dict.keys() == regions, f"regions do not match."

        self._space_dim = space_dim    # we are in n-d space.
        self._coo_dict_list = coo_dict_list

        self._value_shapes = list()
        for i, val_dict_list in enumerate(val_dict_list_group):
            val_shape = _find_shape(val_dict_list)
            assert len(val_shape) == 1, f"put all values in a 1d list or tuple."
            val_shape = val_shape[0]
            self._value_shapes.append(val_shape)
            # self._value_shapes[i] represents the amount of components of ith data.
            # if self._value_shapes[1] == 3 and space_dim == 3, we know the second data is a vector (in 3d space).
            for j, component in enumerate(val_dict_list):
                assert isinstance(component, dict) and component.keys() == regions, \
                    f"{j}th component of {i}th variable has wrong region keys."

        for region in regions:
            region_data_shape = coo_dict_list[0][region].shape
            for coo in coo_dict_list[1:]:
                region_coo = coo[region]
                assert region_coo.shape == region_data_shape, \
                    f"coo shape does not match in region {region}."

            for i, val_dict_list in enumerate(val_dict_list_group):
                for j, component in enumerate(val_dict_list):
                    component_region_shape = component[region].shape
                    assert component_region_shape == region_data_shape, \
                        f"{j}th component of {i}th variable has wrong data shape in region {region}."

        self._val_dict_list_group = val_dict_list_group
        self._dtype = None
        self._individuals_ = {}
        for i, _ in enumerate(val_dict_list_group):
            self._individuals_[i] = None
        self._freeze()

    def __repr__(self):
        super_repr = super().__repr__().split('object')[1]
        return rf"<DDR-RWS-grouped {self.ndim}d [" + ','.join(self.dtype) + ']' + super_repr

    # --------- save & read ------------------------------------------------------------------------------

    def saveto(self, filename):
        """"""
        data_dict = {
            'key': 'dds-rws-grouped',
            'coo_dict_list': self._coo_dict_list,
            'val_dict_list_group': self._val_dict_list_group,
        }
        with open(filename, 'wb') as output:
            pickle.dump(data_dict, output, pickle.HIGHEST_PROTOCOL)
        output.close()

    @classmethod
    def read(cls, filename):
        """"""
        with open(filename, 'rb') as inputs:
            data = pickle.load(inputs)
        inputs.close()
        assert data['key'] == 'dds-rws-grouped', f'I am reading a wrong file.'
        coo_dict_list = data['coo_dict_list']
        val_dict_list_group = data['val_dict_list_group']
        return DDS_RegionWiseStructured_Group(coo_dict_list, val_dict_list_group)

    # -- properties --------------------------------------------------------------------------
    @property
    def dtype(self):
        """'scalar', 'vector', 'tensor' or so on."""
        if self._dtype is None:
            self._dtype = list()
            for val_sp in self._value_shapes:
                if val_sp == 1:
                    self._dtype.append('scalar')
                else:
                    if self._space_dim == val_sp:
                        self._dtype.append('vector')
                    elif val_sp == self._space_dim ** 2:
                        self._dtype.append('tensor')
                    else:
                        raise NotImplementedError()
        return self._dtype

    @property
    def ndim(self):
        """dimensions of the space."""
        return self._space_dim

    # ------------- wrap of dds-rws -----------------------------------------------------------
    def __getitem__(self, i):
        """get the ith variable as a dds-rws."""
        assert i in self._individuals_, f"I have {len(self._individuals_)} variables. i={i} is illegal."
        if self._individuals_[i] is None:
            # noinspection PyTypeChecker
            self._individuals_[i] = DDSRegionWiseStructured(
                self._coo_dict_list, self._val_dict_list_group[i]
            )
        else:
            pass
        return self._individuals_[i]
