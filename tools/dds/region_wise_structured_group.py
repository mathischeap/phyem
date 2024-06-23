r"""For some dds-rws data that use the same coo_dict_list, we can save them all together to
save some space.

"""
import pickle
import numpy as np
from tools.frozen import Frozen
from tools.matplot.contour import contour, contourf
from src.config import RANK, MASTER_RANK

from tools.dds.region_wise_structured import _find_shape


class DDS_RegionWiseStructured_Group(Frozen):
    """"""
    def __init__(self, coo_dict_list, val_dict_list_group):
        """"""
        assert RANK == MASTER_RANK, f"only use this class in the master rank please."
        space_dim = len(coo_dict_list)
        data_shape = None
        for i, coo_dict in enumerate(coo_dict_list):
            for region in coo_dict:
                xyz = coo_dict[region]
                assert isinstance(xyz, np.ndarray), f"coordinate must be put in ndarray."
                if data_shape is None:
                    data_shape = xyz.shape
                else:
                    assert data_shape == xyz.shape, f"#{i}th coordinate in region {region} does not match."

        assert len(data_shape) == space_dim, f'put data into a structured way, so n-d data in n-d space.'
        self._space_dim = space_dim    # we are in n-d space.
        self._data_shape = data_shape  # the shape of all ndarray in coo or val.
        self._coo_dict_list = coo_dict_list

        self._value_shapes = list()
        for val_dict_list in val_dict_list_group:
            val_shape = _find_shape(val_dict_list)
            assert len(val_shape) == 1, f"put all values in a 1d list or tuple."
            val_shape = val_shape[0]
            self._value_shapes.append(val_shape)
        # self._value_shapes[i] represents the amount of components of ith data.
        # if self._value_shapes[1] == 3 and space_dim == 3, we know the second data is a vector (in 3d space).

        self._val_dict_list_group = val_dict_list_group

        self._dtype = None
        self._freeze()
