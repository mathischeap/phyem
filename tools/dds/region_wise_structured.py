# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from tools.matplot.contour import contour, contourf


class DDSRegionWiseStructured(Frozen):
    """We organize data set in a region-wise (as keys of dictionaries) approach. For example, if

        `coo_dict_list = [x, y, z]`
        `val_dict_list = [v0, ]`

    This represents a scalar in 3d space. And x, y, z, v0 in this case must be dictionaries whose values
    are ndarray of ndim = 3 and all values must be of the same shape. The keys of all dictionaries must
    be same as well.

    Similarly, if

        `coo_dict_list = [x, y, z]`
        `val_dict_list = [v0, v1, v2]`

    This represents a vector in 3d space. And, if

        `coo_dict_list = [x, y, z]`
        `val_dict_list = ([v00, v01, v02], [v10, v11, v12], [v20, v21, v22])`

    This represents a tensor in three dimensions.

    """

    def __init__(self, coo_dict_list, val_dict_list):
        """"""
        space_dim = len(coo_dict_list)
        val_shape = _find_shape(val_dict_list)
        data_shape = None
        for i, coo_dict in enumerate(coo_dict_list):
            for region in coo_dict:
                xyz = coo_dict[region]
                assert isinstance(xyz, np.ndarray), f"coordinate must be put in ndarray."
                if data_shape is None:
                    data_shape = xyz.shape
                else:
                    assert data_shape == xyz.shape, f"#{i}th coordinate in region {region} does not match."

        self._space_dim = space_dim
        self._value_shape = val_shape
        self._data_shape = data_shape
        assert len(data_shape) == space_dim, f'put data into a structured way, so n-d data in n-d space.'
        self._coo_dict_list = coo_dict_list
        self._val_dict_list = val_dict_list
        self._freeze()

    def visualize(self, **kwargs):
        """"""
        if self._space_dim == 2 and self._value_shape == [1]:
            # plot a scalar field in 2d.
            return self._2d_scalar_field(**kwargs)
        else:
            raise NotImplementedError

    def _2d_scalar_field(self, plot_type='contourf', **kwargs):
        """"""
        x, y = self._coo_dict_list
        v = self._val_dict_list[0]

        if plot_type == 'contourf':
            fig = contourf(x, y, v, **kwargs)
        elif plot_type == 'contour':
            fig = contour(x, y, v, **kwargs)
        else:
            raise Exception()

        return fig


def _find_shape(list_of_dict):
    """"""
    d = list()
    while 1:
        d.append(len(list_of_dict))
        if isinstance(list_of_dict[0], dict):
            break
        else:
            list_of_dict = list_of_dict[0]
    return d
