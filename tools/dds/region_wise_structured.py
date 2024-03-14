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
        `val_dict_list = ([v00, v01, v02, v10, v11, v12, v20, v21, v22])`

    This represents a tensor in three dimensions.

    """

    def __init__(self, coo_dict_list, val_dict_list):
        """"""
        space_dim = len(coo_dict_list)
        val_shape = _find_shape(val_dict_list)
        assert len(val_shape) == 1, f"put all values in a 1d list or tuple."
        val_shape = val_shape[0]
        data_shape = None
        for i, coo_dict in enumerate(coo_dict_list):
            for region in coo_dict:
                xyz = coo_dict[region]
                assert isinstance(xyz, np.ndarray), f"coordinate must be put in ndarray."
                if data_shape is None:
                    data_shape = xyz.shape
                else:
                    assert data_shape == xyz.shape, f"#{i}th coordinate in region {region} does not match."

        self._space_dim = space_dim    # we are in n-d space.
        self._value_shape = val_shape  # the shape of the val list; implies it is a scalar, vector or something else.
        self._data_shape = data_shape  # the shape of all ndarray in coo or val.
        assert len(data_shape) == space_dim, f'put data into a structured way, so n-d data in n-d space.'
        self._coo_dict_list = coo_dict_list
        self._val_dict_list = val_dict_list
        self._dtype = None
        self._freeze()

    @property
    def dtype(self):
        """'scalar', 'vector', 'tensor' or so on."""
        if self._dtype is None:
            if self._value_shape == 1:
                self._dtype = 'scalar'
            else:
                if self._space_dim == self._value_shape:
                    self._dtype = 'vector'
                elif self._value_shape == self._space_dim ** 2:
                    self._dtype = 'tensor'
                else:
                    raise NotImplementedError()
        return self._dtype

    @property
    def ndim(self):
        """dimensions of the space."""
        return self._space_dim

    def visualize(self, magnitude=False, saveto=None, **kwargs):
        """"""
        if self._space_dim == 2 and self._value_shape == 1:
            # plot a scalar field in 2d.
            return self._2d_scalar_field(magnitude=magnitude, saveto=saveto, **kwargs)

        elif self._space_dim == 2 and self._value_shape == 2:
            # plot a vector field in 2d.
            return self._2d_vector_field(magnitude=magnitude, saveto=saveto, **kwargs)

        else:
            raise NotImplementedError

    def _2d_scalar_field(self, plot_type='contourf', magnitude=False, saveto=None, **kwargs):
        """"""
        x, y = self._coo_dict_list
        v = self._val_dict_list[0]

        if plot_type == 'contourf':
            fig = contourf(x, y, v, magnitude=magnitude, saveto=saveto, **kwargs)
        elif plot_type == 'contour':
            fig = contour(x, y, v, magnitude=magnitude, saveto=saveto, **kwargs)
        else:
            raise Exception()

        return fig

    def _2d_vector_field(self, plot_type='contourf', magnitude=False, saveto=None, **kwargs):
        """"""
        x, y = self._coo_dict_list
        v0, v1 = self._val_dict_list

        if saveto is None:
            pass
        else:
            raise NotImplementedError()

        if plot_type == 'contourf':
            fig0 = contourf(x, y, v0, magnitude=magnitude, **kwargs)
            fig1 = contourf(x, y, v1, magnitude=magnitude, **kwargs)
        elif plot_type == 'contour':
            fig0 = contour(x, y, v0, magnitude=magnitude, **kwargs)
            fig1 = contour(x, y, v1, magnitude=magnitude, **kwargs)
        else:
            raise Exception()

        return fig0, fig1

    def __sub__(self, other):
        """self - other"""
        assert other.__class__ == self.__class__, f"type wrong"
        assert self._space_dim == other._space_dim, f"space ndim wrong"
        assert self._value_shape == other._value_shape, f"value shape wrong"
        assert self._data_shape == other._data_shape, f"data shape wrong"

        for i in range(self._space_dim):
            xyz_self = self._coo_dict_list[i]
            xyz_other = other._coo_dict_list[i]
            for region in xyz_self:
                assert region in xyz_other, f"region does not match"
                assert np.allclose(xyz_self[region], xyz_other[region]), f"coordinates does not match."

        value_dict = [dict() for _ in range(self._value_shape)]
        for _ in range(self._value_shape):
            self_v, other_v = self._val_dict_list[_], other._val_dict_list[_]
            for region in self_v:
                assert region in other_v, f"region in value does not match."
                value_dict[_][region] = self_v[region] - other_v[region]

        return self.__class__(self._coo_dict_list, value_dict)

    def __add__(self, other):
        """self + other"""
        assert other.__class__ == self.__class__, f"type wrong"
        assert self._space_dim == other._space_dim, f"space ndim wrong"
        assert self._value_shape == other._value_shape, f"value shape wrong"
        assert self._data_shape == other._data_shape, f"data shape wrong"

        for i in range(self._space_dim):
            xyz_self = self._coo_dict_list[i]
            xyz_other = other._coo_dict_list[i]
            for region in xyz_self:
                assert region in xyz_other, f"region does not match"
                assert np.allclose(xyz_self[region], xyz_other[region]), f"coordinates does not match."

        value_dict = [dict() for _ in range(self._value_shape)]
        for _ in range(self._value_shape):
            self_v, other_v = self._val_dict_list[_], other._val_dict_list[_]
            for region in self_v:
                assert region in other_v, f"region in value does not match."
                value_dict[_][region] = self_v[region] + other_v[region]

        return self.__class__(self._coo_dict_list, value_dict)

    def x(self, other):
        """cross-product."""

    def cross_product(self, other):
        """cross-product."""


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
