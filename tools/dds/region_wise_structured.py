# -*- coding: utf-8 -*-
r"""
"""
import pickle
import numpy as np
from tools.frozen import Frozen
from tools.matplot.contour import contour, contourf
from src.config import RANK, MASTER_RANK


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
        assert RANK == MASTER_RANK, f"only use this class in the master rank please."
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

    def __repr__(self):
        super_repr = super().__repr__().split('object')[1]
        return rf"<DDR-RWS {self.ndim}d {self.dtype}" + super_repr

    def saveto(self, filename):
        """"""
        data_dict = {
            'key': 'dds-rws',
            'coo_dict_list': self._coo_dict_list,
            'val_dict_list': self._val_dict_list,
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
        assert data['key'] == 'dds-rws', f'I am reading a wrong file.'
        coo_dict_list = data['coo_dict_list']
        val_dict_list = data['val_dict_list']
        return DDSRegionWiseStructured(coo_dict_list, val_dict_list)

    # -- properties --------------------------------------------------------------------------
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

    @property
    def classification(self):
        """"""
        if (self.ndim, self.dtype) == (2, 'scalar'):
            return '2d scalar'
        elif (self.ndim, self.dtype) == (2, 'vector'):
            return '2d vector'
        elif (self.ndim, self.dtype) == (3, 'scalar'):
            return '3d scalar'
        elif (self.ndim, self.dtype) == (3, 'vector'):
            return '3d vector'
        else:
            raise NotImplementedError()

    # -- visualization --------------------------------------------------------------------------
    def visualize(self, magnitude=False, saveto=None, **kwargs):
        """"""
        if self.classification == '2d scalar':
            return self._2d_scalar_field(magnitude=magnitude, saveto=saveto, **kwargs)

        elif self.classification == '2d vector':
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

    # -- Operations --------------------------------------------------------------------------
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

    def __rmul__(self, other):
        """other * self"""
        if isinstance(other, (int, float)):
            value_dict = [dict() for _ in range(self._value_shape)]
            for _ in range(self._value_shape):
                self_v = self._val_dict_list[_]
                for region in self_v:
                    value_dict[_][region] = other * self_v[region]
            return self.__class__(self._coo_dict_list, value_dict)
        else:
            raise NotImplementedError()

    def x(self, other):
        """cross-product."""
        return self.cross_product(other)

    def cross_product(self, other):
        """cross-product."""
        if self.classification == '2d vector':

            if other.classification == '2d vector':
                # let A = [wx wy 0]^T    B = [u v 0]^T
                # A x B = [wy*0 - 0*v   0*u - wx*0   wx*v - wy*u]^T = [0   0   C0]^T

                wx, wy = self._val_dict_list
                u, v = other._val_dict_list

                c0 = {}

                for region in wx:

                    _wx = wx[region]
                    _wy = wy[region]
                    _u = u[region]
                    _v = v[region]

                    c0[region] = _wx*_v - _wy*_u

                return self.__class__(self._coo_dict_list, [c0])

            else:
                raise NotImplementedError
        else:
            raise NotImplementedError()

    def streamfunction(self, shift=0):
        """"""
        sf_dict = dict()
        if self.classification == '2d vector':
            X, Y = self._coo_dict_list
            U, V = self._val_dict_list
            starting_corner = None
            starting_sf = None
            corner_pool = dict()
            element_range = list(X.keys())
            element_range.sort()
            for e in element_range:
                x, y = X[e], Y[e]
                u, v = U[e], V[e]
                if starting_corner is None and starting_sf is None:
                    starting_corner, starting_sf = (0, 0), 0
                else:
                    starting_corner, starting_sf = self._renew_starting_point_(
                        corner_pool, x, y
                    )
                element_sf = self._compute_sf_in_region_(
                    x, y, u, v,
                    starting_corner, starting_sf
                )
                corner_pool.update(
                    self._find_corner_xy_sf_(x, y, element_sf)
                )
                sf_dict[e] = element_sf
        else:
            raise Exception(self.classification)

        if shift != 0:
            for e in sf_dict:
                sf_dict[e] += shift
        else:
            pass

        return self.__class__(self._coo_dict_list, [sf_dict, ])

    @staticmethod
    def _renew_starting_point_(corner_pool, x, y):
        """"""
        xm_ym = '%.7f-%.7f' % (x[0, 0], y[0, 0])
        if xm_ym in corner_pool:
            return (0, 0), corner_pool[xm_ym]

        xp_ym = '%.7f-%.7f' % (x[-1, 0], y[-1, 0])
        if xp_ym in corner_pool:
            return (-1, 0), corner_pool[xp_ym]

        xm_yp = '%.7f-%.7f' % (x[0, -1], y[0, -1])
        if xm_yp in corner_pool:
            return (0, -1), corner_pool[xm_yp]

        xp_yp = '%.7f-%.7f' % (x[-1, -1], y[-1, -1])
        if xp_yp in corner_pool:
            return (-1, -1), corner_pool[xp_yp]

    @staticmethod
    def _find_corner_xy_sf_(x, y, sf):
        """"""
        corner_sf_pool = {
            '%.7f-%.7f' % (x[0, 0], y[0, 0]): sf[0, 0],
            '%.7f-%.7f' % (x[-1, 0], y[-1, 0]): sf[-1, 0],
            '%.7f-%.7f' % (x[0, -1], y[0, -1]): sf[0, -1],
            '%.7f-%.7f' % (x[-1, -1], y[-1, -1]): sf[-1, -1]
        }
        return corner_sf_pool

    def _compute_sf_in_region_(
            self,
            x, y, u, v,
            starting_corner, starting_sf
    ):
        """"""
        sp0, sp1 = x.shape
        sf = np.ones_like(x)
        if starting_corner == (0, 0):
            sf[0, 0] = starting_sf
            for i in range(1, sp0):
                sf_start = sf[i-1, 0]

                sx, sy = x[i-1, 0], y[i-1, 0]
                ex, ey = x[i, 0], y[i, 0]

                dx = ex - sx
                dy = ey - sy

                su, sv = u[i-1, 0], v[i-1, 0]
                eu, ev = u[i, 0], v[i, 0]

                mu = (su + eu) / 2
                mv = (sv + ev) / 2

                d_sf_0 = mu * dy
                d_sf_1 = - mv * dx

                sf_end = sf_start + d_sf_0 + d_sf_1

                sf[i, 0] = sf_end

            # col [1:]
            for j in range(1, sp1):
                # node [j, 0]
                sf_start = sf[0, j-1]

                sx, sy = x[0, j-1], y[0, j-1]
                ex, ey = x[0, j], y[0, j]

                dx = ex - sx
                dy = ey - sy

                su, sv = u[0, j-1], v[0, j-1]
                eu, ev = u[0, j], v[0, j]

                mu = (su + eu) / 2
                mv = (sv + ev) / 2

                d_sf_0 = mu * dy
                d_sf_1 = - mv * dx

                sf_end = sf_start + d_sf_0 + d_sf_1

                sf[0, j] = sf_end

                # row [1:]
                for i in range(1, sp0):
                    sf_start = sf[i-1, j]

                    sx, sy = x[i-1, j], y[i-1, j]
                    ex, ey = x[i, j], y[i, j]

                    dx = ex - sx
                    dy = ey - sy

                    su, sv = u[i-1, j], v[i-1, j]
                    eu, ev = u[i, j], v[i, j]

                    mu = (su + eu) / 2
                    mv = (sv + ev) / 2

                    d_sf_0 = mu * dy
                    d_sf_1 = - mv * dx

                    sf_end = sf_start + d_sf_0 + d_sf_1

                    sf[i, j] = sf_end

            return sf

        else:
            raise NotImplementedError(starting_corner)


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
