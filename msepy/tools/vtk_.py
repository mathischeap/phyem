# -*- coding: utf-8 -*-
r"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""
import numpy as np
from pyevtk.hl import unstructuredGridToVTK
from tools.frozen import Frozen


class BuildVtkHexahedron(Frozen):  # using only No.12 VTK cells.
    """"""

    def __init__(self, x, y, z, cell_layout):
        """

        Parameters
        ----------
        x
        y
        z
        cell_layout :
            In each element, we have *cell_layout along 3-axes.
        """

        numbering = _parse_numbering(x, y, z)
        self._1d_numbering = numbering.ravel('C')
        self._num_dofs = np.max(numbering) + 1
        self._x = self._flat_data_according_to_numbering(x, numbering)
        self._y = self._flat_data_according_to_numbering(y, numbering)
        self._z = self._flat_data_according_to_numbering(z, numbering)
        self._numbering = numbering
        num_cell_each_element = np.prod(cell_layout)
        num_elements = len(numbering)
        num_cells = num_elements * num_cell_each_element
        connectivity = np.zeros((num_cells, 8), dtype=int)
        II, JJ, KK = cell_layout
        points_shape = (II+1, JJ+1, KK+1)
        for e in range(num_elements):    # this is not ideal (but fast enough), can be vectorized.
            numbering_e = numbering[e].reshape(points_shape, order='F')
            for k in range(KK):                    # this is not ideal (but fast enough), can be vectorized.
                for j in range(JJ):                # this is not ideal (but fast enough), can be vectorized.
                    for i in range(II):            # this is not ideal (but fast enough), can be vectorized.
                        cell_num = i + j*II + k*II*JJ + e*num_cell_each_element
                        connectivity[cell_num] = [
                            numbering_e[i, j, k],
                            numbering_e[i+1, j, k],
                            numbering_e[i+1, j+1, k],
                            numbering_e[i, j+1, k],
                            numbering_e[i, j, k+1],
                            numbering_e[i+1, j, k+1],
                            numbering_e[i+1, j+1, k+1],
                            numbering_e[i, j+1, k+1],
                        ]
        self._connectivity = connectivity.ravel('C')
        self._offsets = np.arange(8, 8*(num_cells+1), 8)
        self._cell_types = np.ones(num_cells) * 12
        self._freeze()

    def __call__(self, file_path, point_data=None, cell_data=None):
        """"""
        if point_data is not None:
            assert isinstance(point_data, dict), f"please put point data in a dict."
            for data_name in point_data:
                data = point_data[data_name]
                if isinstance(data, np.ndarray):  # scalar data
                    point_data[data_name] = self._flat_data_according_to_numbering(data, self._numbering)
                elif len(data) == 3:  # vector data
                    vec_data = list()
                    for di in data:
                        vec_data.append(
                            self._flat_data_according_to_numbering(
                                di, self._numbering
                            )
                        )
                    point_data[data_name] = tuple(vec_data)  # use tuple only.
                else:
                    raise NotImplementedError
        else:
            pass

        if cell_data is not None:
            raise NotImplementedError()
        else:
            pass

        unstructuredGridToVTK(
            file_path,
            self._x, self._y, self._z,
            connectivity=self._connectivity, offsets=self._offsets, cell_types=self._cell_types,
            cellData=cell_data,
            pointData=point_data
        )

    def _flat_data_according_to_numbering(self, data, numbering):
        """transfer data into 1-d data according a numbering (msepy gathering matrix)"""
        assert data.shape == numbering.shape, f"x.shape wrong."
        d1 = np.zeros(self._num_dofs)  # 1d data
        d1[self._1d_numbering] = data.ravel('C')
        # indices = self._1d_numbering
        # data = data.ravel('C')
        # indptr = np.array([0, len(data)])
        # spa_data = csr_matrix(  # this is nice, but this does not take average for shared values.
        #     (data, indices, indptr),
        #     shape=(1, self._num_dofs)
        # )
        return d1


def _parse_numbering(x, y, z):
    """"""
    shape = x.shape
    x = np.round(x, decimals=5).ravel('C')
    y = np.round(y, decimals=5).ravel('C')
    z = np.round(z, decimals=5).ravel('C')
    numbering = - np.ones(len(x), dtype=int)
    current_numbering = 0
    for i in range(len(x)):
        if numbering[i] != -1:
            pass
        else:
            xi, yi, zi = x[i], y[i], z[i]
            x_indices = np.where(
                np.logical_and(
                    np.logical_and(x == xi, y == yi),
                    z == zi,
                )
            )[0]
            numbering[x_indices] = current_numbering
            current_numbering += 1

    numbering = numbering.reshape(shape, order='C')

    return numbering
