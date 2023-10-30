# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from pyevtk.hl import unstructuredGridToVTK
from tools.frozen import Frozen


class BuildVtkUnStruct(Frozen):
    """"""

    def __init__(self, numbering, xy, cell_layout):
        """"""
        px = py = cell_layout
        assert px == py, f'for msehy-py2, we must have px == py.'
        num_fc = len(numbering)
        num_cells = num_fc * px * py

        q_local_numbering = np.arange(0, (px+1) * (py+1)).reshape((px+1, py+1), order='F')
        t_indices_top = list(q_local_numbering[0, :].ravel('F'))
        t_indices_off_top = list(q_local_numbering[1:, :].ravel('F'))
        self._top_num = px + 1
        self._triangle_num_data = px * (py+1) + 1
        self._t_indices_top = t_indices_top
        self._t_indices_off_top = t_indices_off_top

        t_offset = [3 + 3*_ for _ in range(py)]
        _sum = t_offset[-1]
        t_offset.extend([_sum + 4 + 4*_ for _ in range((px-1)*py)])

        t_cell_types = [5 for _ in range(py)] + [9 for _ in range((px-1)*py)]

        q_offset = [4 + 4*_ for _ in range(px*py)]

        t_offset = np.array(t_offset)
        q_offset = np.array(q_offset)

        q_cell_types = [9 for _ in range(px*py)]

        offset = []
        cell_types = list()
        for i in numbering:
            if isinstance(i, str):
                if offset == list():
                    offset.extend(t_offset)
                else:
                    offset.extend(offset[-1] + t_offset)
                cell_types.extend(t_cell_types)
            else:
                if offset == list():
                    offset.extend(q_offset)
                else:
                    offset.extend(offset[-1] + q_offset)
                cell_types.extend(q_cell_types)

        assert len(offset) == len(cell_types) == num_cells, f'must be!'
        offset = np.array(offset)

        cell_types = np.array(cell_types)
        self._offsets = offset
        self._cell_types = cell_types

        x, y = xy
        self._numbering = numbering
        self._x = self._flat_data_according_to_numbering(x)
        self._y = self._flat_data_according_to_numbering(y)
        self._z = np.zeros_like(self._y)

        t_local_numbering = np.zeros((px+1, py+1), dtype=int)
        t_local_numbering[1:, :] = np.arange(1, 1+px*(py+1)).reshape((px, py+1), order='F')

        t_max = np.max(t_local_numbering) + 1
        q_max = np.max(q_local_numbering) + 1

        self._connectivity = list()
        for index in numbering:
            gm = numbering[index]
            if isinstance(index, str):
                assert len(gm) == t_max, 'safety check'
                for i in range(px):
                    for j in range(py):
                        if i == 0:  # triangle cell
                            conn = np.zeros(3, dtype=int)
                            conn[0] = gm[0]
                            conn[1] = gm[t_local_numbering[i+1, j]]
                            conn[2] = gm[t_local_numbering[i+1, j+1]]
                        else:
                            conn = np.zeros(4, dtype=int)
                            conn[0] = gm[t_local_numbering[i, j]]
                            conn[1] = gm[t_local_numbering[i+1, j]]
                            conn[2] = gm[t_local_numbering[i+1, j+1]]
                            conn[3] = gm[t_local_numbering[i, j+1]]

                        self._connectivity.append(conn)

            else:
                assert len(gm) == q_max, 'safety check'
                for i in range(px):
                    for j in range(py):
                        conn = np.zeros(4, dtype=int)
                        conn[0] = gm[q_local_numbering[i, j]]
                        conn[1] = gm[q_local_numbering[i+1, j]]
                        conn[2] = gm[q_local_numbering[i+1, j+1]]
                        conn[3] = gm[q_local_numbering[i, j+1]]
                        self._connectivity.append(conn)
        self._connectivity = np.concatenate(self._connectivity)
        self._freeze()

    def __call__(self, file_path, point_data=None, cell_data=None):
        """"""
        if point_data is not None:
            assert isinstance(point_data, dict), f"please put point data in a dict."
            for data_name in point_data:
                data = point_data[data_name]
                if isinstance(data, dict) == 1:  # a scalar
                    point_data[data_name] = self._flat_data_according_to_numbering(data)
                elif len(data) == 2:  # vector data
                    vec_data: list = list()
                    for di in data:
                        vec_data.append(
                            self._flat_data_according_to_numbering(di)
                        )
                    vec_data.append(np.zeros_like(vec_data[0]))
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

    def _flat_data_according_to_numbering(self, data):
        """"""
        assert isinstance(data, dict), 'safety check!'
        for i in self._numbering:
            assert i in data, 'safety check!'
        num_dofs = self._numbering.num_dofs
        flat_data = np.zeros(num_dofs)  # 1d data

        for i in self._numbering:
            data_i = data[i].ravel('F')
            if isinstance(i, str):
                tri_data = np.zeros(self._triangle_num_data)
                tri_data[0] = sum(data_i[self._t_indices_top]) / self._top_num
                tri_data[1:] = data_i[self._t_indices_off_top]
                data_i = tri_data
            else:
                pass
            flat_data[self._numbering[i]] = data_i
        return flat_data
