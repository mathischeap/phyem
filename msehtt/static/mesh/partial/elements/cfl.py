# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from src.config import RANK, MASTER_RANK, COMM
from pyevtk.hl import unstructuredGridToVTK

from tools.frozen import Frozen
from msehtt.static.form.addons.static import MseHttFormStaticCopy


class MseHtt_PartialMesh_Elements_CFL_condition(Frozen):
    """"""

    def __init__(self, elements):
        """"""
        self._elements = elements
        self._freeze()

    @property
    def mn(self):
        return self._elements.mn

    def __call__(self, filename, form, dt):
        """"""
        self._visualize(filename, form, dt)

    def _visualize(self, filename, form, dt):
        if self.mn == (2, 2):
            self._visualize_m2n2_(filename, form, dt)
        else:
            raise NotImplementedError()

    def _visualize_m2n2_(self, filename, form, dt, ddf=1):
        """"""
        assert form.__class__ is MseHttFormStaticCopy, \
            f"Must be a form static copy."

        space_indicator = form._f.space.str_indicator
        if space_indicator == 'm2n2k1_outer':
            pass
        else:
            raise NotImplementedError(space_indicator)
        density = int(8 * ddf)
        if density < 4:
            density = 4
        elif density > 18:
            density = 18
        else:
            pass
        _a_ = np.array([-1, 1])

        linspace = np.linspace(-1, 1, density)
        linspace = (linspace[1:] + linspace[:-1]) / 2

        _, UV = form.reconstruct(linspace, linspace, ravel=True)
        U, V = UV

        CELL_LIST = []
        COO_DICT = {}
        for e in U:

            element = self._elements[e]
            etype = element.etype

            if etype == 'orthogonal rectangle':
                hx, hy = element._parameters['delta']

                u = max(np.abs(U[e]))
                v = max(np.abs(V[e]))

                cfl_x = u * dt / hx
                cfl_y = v * dt / hy

                cfl = max([cfl_x, cfl_y])

                coo_dict, cell_list = element._generate_element_vtk_data_(_a_, _a_)
                assert len(cell_list) == 1, f"each element refers to only one cfl number."
                cell_list = cell_list[0]
                cell_list += (cfl,)
                CELL_LIST.append(cell_list)
                COO_DICT.update(coo_dict)
            else:
                raise NotImplementedError()

        CELL_LIST = COMM.gather(CELL_LIST, root=MASTER_RANK)
        COO_DICT = COMM.gather(COO_DICT, root=MASTER_RANK)
        if RANK != MASTER_RANK:
            return
        else:
            pass

        cell_list = list()
        for _ in CELL_LIST:
            cell_list.extend(_)
        CELL_LIST = cell_list

        coo_dict = {}
        for _ in COO_DICT:
            coo_dict.update(_)
        COO_DICT = coo_dict

        number_points = len(COO_DICT)
        COO_NUMBERING = {}
        for i, index in enumerate(COO_DICT):
            COO_NUMBERING[index] = i

        X = np.zeros(number_points)
        Y = np.zeros(number_points)
        Z = np.zeros(number_points)
        for key in COO_NUMBERING:
            index = COO_NUMBERING[key]
            X[index], Y[index] = COO_DICT[key]

        connections = list()
        offset_current = 0
        offsets = list()
        cell_types = list()
        CFL = np.zeros(len(CELL_LIST))

        for i, cell in enumerate(CELL_LIST):
            cell_connection, cell_offset, cell_type, cfl = cell
            cell_connection = [COO_NUMBERING[_] for _ in cell_connection]
            offset_current += cell_offset
            connections.extend(cell_connection)
            offsets.append(offset_current)
            cell_types.append(cell_type)
            CFL[i] = cfl

        connections = np.array(connections)
        offsets = np.array(offsets)
        cell_types = np.array(cell_types)

        unstructuredGridToVTK(
            filename,
            X, Y, Z,
            connectivity=connections, offsets=offsets, cell_types=cell_types,
            cellData={'cfl': CFL}
        )
