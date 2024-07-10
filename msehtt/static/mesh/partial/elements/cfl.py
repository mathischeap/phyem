# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from src.config import RANK, MASTER_RANK, COMM
from pyevtk.hl import unstructuredGridToVTK

from tools.frozen import Frozen
from msehtt.static.form.addons.static import MseHttFormStaticCopy
from tools.dds.region_wise_structured import DDSRegionWiseStructured


def _merge_dict_(data, root=MASTER_RANK):
    """"""
    assert isinstance(data, dict)
    DATA = COMM.gather(data, root=root)
    if RANK == root:
        data = {}
        for _ in DATA:
            data.update(_)
        return data
    else:
        return None


class MseHtt_PartialMesh_Elements_CFL_condition(Frozen):
    """With this property, we can compute cfl number based on a form on this partial mesh of elements.

    The cfl results can be saved as a vtk file or a rws data structure or else.
    """

    def __init__(self, elements):
        """"""
        self._elements = elements
        self._freeze()

    @property
    def mn(self):
        return self._elements.mn

    def __call__(self, filename, form, dt):
        """"""
        self.export_rws(filename, form, dt)

    def export_vtk(self, filename, form, dt):
        if self.mn == (2, 2):
            self._export_vtk_m2n2_(filename, form, dt)
        else:
            raise NotImplementedError(f"cfl export_vtk not implemented for mn == {self.mn}")

    def export_rws(self, filename, form, dt):
        """"""
        if self.mn == (2, 2):
            self._export_rws_m2n2_(filename, form, dt)
        else:
            raise NotImplementedError(f"cfl export_rws not implemented for mn == {self.mn}")

    def _export_rws_m2n2_(self, filename, form, dt, ddf=1):
        """"""
        assert form.__class__ is MseHttFormStaticCopy, \
            f"Must export cfl vtk from a form static copy."

        space_indicator = form._f.space.str_indicator
        if space_indicator in ('m2n2k1_outer', 'm2n2k1_inner'):
            pass
        else:
            raise NotImplementedError(f"cannot compute cfl number for form of {space_indicator}")
        density = int(33 * ddf)
        if density < 17:
            density = 17
        elif density > 99:
            density = 99
        else:
            pass

        linspace = np.linspace(-1, 1, density)
        linspace = (linspace[1:] + linspace[:-1]) / 2
        _, UV = form.reconstruct(linspace, linspace, ravel=True)
        U, V = UV

        element_cfl_dict = {}
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

            else:
                raise NotImplementedError()

            element_cfl_dict[e] = cfl

        _a_ = np.array([-1, 1])
        coo_rct = form.reconstruct(_a_, _a_, ravel=False)[0]
        x, y = coo_rct

        element_cfl_dict = _merge_dict_(element_cfl_dict, root=MASTER_RANK)
        x = _merge_dict_(x, root=MASTER_RANK)
        y = _merge_dict_(y, root=MASTER_RANK)
        if RANK != MASTER_RANK:
            return
        else:
            pass

        for e in element_cfl_dict:
            cfl = element_cfl_dict[e]
            element_cfl_dict[e] = np.ones_like(x[e]) * cfl

        dds_rws = DDSRegionWiseStructured([x, y], [element_cfl_dict, ])
        dds_rws.saveto(filename)

    def _export_vtk_m2n2_(self, filename, form, dt, ddf=1):
        """"""
        assert form.__class__ is MseHttFormStaticCopy, \
            f"Must export cfl vtk from a form static copy."

        space_indicator = form._f.space.str_indicator
        if space_indicator in ('m2n2k1_outer', 'm2n2k1_inner'):
            pass
        else:
            raise NotImplementedError(f"cannot compute cfl number for form of {space_indicator}")
        density = int(33 * ddf)
        if density < 17:
            density = 17
        elif density > 99:
            density = 99
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
