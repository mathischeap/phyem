# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from pyevtk.hl import unstructuredGridToVTK

from phyem.src.config import RANK, MASTER_RANK, COMM
from phyem.msehtt.static.mesh.partial.elements.main import MseHttElementsPartialMesh


def ___ph_vtk_msehtt_static_copy___(filename, *forms, ddf=1):
    """"""
    tpm = None
    for form in forms:
        if tpm is None:
            tpm = form.tpm
        else:
            assert tpm is form.tpm, f"forms save in one file must be in the same tpm."

    assert tpm.composition.__class__ is MseHttElementsPartialMesh
    elements = tpm.composition
    # ndim = tpm.abstract.n
    # num_global_elements = elements._num_global_elements

    data_density = int(5 * ddf)
    if data_density < 3:
        data_density = 3
    elif data_density > 50:
        data_density = 50
    else:
        pass

    numbering_dict = None
    X, Y, Z = None, None, None
    connections, offsets, cell_types = None, None, None
    pointData = {}

    for ith_form, form in enumerate(forms):

        COMM.barrier()
        if form._f._is_base():
            name = form._f.abstract._pure_lin_repr
        else:
            name = form._f._base.abstract._pure_lin_repr
        form_vtk_data = {}
        form_cell_list = list()
        space = form._f.space
        indicator = space.str_indicator
        degree = form._f.degree
        dtype = None
        for element_index in elements:
            element = elements[element_index]
            element_cochain = form.cochain[element_index]
            data_dict, cell_list, element_dtype = element._generate_vtk_data_for_form(
                indicator, element_cochain, degree, data_density)
            form_vtk_data.update(data_dict)
            form_cell_list.extend(cell_list)
            if dtype is None:
                dtype = element_dtype
            else:
                assert dtype == element_dtype, f"each element must give vtk data of same type, like 2d-scalar."

        form_vtk_data = COMM.gather(form_vtk_data, root=MASTER_RANK)
        form_cell_list = COMM.gather(form_cell_list, root=MASTER_RANK)
        dtype = COMM.gather(dtype, root=MASTER_RANK)

        if RANK != MASTER_RANK:
            continue  # the remaining can be done in the master rank only.
        else:
            pass

        DTYPE = None   # the vtk data type: 2d-scalar, 2d-vector or like-wise.
        for _ in dtype:
            if _ is not None:
                if DTYPE is None:
                    DTYPE = _
                else:
                    assert DTYPE == _
            else:
                pass
        assert DTYPE is not None, f"we must have found ONE data type."

        FORM_VTK_DATA = {}
        for _ in form_vtk_data:
            FORM_VTK_DATA.update(_)
        form_vtk_data = FORM_VTK_DATA

        FORM_CELL_LIST = list()
        for _ in form_cell_list:
            FORM_CELL_LIST.extend(_)
        form_cell_list = FORM_CELL_LIST

        # ---------- get cell information: X, Y, Z, connections, offsets, cell_types --------------------
        if ith_form == 0:
            numbering_dict = {}
            for i, key in enumerate(form_vtk_data):
                numbering_dict[key] = i

            number_points = len(form_vtk_data)
            if DTYPE in ('2d-scalar', '2d-vector'):
                X = np.zeros(number_points)
                Y = np.zeros(number_points)
                Z = np.zeros(number_points)
                for key in numbering_dict:
                    index = numbering_dict[key]
                    X[index], Y[index] = form_vtk_data[key][:2]

            elif DTYPE in ('3d-scalar', '3d-vector'):
                X = np.zeros(number_points)
                Y = np.zeros(number_points)
                Z = np.zeros(number_points)
                for key in numbering_dict:
                    index = numbering_dict[key]
                    X[index], Y[index], Z[index] = form_vtk_data[key][:3]

            else:
                raise NotImplementedError()

            connections = list()
            offset_current = 0
            offsets = list()
            cell_types = list()
            for cell in form_cell_list:
                cell_connection, cell_offset, cell_type = cell
                cell_connection = [numbering_dict[_] for _ in cell_connection]
                offset_current += cell_offset
                connections.extend(cell_connection)
                offsets.append(offset_current)
                cell_types.append(cell_type)

            connections = np.array(connections)
            offsets = np.array(offsets)
            cell_types = np.array(cell_types)
        else:
            pass

        # ------------ get point data for the form -------------------------------------------------
        if DTYPE == '2d-scalar':
            scalar = np.zeros_like(X)
            for key in numbering_dict:
                index = numbering_dict[key]
                scalar[index] = form_vtk_data[key][2]
            scalar = {name: scalar}
            pointData.update(scalar)
        elif DTYPE == '2d-vector':
            u = np.zeros_like(X)
            v = np.zeros_like(X)
            w = np.zeros_like(X)
            for key in numbering_dict:
                index = numbering_dict[key]
                u[index], v[index] = form_vtk_data[key][2:4]
            vector = {name: (u, v, w)}
            pointData.update(vector)
        elif DTYPE == '3d-scalar':
            scalar = np.zeros_like(X)
            for key in numbering_dict:
                index = numbering_dict[key]
                scalar[index] = form_vtk_data[key][3]
            scalar = {name: scalar}
            pointData.update(scalar)
        elif DTYPE == '3d-vector':
            u = np.zeros_like(X)
            v = np.zeros_like(X)
            w = np.zeros_like(X)
            for key in numbering_dict:
                index = numbering_dict[key]
                u[index], v[index], w[index] = form_vtk_data[key][3:6]
            vector = {name: (u, v, w)}
            pointData.update(vector)

        else:
            raise NotImplementedError()
        # ==============================================================================================

    # ------- make the vtk file: must do it outside the for loop --------------------------------------------
    if RANK == MASTER_RANK:
        unstructuredGridToVTK(
            filename,
            X, Y, Z,
            connectivity=connections, offsets=offsets, cell_types=cell_types,
            pointData=pointData
        )
    else:
        pass
