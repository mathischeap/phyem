# -*- coding: utf-8 -*-
r"""
"""

import numpy as np
from pyevtk.hl import unstructuredGridToVTK

from phyem.src.config import RANK, MASTER_RANK, COMM


def ___vtk_m3n3_partial_mesh_elements___(elements, saveto=None):
    """"""
    if saveto is None:
        saveto = 'partial_mesh_elements'
    else:
        pass

    xi = et = sg = np.array([-1, 1])

    COO_DICT = {}
    CELL_LIST = list()

    for e in elements:
        element = elements[e]
        coo_dict, cell_list = element._generate_element_vtk_data_(xi, et, sg)
        COO_DICT.update(coo_dict)
        CELL_LIST.extend(cell_list)

    COO_DICT = COMM.gather(COO_DICT, root=MASTER_RANK)
    CELL_LIST = COMM.gather(CELL_LIST, root=MASTER_RANK)

    if RANK != MASTER_RANK:
        return
    else:
        pass

    # ------ NOW ONLY IN THE MASTER RANK --------------------------------------------
    coo_dict = {}
    for _ in COO_DICT:
        coo_dict.update(_)
    COO_DICT = coo_dict

    cell_list = list()
    for _ in CELL_LIST:
        cell_list.extend(_)
    CELL_LIST = cell_list

    numbering_dict = {}
    for i, key in enumerate(COO_DICT):
        numbering_dict[key] = i

    number_nodes = len(numbering_dict)
    X = np.zeros(number_nodes)
    Y = np.zeros(number_nodes)
    Z = np.zeros(number_nodes)
    for key in numbering_dict:
        index = numbering_dict[key]
        X[index], Y[index], Z[index] = COO_DICT[key]

    connections = list()
    offset_current = 0
    offsets = list()
    cell_types = list()

    for cell in CELL_LIST:
        cell_connection, cell_offset, cell_type = cell
        cell_connection = [numbering_dict[_] for _ in cell_connection]
        offset_current += cell_offset
        connections.extend(cell_connection)
        offsets.append(offset_current)
        cell_types.append(cell_type)

    connections = np.array(connections)
    offsets = np.array(offsets)
    cell_types = np.array(cell_types)

    unstructuredGridToVTK(
        saveto,
        X, Y, Z,
        connectivity=connections, offsets=offsets, cell_types=cell_types,
    )
