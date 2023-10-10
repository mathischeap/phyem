# -*- coding: utf-8 -*-
r"""
"""
from src.config import RANK, MASTER_RANK
from tools.frozen import Frozen
from msehy.py._2d.mesh.elements.main import MseHyPy2MeshElements
from _MPI.generic.py._2d_unstruct.mesh.elements.main import MPI_Py_2D_Unstructured_MeshElements


class Generic_Elements_Maker(Frozen):
    """"""

    def __init__(self, serial_generic_mesh):
        """"""
        if RANK == MASTER_RANK:
            assert serial_generic_mesh.__class__ is MseHyPy2MeshElements, \
                f"Must have a serial msehy-py2 mesh elements in the master"
        else:
            assert serial_generic_mesh is None, f"we must receive None in non-master cores."
        sgm = serial_generic_mesh
        if RANK == MASTER_RANK:
            type_dict, vertex_dict, vertex_coordinates, same_vertices_dict \
                = sgm._make_generic_element_input_dict(sgm._indices)
        else:
            type_dict, vertex_dict, vertex_coordinates, same_vertices_dict = {}, {}, {}, {}
        self._inputs = type_dict, vertex_dict, vertex_coordinates, same_vertices_dict
        self._freeze()

    def __call__(self):
        """"""
        return MPI_Py_2D_Unstructured_MeshElements(*self._inputs)
