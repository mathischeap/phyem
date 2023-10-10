# -*- coding: utf-8 -*-
r"""
"""
from src.config import RANK, MASTER_RANK, COMM
from tools.frozen import Frozen
from msehy.py._2d.mesh.boundary_faces.main import MseHyPy2MeshFaces
from _MPI.generic.py._2d_unstruct.mesh.boundary_section.main import MPI_Py_2D_Unstructured_BoundarySection


class Generic_BoundarySection_Maker(Frozen):
    """"""

    def __init__(self, serial_generic_faces):

        from _MPI.msehy.py._2d.main import base
        all_meshes = base['meshes']
        if RANK == MASTER_RANK:
            assert serial_generic_faces.__class__ is MseHyPy2MeshFaces, \
                f"Must have a serial msehy mesh faces in the master"
            mesh = None
            sym = None
            for sym in all_meshes:
                mesh = all_meshes[sym]
                if mesh.background.representative is serial_generic_faces._current_elements:
                    break

            assert mesh is not None, f"must have found a mesh"

            self._including_element_faces = serial_generic_faces._including_element_faces
        else:
            assert serial_generic_faces is None, f"we must receive None in non-master cores."
            sym = None
            self._including_element_faces = list()

        sym = COMM.bcast(sym, root=MASTER_RANK)
        mesh = all_meshes[sym]
        base_elements = mesh.generic
        self._base_elements = base_elements
        self._freeze()

    def __call__(self):
        """"""
        return MPI_Py_2D_Unstructured_BoundarySection(
            self._base_elements,
            self._including_element_faces,
        )
