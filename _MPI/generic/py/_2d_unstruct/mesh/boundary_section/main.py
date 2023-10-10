# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.config import RANK, MASTER_RANK, COMM
from _MPI.generic.py._2d_unstruct.mesh.elements.main import MPI_Py_2D_Unstructured_MeshElements
from _MPI.generic.py._2d_unstruct.mesh.boundary_section.coordinate_transformation import BS_CT


class MPI_Py_2D_Unstructured_BoundarySection(Frozen):
    """"""

    def __init__(self, base_elements, including_element_faces):
        assert base_elements.__class__ is MPI_Py_2D_Unstructured_MeshElements, \
            f"must based on an elements-class."

        including_element_faces = COMM.gather(including_element_faces, root=MASTER_RANK)
        if RANK == MASTER_RANK:
            _all = list()
            for _ in including_element_faces:
                _all.extend(_)
            including_element_faces = _all
        else:
            pass
        including_element_faces = COMM.bcast(including_element_faces, root=MASTER_RANK)
        local_element_faces = list()
        for ef in including_element_faces:
            element_index = ef[0]
            if element_index in base_elements:
                local_element_faces.append(ef)
        self._base = base_elements
        self._indices = local_element_faces  # the element-face-indices that involved in this rank.
        self._ct = BS_CT(self)
        self._freeze()

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        return rf"<MPI-generic-py-un-struct-2d-BoundarySection @ RANK {RANK}" + super_repr

    @property
    def ct(self):
        """coordinate transformation."""
        return self._ct

    @property
    def base(self):
        """The base generic mpi version 2d mesh elements."""
        return self._base

    def __iter__(self):
        """go through all indices of local faces."""
        for index in self._indices:
            yield index

    def __contains__(self, index):
        """If this index indicating a local face?"""
        return index in self._indices

    def __len__(self):
        """How many local faces?"""
        return len(self._indices)

    def __getitem__(self, index):
        """"""
        return self._base._make_boundary_face(index)
