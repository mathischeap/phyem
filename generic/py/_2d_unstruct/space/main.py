# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from random import random
from time import time

from generic.py._2d_unstruct.mesh.main import GenericUnstructuredMesh2D
from _MPI.generic.py._2d_unstruct.mesh.elements.main import MPI_Py_2D_Unstructured_MeshElements

from generic.py._2d_unstruct.space.degree import Py2SpaceDegree

from generic.py._2d_unstruct.space.reduce.main import Reduce
from generic.py._2d_unstruct.space.reconstruct.main import Reconstruct
from generic.py._2d_unstruct.space.num_local_dofs.main import NumLocalDofs
from generic.py._2d_unstruct.space.num_local_dof_components.main import NumLocalDofComponents
from generic.py._2d_unstruct.space.local_numbering.main import LocalNumbering
from generic.py._2d_unstruct.space.basis_functions.main import BasisFunctions
from generic.py._2d_unstruct.space.gathering_matrix.main import GatheringMatrix
from generic.py._2d_unstruct.space.incidence_matrix.main import IncidenceMatrix
from generic.py._2d_unstruct.space.mass_matrix.main import MassMatrix
from generic.py._2d_unstruct.space.find.main import Find
from generic.py._2d_unstruct.space.error.main import Error
from generic.py._2d_unstruct.space.norm.main import Norm
from generic.py._2d_unstruct.space.reconstruction_matrix.main import ReconstructMatrix


class GenericUnstructuredSpace2D(Frozen):
    """"""

    def __init__(self, mesh, abstract_space):
        """"""
        assert mesh.__class__ in (
            GenericUnstructuredMesh2D,
            MPI_Py_2D_Unstructured_MeshElements,
        ), f"I must be built on a proper mesh class."

        self._mesh = mesh
        self._abstract = abstract_space
        self._signature = str(random()+time())[-10:]
        self._degree_cache = {}

        self._reduce = Reduce(self)
        self._reconstruct = Reconstruct(self)
        self._num_local_dofs = NumLocalDofs(self)
        self._num_local_dof_components = NumLocalDofComponents(self)
        self._local_numbering = LocalNumbering(self)
        self._bfs = BasisFunctions(self)
        self._gm = GatheringMatrix(self)
        self._E = IncidenceMatrix(self)
        self._M = MassMatrix(self)
        self._find = Find(self)
        self._error = Error(self)
        self._norm = Norm(self)
        self._rm = ReconstructMatrix(self)

        self._freeze()

    @property
    def mesh(self):
        return self._mesh

    @property
    def abstract(self):
        return self._abstract

    def __repr__(self):
        """Repr"""
        return f"<Py2-Space of {self.abstract} at {self._signature}>"

    @property
    def n(self):
        """the dimensions of the space."""
        return 2

    @property
    def reduce(self):
        return self._reduce

    @property
    def reconstruct(self):
        return self._reconstruct

    def __getitem__(self, degree):
        """"""
        key = str(degree)
        if key in self._degree_cache:
            pass
        else:
            assert isinstance(degree, int) and degree > 0, \
                f"msehy-py2 can only accept integer degree that > 0."
            self._degree_cache[key] = Py2SpaceDegree(self, degree)
        return self._degree_cache[key]

    @property
    def num_local_dofs(self):
        """"""
        return self._num_local_dofs

    @property
    def num_local_dof_components(self):
        """"""
        return self._num_local_dof_components

    @property
    def local_numbering(self):
        return self._local_numbering

    @property
    def basis_functions(self):
        return self._bfs

    @property
    def gathering_matrix(self):
        return self._gm

    @property
    def incidence_matrix(self):
        return self._E

    @property
    def mass_matrix(self):
        return self._M

    @property
    def find(self):
        return self._find

    @property
    def error(self):
        return self._error

    @property
    def norm(self):
        return self._norm

    @property
    def reconstruction_matrix(self):
        return self._rm
    