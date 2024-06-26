# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msepy.space.gathering_matrix.main import MsePyGatheringMatrix
from msepy.space.incidence_matrix.main import MsePyIncidenceMatrix
from msepy.space.mass_matrix.main import MsePyMassMatrix
from msepy.space.trace_matrix.main import MsePyTraceMatrix
from msepy.space.wedge_matrix.main import MsePyWedgeMatrix
from msepy.space.inner_product.main import MsePyInnerProduct
from msepy.space.local_numbering.main import MsePyLocalNumbering
from msepy.space.num_local_dofs.main import MsePyNumLocalDofs
from msepy.space.num_local_dof_components.main import MsePyNumLocalDofComponents
from msepy.space.basis_functions.main import MsePyBasisFunctions
from msepy.space.reduce.main import MsePySpaceReduce
from msepy.space.reconstruct.main import MsePySpaceReconstruct
from msepy.space.error.main import MsePySpaceError
from msepy.space.norm.main import MsePySpaceNorm
from msepy.space.degree import PySpaceDegree
from msepy.space.reconstruction_matrix.main import MsePyReconstructMatrix

from msepy.mesh.main import MsePyMesh

from msepy.space.find.main import MsePySpaceFind

from src.spaces.finite import SpaceFiniteSetting


class MsePySpace(Frozen):
    """"""

    def __init__(self, abstract_space):
        """"""
        self._abstract = abstract_space
        abstract_mesh = abstract_space.mesh
        mesh = abstract_mesh._objective
        abstract_space._objective = self   # this is important, we use it for making forms.
        assert mesh.__class__ is MsePyMesh, f"mesh type wrong."
        self._mesh = mesh
        self._finite = SpaceFiniteSetting(self)  # this is a necessary attribute for a particular space.
        self._degree_cache = {}
        self._local_numbering = None
        self._gathering_matrix = None
        self._incidence_matrix = None
        self._mass_matrix = None
        self._trace_matrix = None
        self._wedge_matrix = None
        self._basis_functions = None
        self._num_local_dofs = None
        self._num_local_dof_components = None
        self._reduce = None
        self._reconstruct = None
        self._reconstruction_matrix = None
        self._error = None
        self._norm = None
        self._ip = None
        self._find = None
        self._freeze()

    @property
    def abstract(self):
        return self._abstract

    def __repr__(self):
        """repr"""
        ab_space_repr = self.abstract.__repr__().split(' at ')[0][1:]
        return '<MsePy ' + ab_space_repr + super().__repr__().split('object')[1]

    @property
    def esd(self):
        return self.abstract.mesh.manifold.esd

    @property
    def ndim(self):
        return self.abstract.mesh.ndim

    @property
    def m(self):
        return self.esd

    @property
    def n(self):
        return self.ndim

    @property
    def mesh(self):
        """The mesh"""
        return self._mesh

    @property
    def manifold(self):
        """The manifold."""
        return self._mesh.manifold

    @property
    def finite(self):
        """The finite setting."""
        return self._finite

    def __getitem__(self, degree):
        """"""
        key = str(degree)
        if key in self._degree_cache:
            pass
        else:
            self._degree_cache[key] = PySpaceDegree(self, degree)
        return self._degree_cache[key]

    @property
    def local_numbering(self):
        """local numbering"""
        if self._local_numbering is None:
            self._local_numbering = MsePyLocalNumbering(self)
        return self._local_numbering

    @property
    def num_local_dofs(self):
        """local numbering"""
        if self._num_local_dofs is None:
            self._num_local_dofs = MsePyNumLocalDofs(self)
        return self._num_local_dofs

    @property
    def num_local_dof_components(self):
        """local numbering"""
        if self._num_local_dof_components is None:
            self._num_local_dof_components = MsePyNumLocalDofComponents(self)
        return self._num_local_dof_components

    @property
    def incidence_matrix(self):
        """incidence_matrix"""
        if self._incidence_matrix is None:
            self._incidence_matrix = MsePyIncidenceMatrix(self)
        return self._incidence_matrix

    @property
    def mass_matrix(self):
        """incidence_matrix"""
        if self._mass_matrix is None:
            self._mass_matrix = MsePyMassMatrix(self)
        return self._mass_matrix

    @property
    def trace_matrix(self):
        """trace matrix."""
        if self._trace_matrix is None:
            self._trace_matrix = MsePyTraceMatrix(self)
        return self._trace_matrix

    @property
    def wedge_matrix(self):
        """incidence_matrix"""
        if self._wedge_matrix is None:
            self._wedge_matrix = MsePyWedgeMatrix(self)
        return self._wedge_matrix

    @property
    def gathering_matrix(self):
        """gathering matrix"""
        if self._gathering_matrix is None:
            self._gathering_matrix = MsePyGatheringMatrix(self)
        return self._gathering_matrix

    @property
    def basis_functions(self):
        if self._basis_functions is None:
            self._basis_functions = MsePyBasisFunctions(self)
        return self._basis_functions

    @property
    def reduce(self):
        if self._reduce is None:
            self._reduce = MsePySpaceReduce(self)
        return self._reduce

    @property
    def reconstruct(self):
        if self._reconstruct is None:
            self._reconstruct = MsePySpaceReconstruct(self)
        return self._reconstruct

    @property
    def reconstruction_matrix(self):
        """"""
        if self._reconstruction_matrix is None:
            self._reconstruction_matrix = MsePyReconstructMatrix(self)
        return self._reconstruction_matrix

    @property
    def error(self):
        """norm"""
        if self._error is None:
            self._error = MsePySpaceError(self)
        return self._error

    @property
    def norm(self):
        """norm"""
        if self._norm is None:
            self._norm = MsePySpaceNorm(self)
        return self._norm

    @property
    def find(self):
        if self._find is None:
            self._find = MsePySpaceFind(self)
        return self._find

    @property
    def inner_product(self):
        if self._ip is None:
            self._ip = MsePyInnerProduct(self)
        return self._ip
