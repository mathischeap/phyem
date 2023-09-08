# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.spaces.finite import SpaceFiniteSetting
from msehy.py2.mesh.main import MseHyPy2Mesh
from msepy.space.degree import PySpaceDegree

from msehy.py2.space.gathering_matrix.main import MseHyPy2GatheringMatrix
from msehy.py2.space.local_numbering.main import MseHyPy2LocalNumbering
from msehy.py2.space.basis_functions.main import MseHyPy2BasisFunctions
from msehy.py2.space.num_local_dofs.main import MseHyPy2NumLocalDofs
from msehy.py2.space.num_local_dof_components.main import MseHyPy2NumLocalDofComponents
from msehy.py2.space.local_dof_representative_coordinates.main import MseHyPy2LocalDofsRepresentativeCoo
from msehy.py2.space.reduce.main import MseHyPySpaceReduce
from msehy.py2.space.reconstruct.main import MseHyPy2SpaceReconstruct
from msehy.py2.space.error.main import MseHyPy2SpaceError
from msehy.py2.space.norm.main import MseHyPy2SpaceNorm


class MseHyPy2Space(Frozen):
    """"""

    def __init__(self, abstract_space):
        """"""
        self._abstract = abstract_space
        abstract_mesh = abstract_space.mesh
        mesh = abstract_mesh._objective
        abstract_space._objective = self   # this is important, we use it for making forms.
        assert mesh.__class__ is MseHyPy2Mesh, f"mesh type {mesh} wrong."
        self._mesh = mesh
        self._finite = SpaceFiniteSetting(self)  # this is a necessary attribute for a particular space.
        self._degree_cache = {}
        self._gm = None
        self._ln = None
        self._bfs = None
        self._num_local_dofs = None
        self._num_local_dof_components = None
        self._local_dof_rc = None
        self._reduce = None
        self._reconstruct = None
        self._error = None
        self._norm = None
        self._freeze()

    @property
    def abstract(self):
        return self._abstract

    def __repr__(self):
        """repr"""
        ab_space_repr = self.abstract.__repr__().split(' at ')[0][1:]
        return '<MseHy-py2 ' + ab_space_repr + super().__repr__().split('object')[1]

    @property
    def finite(self):
        """The finite setting."""
        return self._finite

    @property
    def mesh(self):
        return self._mesh

    def _pg(self, generation):
        return self.mesh._pg(generation)

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
    def manifold(self):
        """The manifold."""
        return self._mesh.manifold

    def __getitem__(self, degree):
        """"""
        key = str(degree)
        if key in self._degree_cache:
            pass
        else:
            assert isinstance(degree, int) and degree > 0, \
                f"msehy-py2 can only accept integer degree that > 0."
            self._degree_cache[key] = PySpaceDegree(self, degree)
        return self._degree_cache[key]

    @property
    def gathering_matrix(self):
        """Gathering matrix; generation dependent"""
        if self._gm is None:
            self._gm = MseHyPy2GatheringMatrix(self)
        return self._gm

    @property
    def local_numbering(self):
        """local numbering; generation in-dependent."""
        if self._ln is None:
            self._ln = MseHyPy2LocalNumbering(self)
        return self._ln

    @property
    def basis_functions(self):
        """local numbering; generation in-dependent."""
        if self._bfs is None:
            self._bfs = MseHyPy2BasisFunctions(self)
        return self._bfs

    @property
    def num_local_dofs(self):
        """local numbering; generation in-dependent."""
        if self._num_local_dofs is None:
            self._num_local_dofs = MseHyPy2NumLocalDofs(self)
        return self._num_local_dofs

    @property
    def num_local_dof_components(self):
        """local numbering; generation in-dependent."""
        if self._num_local_dof_components is None:
            self._num_local_dof_components = MseHyPy2NumLocalDofComponents(self)
        return self._num_local_dof_components

    @property
    def local_dof_representative_coordinates(self):
        if self._local_dof_rc is None:
            self._local_dof_rc = MseHyPy2LocalDofsRepresentativeCoo(self)
        return self._local_dof_rc

    @property
    def reduce(self):
        """Reduction; generation dependent"""
        if self._reduce is None:
            self._reduce = MseHyPySpaceReduce(self)
        return self._reduce

    @property
    def reconstruct(self):
        """Reconstruction; generation dependent"""
        if self._reconstruct is None:
            self._reconstruct = MseHyPy2SpaceReconstruct(self)
        return self._reconstruct

    @property
    def error(self):
        """Reconstruction; generation dependent"""
        if self._error is None:
            self._error = MseHyPy2SpaceError(self)
        return self._error

    @property
    def norm(self):
        """Reconstruction; generation dependent"""
        if self._norm is None:
            self._norm = MseHyPy2SpaceNorm(self)
        return self._norm
