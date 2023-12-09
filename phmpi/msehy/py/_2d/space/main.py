# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.config import RANK
from phmpi.msehy.py._2d.mesh.main import MPI_MseHy_Py2_Mesh
from phmpi.generic.py._2d_unstruct.space.main import MPI_Py_2D_Unstructured_Space

from legacy.msehy.py._2d.space.refine import Refine
from legacy.msehy.py._2d.space.coarsen import Coarsen


class MPI_MseHy_Py2_Space(Frozen):
    """"""

    def __init__(self, abstract_space):
        """"""
        self._abstract = abstract_space
        abstract_space._objective = self   # this is important, we use it for making forms.
        abstract_mesh = abstract_space.mesh
        self._mesh = abstract_mesh._objective
        assert self.mesh.__class__ is MPI_MseHy_Py2_Mesh, \
            f"shell mesh type {self.mesh} wrong."

        # -- checks and initialize --------------------------------------------------------------------------
        assert self.mesh.generation == 0, \
            f'When initialize msehy-py2 space, the msehy-py mesh must be on G0'
        self._generation = None
        self._previous = None
        self._generic = None
        self._do_initialize = True

        self._coarsen = Coarsen(self)
        self._refine = Refine(self)
        self._freeze()

    @property
    def abstract(self):
        return self._abstract

    @property
    def mesh(self):
        """mesh"""
        return self._mesh

    def __repr__(self):
        """repr"""
        ab_space_repr = self.abstract.__repr__().split(' at ')[0][1:]
        return f'<MseHy-py2 @RANK{RANK} ' + ab_space_repr + super().__repr__().split('object')[1]

    @property
    def refine(self):
        return self._refine

    @property
    def coarsen(self):
        return self._coarsen

    # ----------- generic ------------------------------------------------------------------------
    @property
    def generation(self):
        if self._do_initialize:
            self._initialize()
        return self._generation

    @property
    def generic(self):
        if self._do_initialize:
            self._initialize()
        return self._generic

    @property
    def previous(self):
        if self._do_initialize:
            self._initialize()
        return self._previous

    def _initialize(self):
        """update generation, generic, previous according to the newest generation of the mesh."""
        if self._do_initialize:
            generic = self.mesh.generic
            previous = self.mesh.previous
            self._generic = MPI_Py_2D_Unstructured_Space(generic, self.abstract)
            self._generation = self.mesh.generation
            if previous is None:
                assert self._generation == 0, 'must be!'
            else:
                self._previous = MPI_Py_2D_Unstructured_Space(previous, self.abstract)
            self._do_initialize = False
        else:
            pass

    def _update(self):
        """update according to the most recent generation of the mesh."""
        if self._do_initialize:
            self._initialize()
        else:
            if self.mesh.generation == self.generation:
                pass
            else:
                old_generation = self.generation
                assert self.mesh.generation == self.generation + 1, \
                    f"mesh generation must only be 1-gen ahead."
                generic = self.mesh.generic
                self._previous = self._generic
                self._generic = MPI_Py_2D_Unstructured_Space(generic, self.abstract)
                assert self._previous is not self._generic, 'must be'
                self._generation = self.mesh.generation
                assert self._generation == old_generation + 1, f'must be!'

    # -------------------------------------------------------------------------------------------------
    def mass_matrix(self, degree):
        return self.generic.mass_matrix(degree)

    def incidence_matrix(self, degree):
        return self.generic.incidence_matrix(degree)
