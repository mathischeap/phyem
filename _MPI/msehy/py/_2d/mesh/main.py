# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehy.py._2d.main import base as msehy_base
from src.config import RANK, MASTER_RANK, COMM
from _MPI.msehy.py._2d.mesh.elements_maker import Generic_Elements_Maker
from _MPI.msehy.py._2d.mesh.boundary_section_maker import Generic_BoundarySection_Maker


class MPI_MseHy_Py2_Mesh(Frozen):
    """"""

    def __init__(self, abstract_mesh):
        """"""
        self._abstract = abstract_mesh
        abstract_mesh._objective = self

        self._previous_elements = None
        self._previous_bd_faces = None
        self._current_elements = None   # elements of this mesh
        self._current_bd_faces = None   # boundary faces of this mesh

        self._manifold = None
        self.___is_mesh___ = None
        self._freeze()

    @property
    def abstract(self):
        """"""
        return self._abstract

    @property
    def background(self):
        """We return it in realtime."""
        if RANK == MASTER_RANK:
            return msehy_base['meshes'][self.abstract._sym_repr]
        else:
            return None

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        return f"<{self.__class__.__name__} {self.abstract._sym_repr} @ RANK {RANK}" + super_repr

    @property
    def manifold(self):
        """Return the msehy-py-2d manifold this mesh is build on."""
        if self._manifold is None:
            from _MPI.msehy.py._2d.main import base
            all_manifolds = base['manifolds']
            if RANK == MASTER_RANK:
                the_manifold_sym = None
                for sym in all_manifolds:
                    manifold = all_manifolds[sym]
                    if self.abstract.manifold is manifold.abstract:
                        the_manifold_sym = sym
                        break
                    else:
                        pass
                assert the_manifold_sym is not None, f"must have found a manifold."
            else:
                the_manifold_sym = None
            the_manifold_sym = COMM.bcast(the_manifold_sym, root=MASTER_RANK)
            self._manifold = all_manifolds[the_manifold_sym]
        return self._manifold

    @property
    def generation(self):
        """"""
        if RANK == MASTER_RANK:
            generation = self.background.generation
        else:
            generation = None
        generation = COMM.bcast(generation, root=MASTER_RANK)
        return generation

    def _is_mesh(self):
        """"""
        if self.___is_mesh___ is None:
            if RANK == MASTER_RANK:
                is_mesh = self.background._is_mesh()
            else:
                is_mesh = None
            self.___is_mesh___ = COMM.bcast(is_mesh, root=MASTER_RANK)
        return self.___is_mesh___
    
    @property
    def generic(self):
        """The current generic mesh of mpi-msehy-py2 mesh."""
        if self._is_mesh():
            if self._current_elements is None:
                if RANK == MASTER_RANK:
                    maker = Generic_Elements_Maker(self.background.representative)
                else:
                    maker = Generic_Elements_Maker(None)
                self._current_elements = maker()
            else:
                pass
            return self._current_elements
        else:
            if self._current_bd_faces is None:
                if RANK == MASTER_RANK:
                    maker = Generic_BoundarySection_Maker(self.background.representative)
                else:
                    maker = Generic_BoundarySection_Maker(None)
                self._current_bd_faces = maker()
            else:
                pass
            return self._current_bd_faces

    @property
    def previous(self):
        """The previous generic mesh of mpi-msehy-py2 mesh."""
        if self._is_mesh():
            return self._previous_elements
        else:
            return self._previous_bd_faces

    def renew(self, region_wise_refining_strength_function, refining_thresholds, evolve=0):
        """

        Parameters
        ----------
        region_wise_refining_strength_function
        refining_thresholds
        evolve

        Returns
        -------

        """
        # -----------------------------------------------------------------------------------
        from _MPI.msehy.py._2d.main import base
        all_meshes = base['meshes']
        for sym in all_meshes:
            mesh = all_meshes[sym]
            _ = mesh.generic

        if RANK == MASTER_RANK:
            self.background.renew(
                # this will make sure the background is a mesh rather than boundary section.
                region_wise_refining_strength_function,
                refining_thresholds,
                evolve=0,  # turn off the `form-renew` there.
            )
        else:
            pass

        self._previous_elements = self._current_elements
        if RANK == MASTER_RANK:
            maker = Generic_Elements_Maker(self.background.representative)
        else:
            maker = Generic_Elements_Maker(None)
        self._current_elements = maker()

        # -----------------------------------------------------------------------------------
        for sym in all_meshes:
            mesh = all_meshes[sym]
            is_boundary_section = False
            if RANK == MASTER_RANK:
                background = mesh.background
                if not background._is_mesh():
                    msepy = background.background.base
                    if msepy is self.background.background:
                        is_boundary_section = True
                else:
                    pass
            else:
                is_boundary_section = None

            is_boundary_section = COMM.bcast(is_boundary_section, root=MASTER_RANK)
            if is_boundary_section:
                mesh._previous_bd_faces = mesh._current_bd_faces
                if RANK == MASTER_RANK:
                    maker = Generic_BoundarySection_Maker(mesh.background.representative)
                else:
                    maker = Generic_BoundarySection_Maker(None)
                mesh._current_bd_faces = maker()
            else:
                pass

        # ======================================================================================
        all_forms = base['forms']
        for sym in all_forms:
            form = all_forms[sym]
            if form._is_base():
                form._update()  # update all form automatically to the newest generation.
                if evolve > 0:
                    form.evolve(amount_of_cochain=evolve)
            else:
                pass

        for sym in all_forms:
            form = all_forms[sym]
            if form._is_base():
                pass
            else:
                form._update()  # update all form automatically to the newest generation.
                _ = form.generic._base  # make sure base form is correctly linked.

    def visualize(self, title=None, **kwargs):
        """"""
        if title is None:
            title = rf"${self.abstract._sym_repr}$"
        else:
            pass
        return self.generic.visualize(title=title, **kwargs)
