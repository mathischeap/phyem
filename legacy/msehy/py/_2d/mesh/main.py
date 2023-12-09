# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msepy.mesh.main import MsePyMesh
from msepy.mesh.boundary_section.main import MsePyBoundarySectionMesh
from msepy.main import base as msepy_base

from legacy.msehy.py._2d.mesh.elements.main import MseHyPy2MeshElements
from legacy.msehy.py._2d.mesh.boundary_faces.main import MseHyPy2MeshFaces


class MseHyPy2Mesh(Frozen):
    """It is called mesh, but it also can represent a boundary section depends on the background."""

    def __init__(self, abstract_mesh):
        self._abstract = abstract_mesh
        abstract_mesh._objective = self

        self._generation = 0
        self._previous_elements = None
        self._previous_bd_faces = None
        self._current_elements = None   # elements of this mesh
        self._current_bd_faces = None   # boundary faces of this mesh

        self._link_cache = {
            'pair': (-1, -1),
            'link': dict()
        }

        self._manifold = None
        self._freeze()

    @property
    def abstract(self):
        return self._abstract

    @property
    def manifold(self):
        """the msehy-py-2d manifold this mesh is build on."""
        if self._manifold is None:
            from legacy.msehy.py._2d.main import base
            all_manifolds = base['manifolds']
            for sym in all_manifolds:
                manifold = all_manifolds[sym]

                if self.representative.background.manifold.abstract is manifold.abstract:
                    self._manifold = manifold
                    break
                else:
                    pass
        return self._manifold

    @property
    def background(self):
        """We return it in realtime."""
        return msepy_base['meshes'][self.abstract._sym_repr]

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        if self._is_mesh():
            return f"<MseHyPy2-Mesh " + self._abstract._sym_repr + super_repr
        else:
            return f"<MseHyPy2-Boundary-Section " + self._abstract._sym_repr + super_repr

    @property
    def generation(self):
        """I am now at this generation."""
        return self._generation

    def _is_mesh(self):
        """If this mesh is representing a mesh? (otherwise, it is representing a boundary section.)"""
        if self.background.__class__ is MsePyMesh:
            return True
        else:
            assert self.background.__class__ is MsePyBoundarySectionMesh, f"must be!"
            return False

    @property
    def n(self):
        """the dimension of the mesh"""
        return 2

    @property
    def previous(self):
        """"""
        if self._is_mesh():
            return self._previous_elements
        else:
            return self._previous_bd_faces

    @property
    def representative(self):
        """"""
        if self._is_mesh():
            if self._current_elements is None:
                assert self._generation == 0, f"only 0-th generation could be not made yet"
                self._current_elements = MseHyPy2MeshElements(
                    0, self.background, None, []
                )
            return self._current_elements
        else:
            if self._current_bd_faces is None:
                assert self._generation == 0, f"only 0-th generation could be not made yet"
                from legacy.msehy.py._2d.main import base
                all_meshes = base['meshes']
                mesh = None
                for sym in all_meshes:
                    mesh = all_meshes[sym]
                    if self.background.base is mesh.background:
                        break
                    else:
                        pass

                self._current_bd_faces = MseHyPy2MeshFaces(
                    self.background,
                    mesh._current_elements
                )

            return self._current_bd_faces

    @property
    def visualize(self):
        """visualize"""
        return self.representative.visualize

    def renew(self, region_wise_refining_strength_function, refining_thresholds, evolve=1):
        """

        Parameters
        ----------
        region_wise_refining_strength_function : dict
            A dict of scalar functions. Will use abs(func).
        refining_thresholds :
            A 1-d increasing data structure of refining_thresholds[0] = 0.
        evolve:
        Returns
        -------

        """
        # -------------------------------------------------------------------------------------------
        from legacy.msehy.py._2d.main import base
        all_meshes = base['meshes']
        for sym in all_meshes:
            mesh = all_meshes[sym]
            _ = mesh.representative   # make sure the 0th generation is made.

        # ------------------------------------------------------------------------------------------
        self._generation += 1
        assert self.background.__class__ is MsePyMesh, \
            f"can only renew based on a msepy mesh background."
        # - Now we make the elements ---------------------------------------------------------------
        new_elements = MseHyPy2MeshElements(
            self._generation,
            self.background,
            region_wise_refining_strength_function,
            refining_thresholds,
        )
        self._previous_elements = self._current_elements
        self._current_elements = new_elements  # override the current elements.

        # -- renew dependent boundary section faces. ----------------------------------------------
        from msepy.mesh.boundary_section.main import MsePyBoundarySectionMesh
        for sym in all_meshes:
            mesh = all_meshes[sym]
            if mesh.background.__class__ is MsePyBoundarySectionMesh:
                if mesh.background.base == self.background:
                    # this mesh is a msehy-py2 mesh upon a dependent boundary section.
                    mesh._previous_bd_faces = mesh._current_bd_faces
                    mesh._current_bd_faces = MseHyPy2MeshFaces(
                        mesh.background,
                        self.representative
                    )
                    mesh._generation += 1
                    assert mesh._generation == self.generation, 'must be!'
            else:
                pass
        assert self.representative.generation == self.generation, 'must be!'

        # ------------------------------------------------------------------------------------------
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

    @property
    def link(self):
        """We link all elements of previous elements and current elements.

        {
            dest_element_index: None,                           # two fc be same (so they have same indices)
            dest_element_index: [source_element_index_0, ...],  # dest fc is consist of multiple source cells.
            dest_element_index: source_element_index,           # dest fc is a part of the source cell.
            ...,
        }

        Parameters
        ----------

        Returns
        -------

        """
        cur_g = self.representative.generation
        pre_g = self.previous.generation
        assert cur_g == pre_g+1, f"must be"
        assert self._is_mesh(), f"can only link elements of a msepy mesh background."

        key = (cur_g, pre_g)
        if key == self._link_cache['pair']:   # only cache the last computed link!
            return self._link_cache['link']
        else:
            self._link_cache['pair'] = key

        cur_sort = self.representative.indices_in_base_element
        pre_sort = self.previous.indices_in_base_element

        link = dict()
        for e in cur_sort:
            dest_indices = cur_sort[e]
            sour_indices = pre_sort[e]

            for dest_index in dest_indices:
                if dest_index in sour_indices:
                    link[dest_index] = None
                else:
                    if isinstance(dest_index, str):
                        if len(sour_indices) == 1:
                            assert sour_indices[0] == e, f'must be'
                            link[dest_index] = e
                        else:
                            for sour_index in sour_indices:
                                assert sour_index != dest_index
                                if sour_index in dest_index:
                                    assert dest_index not in link, f'must be!'
                                    link[dest_index] = sour_index
                                elif dest_index in sour_index:
                                    if dest_index not in link:
                                        link[dest_index] = list()
                                    else:
                                        pass
                                    link[dest_index].append(sour_index)
                                else:
                                    pass

                    else:
                        assert len(dest_indices) == 1 and dest_index == e, f'must be'
                        assert len(sour_indices) > 1, f'must be!'
                        link[dest_index] = sour_indices

        self._link_cache['link'] = link
        return link
