# -*- coding: utf-8 -*-
r"""
"""
import sys
if './' not in sys.path:
    sys.path.append('./')

import numpy as np
from tools.frozen import Frozen
from msepy.mesh.main import MsePyMesh
from msepy.mesh.boundary_section.main import MsePyBoundarySectionMesh
from msepy.main import base as msepy_base
from msehy.py2.mesh.elements.main import MseHyPy2MeshElements
from msehy.py2.mesh.faces.main import MseHyPy2MeshFaces
from msehy.py2.mesh.generations import _MesHyPy2MeshGenerations


class MseHyPy2Mesh(Frozen):
    """It is called mesh, but it also can represent a boundary section depends on the background."""

    def __init__(self, abstract_mesh):
        self._abstract = abstract_mesh
        abstract_mesh._objective = self
        self.___current_generation___ = 0
        self.___current_elements___ = None
        self.___current_faces___ = None
        self.___generations___ = _MesHyPy2MeshGenerations(self)
        self._link_cache = {
            'pair': (-1, -1),
            'link': dict()
        }
        self._freeze()

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        if self._is_mesh():
            return f"<MseHyPy2-Mesh " + self._abstract._sym_repr + super_repr
        else:
            return f"<MseHyPy2-Boundary-Section " + self._abstract._sym_repr + super_repr

    def _is_mesh(self):
        """"""
        return self.background.__class__ is MsePyMesh

    @property
    def abstract(self):
        return self._abstract

    @property
    def background(self):
        """We return it in realtime."""
        return msepy_base['meshes'][self.abstract._sym_repr]

    @property
    def visualize(self):
        """visualization object."""
        return self.current_representative.visualize

    def _set_max_generation_cache(self, max_generations):
        """"""
        assert max_generations % 1 == 0 and max_generations >= 2, \
            f"max_generations={max_generations} wrong; it must be an integer >= 2"
        self.generations._max_generations = max_generations

    @property
    def generations(self):
        """"""
        return self.___generations___

    def __len__(self):
        """How many generations I am caching."""
        return len(self.generations._pool)

    def __contains__(self, g):
        """If the generation #g is cached."""
        return g in self.generations

    def _pg(self, generation):
        """"""
        if isinstance(generation, (int, float)):
            assert generation % 1 == 0, f"generation = {generation} wrong, must be integer."
            if generation >= 0:
                pass
            else:  # generation < 0
                cg = self.current_generation
                generation = cg + generation + 1
        elif generation is None:
            generation = self.current_generation
        else:
            raise Exception(f"{generation}")

        assert generation >= 0 and generation % 1 == 0, f"generation={generation} wrong!"
        return generation

    def __getitem__(self, generation):
        """Get the representative of `generation`."""
        generation = self._pg(generation)
        return self.generations[generation]

    @property
    def current_generation(self):
        """Generation."""
        return self.___current_generation___

    @property
    def g(self):
        """short-cut of current generation."""
        return self.current_generation

    @property
    def current_representative(self):
        """"""
        if self._is_mesh():
            return self.current_elements
        else:
            return self.current_faces

    @property
    def current_elements(self):
        """"""
        assert self.background.__class__ is MsePyMesh, \
            f"Only meshes access to elements."

        if self.___current_elements___ is None:
            # the 0-generation is not made yet!
            assert self.___current_generation___ == 0, f"only 0-th generation could be not made yet"
            self.___current_elements___ = MseHyPy2MeshElements(
                0, self.background, None, []
            )  # initialize the elements as a not-refined one.
            self.generations._add(self.___current_elements___)

        return self.___current_elements___

    @property
    def current_faces(self):
        assert self.background.__class__ is MsePyBoundarySectionMesh, \
            f"Only boundary sections access to elements."
        if self.___current_faces___ is None:
            # the 0-generation is not made yet!
            assert self.___current_generation___ == 0, f"only 0-th generation could be not made yet"
            from msehy.py2.main import base
            all_meshes = base['meshes']
            mesh = None
            for sym in all_meshes:
                mesh = all_meshes[sym]
                if self.background.base is mesh.background:
                    break
                else:
                    pass

            self.___current_faces___ = MseHyPy2MeshFaces(
                self.background,
                mesh.current_elements
            )
            assert self.___current_faces___.generation == 0, f'Must be!'

            self.generations._add(self.___current_faces___)

        return self.___current_faces___

    def renew(self, region_wise_refining_strength_function, refining_thresholds):
        """

        Parameters
        ----------
        region_wise_refining_strength_function : dict
            A dict of scalar functions. Will use abs(func).
        refining_thresholds :
            A 1-d increasing data structure of refining_thresholds[0] = 0.
        Returns
        -------

        """
        # -------------------------------------------------------------------------------------------
        from msehy.py2.main import base
        all_meshes = base['meshes']
        for sym in all_meshes:
            mesh = all_meshes[sym]
            _ = mesh.current_representative   # make sure the 0th generation is made.

        # ------------------------------------------------------------------------------------------
        self.___current_generation___ += 1
        assert self.background.__class__ is MsePyMesh, \
            f"can only renew based on a msepy mesh background."
        # - Now we make the elements -----------------------------------------------------------------
        new_elements = MseHyPy2MeshElements(
            self.___current_generation___,
            self.background,
            region_wise_refining_strength_function,
            refining_thresholds
        )
        self.___current_elements___ = new_elements  # override the current elements.
        self.generations._add(self.current_representative)
        # -- renew dependent boundary section faces. --------------------------------------------------
        from msepy.mesh.boundary_section.main import MsePyBoundarySectionMesh
        for sym in all_meshes:
            mesh = all_meshes[sym]
            if mesh.background.__class__ is MsePyBoundarySectionMesh:
                if mesh.background.base == self.background:
                    # this mesh is a msehy-py2 mesh upon a dependent boundary section.
                    mesh.___current_faces___ = MseHyPy2MeshFaces(
                        mesh.background,
                        self.current_elements
                    )
                    mesh.generations._add(mesh.current_representative)
                    mesh.___current_generation___ += 1
                    assert mesh.current_generation == self.current_generation, 'must be!'
            else:
                pass
        assert self.current_representative.generation == self.current_generation, 'must be!'

    def link(self, dest_g, source_g):
        """We link all fundamental cells of dest_g to fundamental cells of source_g.

        {
            dest_fc_index: None,                                         # two fc be same (so they have same indices)
            dest_fc_index: [source_fc_index_0, source_fc_index_1, ...],  # dest fc is consist of multiple source cells.
            dest_fc_index: source_fc_index,                              # dest fc is a part of the source cell.
            ...
        }

        Parameters
        ----------
        dest_g
        source_g

        Returns
        -------

        """
        dest_g = self._pg(dest_g)
        source_g = self._pg(source_g)
        assert dest_g != source_g, f"cannot link to itself!"
        assert self.background.__class__ is MsePyMesh, f"can only link elements of a msepy mesh background."
        assert dest_g in self, f"destination generation {dest_g} is not available."
        assert source_g in self, f"source generation {source_g} is not available."

        key = (dest_g, source_g)
        if key == self._link_cache['pair']:   # only cache the last computed link!
            return self._link_cache['link']
        else:
            self._link_cache['pair'] = key

        dest = self[dest_g]
        sour = self[source_g]
        dest_sort = dest.sort(order='background')
        sour_sort = sour.sort(order='background')

        link = dict()
        for e in dest_sort:
            dest_indices = dest_sort[e]
            sour_indices = sour_sort[e]
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

    
if __name__ == '__main__':
    # python msehy/py2/mesh/main.py
    import __init__ as ph

    space_dim = 2
    ph.config.set_embedding_space_dim(space_dim)

    # manifold = ph.manifold(space_dim, is_periodic=True)
    manifold = ph.manifold(space_dim, is_periodic=False)
    mesh = ph.mesh(manifold)

    # mesh.boundary_partition(r"\Gamma_\perp", r"\Gamma_P")

    msehy, obj = ph.fem.apply('msehy', locals())
    manifold = msehy.base['manifolds'][r'\mathcal{M}']
    mesh = msehy.base['meshes'][r'\mathfrak{M}']

    # msehy.config(manifold)('crazy', c=0., bounds=([-1, 1], [-1, 1]), periodic=True)
    msehy.config(manifold)('crazy', c=0., bounds=([-1, 1], [-1, 1]), periodic=False)
    # Gamma_perp = msehy.base['manifolds'][r"\Gamma_\perp"]
    # msehy.config(Gamma_perp)(
    #     manifold, {
    #         0: [1, 0, 0, 0],
    #     }
    # )

    msehy.config(mesh)([7, 7])    # element layout

    # msh = msehy.base['meshes'][msh]
    # for msh in msehy.base['meshes']:
    #     msh = msehy.base['meshes'][msh]
    #     cr = msh.current_representative
    # print(cr)
    # print(msh)
    # mesh.visualize()

    def refining_strength(x, y):
        """"""
        return np.sin(2*np.pi*x) * np.cos(2*np.pi*y)

    mesh.renew(
        {0: refining_strength}, [0.3, 0.5]
    )
    mesh.visualize()
    # mesh.renew(
    #     {0: refining_strength}, [0.3, 0.5, 0.7, 0.9]
    # )
    # mesh.visualize()
    # MAP = mesh.current_representative.map
    # for i in MAP:
    #     print(i, MAP[i])

    # # for msh in msehy.base['meshes']:
    # #     msh = msehy.base['meshes'][msh]
    # #     print(msh)
    # current_elements = mesh.current_elements
    # # print(current_elements.thresholds)
    # levels = current_elements.levels
    # # print(levels[1].num)
    # triangles = levels[0].triangles
    # for i in triangles:
    #     triangle = triangles[i]
    #     # p2 = triangle.pair_to
    #     # if isinstance(p2, str):
    #     #     print(i, elements[p2].pair_to)
    #     print(i, triangle.angle_degree)
    #
    # print(len(current_elements), current_elements.num_levels)
    # print(current_elements.map)
    # for i in current_elements:
    #     print(i, current_elements[i].ct)
    # for level in levels:
    #     print(len(level.triangles))

    # for msh in msehy.base['meshes']:
    #     msh = msehy.base['meshes'][msh]
    #     msh.visualize()
    #
    # for msh in msehy.base['meshes']:
    #     msh = msehy.base['meshes'][msh]
    #     # msh.visualize()
    #     print(msh.generations[-2])
