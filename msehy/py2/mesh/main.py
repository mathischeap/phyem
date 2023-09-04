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


class MseHyPy2Mesh(Frozen):
    """It is called mesh, but it also can represent a boundary section depends on the background."""

    def __init__(self, abstract_mesh):
        self._abstract = abstract_mesh
        self.___most_recent_generation___ = 0
        self.___current_elements___ = MseHyPy2MeshElements(
            0, self.background, None, []
        )  # initialize the elements as a not-refined one.
        self.___current_faces___ = MseHyPy2MeshFaces(
            self.background,
            self.___current_elements___,
        )

        self.___last_elements___ = None
        self.___last_faces___ = None

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

    def _is_boundary_section(self):
        """"""
        return self.background.__class__ is MsePyBoundarySectionMesh

    @property
    def abstract(self):
        return self._abstract

    @property
    def background(self):
        """We return it in realtime."""
        return msepy_base['meshes'][self.abstract._sym_repr]

    @property
    def current_elements(self):
        """"""
        assert self.background.__class__ is MsePyMesh, \
            f"Only meshes access to elements."
        return self.___current_elements___

    @property
    def current_faces(self):
        assert self.background.__class__ is MsePyBoundarySectionMesh, \
            f"Only boundary sections access to elements."
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
        self.___most_recent_generation___ += 1
        assert self.background.__class__ is MsePyMesh, \
            f"can only renew based on a msepy mesh background."
        # - Now we make the elements -----------------------------------------------------------------
        new_elements = MseHyPy2MeshElements(
            self.___most_recent_generation___,
            self.background,
            region_wise_refining_strength_function,
            refining_thresholds
        )
        new_faces = MseHyPy2MeshFaces(
            self.background,
            new_elements,
        )  # to be implemented

        self.___last_elements___ = self.___current_elements___    # save last elements
        self.___last_faces___ = self.___current_faces___          # save last faces

        self.___current_elements___ = new_elements  # override the current elements.
        self.___current_faces___ = new_faces        # override the current faces.


if __name__ == '__main__':
    # python msehy/py2/mesh/main.py
    import __init__ as ph

    space_dim = 2
    ph.config.set_embedding_space_dim(space_dim)

    manifold = ph.manifold(space_dim, is_periodic=False)
    mesh = ph.mesh(manifold)
    mesh.boundary_partition(r"\Gamma_\perp", r"\Gamma_P")

    msehy, obj = ph.fem.apply('msehy', locals())

    manifold = msehy.base['manifolds'][r'\mathcal{M}']
    mesh = msehy.base['meshes'][r'\mathfrak{M}']

    msehy.config(manifold)('crazy', c=0., bounds=([-1, 1], [-1, 1]))

    # manifold.background.visualize()

    Gamma_perp = msehy.base['manifolds'][r"\Gamma_\perp"]

    msehy.config(Gamma_perp)(
        manifold, {
            0: [1, 0, 1, 0],
        }
    )
    msehy.config(mesh)([20, 20])    # element layout

    # for msh in msehy.base['meshes']:
    #     msh = msehy.base['meshes'][msh]
    #     msh.background.visualize()
    #     print(msh)

    def refining_strength(x, y):
        """"""
        return np.sin(2*np.pi*x) * np.cos(2*np.pi*y)

    mesh.renew(
        {0: refining_strength}, [0.3, 0.5, 0.7, 0.9]
    )

    # # for msh in msehy.base['meshes']:
    # #     msh = msehy.base['meshes'][msh]
    # #     print(msh)
    current_elements = mesh.current_elements
    # # print(current_elements.thresholds)
    #
    levels = current_elements.levels
    # # print(levels[1].num)
    # triangles = levels[0].triangles
    #
    # for i in triangles:
    #     triangle = triangles[i]
    #     # p2 = triangle.pair_to
    #     # if isinstance(p2, str):
    #     #     print(i, elements[p2].pair_to)
    #     print(i, triangle.angle_degree)

    mesh.current_elements.visualize()
    #
    # print(len(current_elements), current_elements.num_levels)
    # print(current_elements.map)
    # for i in current_elements:
    #     print(i, current_elements[i].ct)

    # for level in levels:
    #     print(len(level.triangles))
