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
from msehy.py2.mesh.boundary_section.faces import MseHyPy2BoundarySectionFaces


class MseHyPy2Mesh(Frozen):
    """It is called mesh, but it also can represent a boundary section depends on the background."""

    def __init__(self, abstract_mesh):
        self._abstract = abstract_mesh
        self.___generation___ = 0
        self.___elements___ = MseHyPy2MeshElements(
            self, None, []
        )  # initialize the elements as a not-refined one.
        self.___faces___ = None
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
    def elements(self):
        """"""
        assert self.background.__class__ is MsePyMesh, f"Only meshes access to elements."
        return self.___elements___

    @property
    def faces(self):
        assert self.background.__class__ is MsePyBoundarySectionMesh, f"Only boundary sections access to elements."
        if self.___faces___ is None:
            self.___faces___ = MseHyPy2BoundarySectionFaces(self)
        return self.___faces___

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
        assert self.background.__class__ is MsePyMesh, f"can only renew based on a msepy mesh background."
        # - Now we make the elements ------------------------------------------------------------------------
        elements = MseHyPy2MeshElements(self, region_wise_refining_strength_function, refining_thresholds)

        # --- renew elements --------------------------------------------------------------------------------
        self.___elements___ = elements
        self.___generation___ += 1
        # --- renew all boundary sections -------------------------------------------------------------------
        self._renew_boundary_sections()

    def _renew_boundary_sections(self):
        """"""
        from msehy.py2.main import base
        all_meshes = base['meshes']
        boundary_sections_2b_refined = list()
        for sym in all_meshes:
            mesh = all_meshes[sym]
            if mesh.background.__class__ is MsePyBoundarySectionMesh:
                if mesh.background.base is self.background:
                    boundary_sections_2b_refined.append(mesh)
            else:
                pass
        # to implement the refining of boundary sections. For example,
        for bs in boundary_sections_2b_refined:
            bs.faces._renew()


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

    msehy.config(manifold)('crazy')

    # manifold.background.visualize()

    Gamma_perp = msehy.base['manifolds'][r"\Gamma_\perp"]

    msehy.config(Gamma_perp)(
        manifold, {
            0: [1, 0, 1, 0],
        }
    )
    msehy.config(mesh)(5)    # element layout

    for msh in msehy.base['meshes']:
        msh = msehy.base['meshes'][msh]
        # msh.background.visualize()
        # print(msh.background)

    def refining_strength(x, y):
        """"""
        return np.sin(2*np.pi*x) * np.cos(2*np.pi*y)

    mesh.renew(
        {0: refining_strength}, [0.3, 0.7]
    )

    # for msh in msehy.base['meshes']:
    #     msh = msehy.base['meshes'][msh]
    #     print(msh)
    print(mesh.elements.levels[0].elements)
