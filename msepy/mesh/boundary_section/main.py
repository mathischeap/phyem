# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 4:25 PM on 5/25/2023
"""

import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msepy.mesh.boundary_section.elements import MsePyBoundarySectionElements


class MsePyBoundarySectionMesh(Frozen):
    """"""

    def __init__(self, msepy_mesh, msepy_boundary_manifold, abstract_mesh):
        """"""
        self._base = msepy_mesh
        self._manifold = msepy_boundary_manifold
        self._abstract = abstract_mesh

        assert self.abstract.manifold is self.manifold.abstract, f"safety check!"
        assert self.manifold.regions._base_regions is self.base.manifold.regions, f"safety check!"

        self._elements = None
        self._freeze()

    @property
    def base(self):
        """the base msepy mesh I am built upon."""
        return self._base

    @property
    def manifold(self):
        """the msepy manifold I am built upon."""
        return self._manifold

    @property
    def abstract(self):
        """the abstract mesh I am representing."""
        return self._abstract

    @property
    def elements(self):
        """the elements of this boundary section."""
        if self._elements is None:
            self._elements = MsePyBoundarySectionElements(self)
        return self._elements


if __name__ == '__main__':
    # python msepy/mesh/boundary_section/main.py
    import __init__ as ph
    n = 2
    ls = ph.samples.wf_div_grad(n=n, degree=8, orientation='outer', periodic=False)
    # ls.pr()

    msepy, obj = ph.fem.apply('msepy', locals())

    manifold = msepy.base['manifolds'][r"\mathcal{M}"]
    boundary_manifold = msepy.base['manifolds'][r"\partial\mathcal{M}"]
    Gamma_phi = msepy.base['manifolds'][r"\Gamma_\phi"]
    Gamma_u = msepy.base['manifolds'][r"\Gamma_u"]

    # msepy.config(manifold)(
    #     'crazy_multi', c=0.1, bounds=[[0, 1] for _ in range(n)], periodic=False,
    # )

    msepy.config(manifold)('backward_step')
    msepy.config(Gamma_phi)(manifold, {0: [1, 0, 0, 0]})

    # manifold.visualize()
    # boundary_manifold.visualize()
    # Gamma_phi.visualize()
    # Gamma_u.visualize()

    mesh = msepy.base['meshes'][r'\mathfrak{M}']
    msepy.config(mesh)([15, 15])
    # print(msepy.base['meshes'])
    # elements = Gamma_phi.elements

    mesh_phi = msepy.find_mesh_of_manifold(Gamma_phi)
    elements = mesh_phi.elements
