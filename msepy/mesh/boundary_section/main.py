# -*- coding: utf-8 -*-
r"""
"""

import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msepy.mesh.boundary_section.faces import MsePyBoundarySectionFaces
from msepy.mesh.boundary_section.coordinate_transformation import MsePyBoundarySectionMeshCooTrans
from msepy.mesh.boundary_section.visualize.main import MsePyBoundarySectionVisualize


class MsePyBoundarySectionMesh(Frozen):
    """"""

    def __init__(self, msepy_mesh, msepy_boundary_manifold, abstract_mesh):
        """"""
        self._base = msepy_mesh
        self._manifold = msepy_boundary_manifold
        self._abstract = abstract_mesh

        assert self.abstract.manifold is self.manifold.abstract, f"safety check!"
        assert self.manifold.regions._base_regions is self.base.manifold.regions, f"safety check!"

        self._faces = None
        self._ct = None
        self._visualize = MsePyBoundarySectionVisualize(self)
        self._freeze()

    def __repr__(self):
        """"""
        base_repr = self.base.__repr__()
        self_repr = '<BoundarySection ' + self.abstract._sym_repr + ' of '
        return self_repr + base_repr + '>'

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
    def faces(self):
        """the elements of this boundary section."""
        if self._faces is None:
            self._faces = MsePyBoundarySectionFaces(self)
        return self._faces

    @property
    def n(self):
        """ndim"""
        return self._base.n - 1

    @property
    def m(self):
        """esd"""
        return self._base.m

    @property
    def visualize(self):
        return self._visualize

    @property
    def ct(self):
        """"""
        if self._ct is None:
            self._ct = MsePyBoundarySectionMeshCooTrans(self)
        return self._ct


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
    #     'crazy_multi.rst', c=0.1, bounds=[[0, 1] for _ in range(n)], periodic=False,
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
    print(elements._elements)
    print(elements[0])
