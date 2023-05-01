# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
"""
import numpy as np
import sys
if './' not in sys.path:
    sys.path.append('./')

from tools.frozen import Frozen
from msepy.mesh.visualize.matplot import MsePyMeshVisualizeMatplot
from msepy.mesh.visualize.vtk_ import MsePyMeshVisualizeVTK


class MsePyMeshVisualize(Frozen):
    """"""

    def __init__(self, mesh):
        self._mesh = mesh
        self._matplot = MsePyMeshVisualizeMatplot(mesh)
        self._vtk = MsePyMeshVisualizeVTK(mesh)
        self._freeze()

    def __call__(self, *args, **kwargs):
        """"""
        return self.matplot(*args, **kwargs)

    @property
    def matplot(self):
        return self._matplot

    @property
    def vtk(self):
        return self._vtk

    def _generate_mesh_grid_data(
            self,
            sampling_factor=1,
    ):
        """"""
        if sampling_factor <= 0.1:
            sampling_factor = 0.1
        samples = 2 ** self._mesh.m * 50000 * sampling_factor
        samples = int((np.ceil(samples / self._mesh.elements._num))**(1/self._mesh.m))
        if samples >= 100:
            samples = 100
        elif samples < 2:
            samples = 2
        else:
            samples = int(samples)

        ndim = self._mesh.ndim
        linspace = np.linspace(0, 1, samples)
        Nodes = self._mesh.elements._nodes
        Lines = dict()
        for i in Nodes:  # region #i
            nodes = Nodes[i]
            assert ndim == len(nodes), f"trivial check."
            Lines[i] = list()    # the lines in region #i,
            if len(nodes) == 1:   # mesh ndim == 1
                # we do not use region mapping for this because we want to see the line segments.
                linspace_segment = np.array([-1, 1])

                coo_lines = self._mesh.ct.mapping(linspace_segment, regions=i)

                Lines[i].append(coo_lines)

            elif len(nodes) == 2:   # mesh ndim == 2

                nodes0, nodes1 = nodes

                axis0_rst = np.meshgrid(linspace, nodes1, indexing='ij')
                axis1_rst = np.meshgrid(nodes0, linspace, indexing='ij')

                axis0_lines = self._mesh.manifold.ct.mapping(*axis0_rst, regions=i)[i]
                axis1_lines = self._mesh.manifold.ct.mapping(*axis1_rst, regions=i)[i]

                Lines[i].append(axis0_lines)
                Lines[i].append(axis1_lines)

            elif len(nodes) == 3:   # mesh ndim == 3
                nodes0, nodes1, nodes2 = nodes

                axis0_rst = np.meshgrid(linspace, nodes1, nodes2, indexing='ij')
                axis1_rst = np.meshgrid(nodes0, linspace, nodes2, indexing='ij')
                axis2_rst = np.meshgrid(nodes0, nodes1, linspace, indexing='ij')

                axis0_lines = self._mesh.manifold.ct.mapping(*axis0_rst, regions=i)[i]
                axis1_lines = self._mesh.manifold.ct.mapping(*axis1_rst, regions=i)[i]
                axis2_lines = self._mesh.manifold.ct.mapping(*axis2_rst, regions=i)[i]

                Lines[i].append(axis0_lines)
                Lines[i].append(axis1_lines)
                Lines[i].append(axis2_lines)

        return Lines


if __name__ == '__main__':
    # python msepy/mesh/visualize/main.py
    import __init__ as ph
    space_dim = 2
    ph.config.set_embedding_space_dim(space_dim)

    manifold = ph.manifold(space_dim)
    mesh = ph.mesh(manifold)

    msepy, obj = ph.fem.apply('msepy', locals())

    mnf = obj['manifold']
    msh = obj['mesh']

    # msepy.config(mnf)('crazy', c=0., periodic=True, bounds=[[0, 2] for _ in range(space_dim)])
    msepy.config(mnf)('backward_step')
    msepy.config(msh)([3 for _ in range(space_dim)])

    # msh.visualize()
    # print(msh.elements._layout_cache_key)
    msh.visualize()
