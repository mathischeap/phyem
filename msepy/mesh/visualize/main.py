# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from msepy.mesh.visualize.matplot import MsePyMeshVisualizeMatplot
from msepy.mesh.visualize.vtk_ import MsePyMeshVisualizeVTK
from msepy.mesh.visualize.target import MsePyMeshVisualizeTarget


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

    def _target(self, function, sampling_factor=1, **kwargs):
        """We plot the function on this mesh

        Parameters
        ----------
        function
            Can be called like ``function(*coo)`` and return a list of components. If it returns a scalar, then
            it should be like `[s, ]`.

            ``coo`` are the coordinates in each mesh element.
        kwargs

        Returns
        -------

        """
        return MsePyMeshVisualizeTarget(self._mesh)(function, sampling_factor=sampling_factor, **kwargs)

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
