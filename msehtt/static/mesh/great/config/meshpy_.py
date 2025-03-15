# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen

try:
    import meshpy.triangle as triangle
except ModuleNotFoundError:
    pass

from msehtt.static.mesh.great.config.vtu import MseHttVtuInterface


class MseHtt_API_2_MeshPy(Frozen):
    r""""""

    def __init__(self, **kwargs):
        r""""""

        if len(kwargs) == 1 and 'points' in kwargs:
            case = 0
            # case 0: only provide the outline points of the mesh.
            # it must be a 2d mesh, we can build the facets automatically.
            # All points are in a sequence around the domain.
            # the mesh will be triangular mesh.
            pass
        else:
            raise NotImplementedError()
        self._case = case
        self.___kwargs___ = kwargs
        self._freeze()

    def __call__(self, **kwargs):
        r"""build the mesh"""

        if self._case == 0:
            return self.___case_0_parser___(**kwargs)
        else:
            raise NotImplementedError

    def ___case_0_parser___(self, max_volume=None,):
        r"""Only provided points. Then must be 2d mesh. All points are in a sequence around the domain.
        The mesh will be triangular mesh.

        The elements will be almost equally distributed. So no local refinement can be applied for this case.

        Parameters
        ----------
        max_volume :
            Determine the mesh density. The lower, the denser.

        """

        points = self.___kwargs___['points']
        seq = range(len(points))
        facets = ___round_trip_connect___(seq)

        info = triangle.MeshInfo()
        info.set_points(points)
        info.set_facets(facets)
        mesh = triangle.build(info, max_volume=max_volume)

        mesh_points = np.array(mesh.points)
        mesh_tris = np.array(mesh.elements)

        coo = {}
        connections = {}
        cell_types = {}

        for i, ___ in enumerate(mesh_points):
            coo[i] = ___

        for element_index, connection in enumerate(mesh_tris):
            cell_types[element_index] = 5  # all triangles
            connections[element_index] = [int(_) for _ in connection]

        api2vtu = MseHttVtuInterface(
            coo, connections, cell_types,
            redistribute=True
        )
        return api2vtu


def ___round_trip_connect___(seq):
    r""""""
    result = []
    for i in range(len(seq)):
        result.append((seq[i], seq[(i + 1) % len(seq)]))
    return result
