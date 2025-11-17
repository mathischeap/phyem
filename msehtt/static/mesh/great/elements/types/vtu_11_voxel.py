# -*- coding: utf-8 -*-
r"""
"""
from phyem.msehtt.static.mesh.great.elements.types.orthogonal_hexahedron import MseHttGreatMeshOrthogonalHexahedronElement


class Vtu_11_Voxel(MseHttGreatMeshOrthogonalHexahedronElement):
    """A voxel cell is actually an orthogonal hexahedron cell."""
    def __init__(self, element_index, parameters, _map):
        """"""
        x0, y0, z0 = parameters[0]
        x1, y1, z1 = parameters[1]
        x2, y2, z2 = parameters[2]
        x3, y3, z3 = parameters[3]
        x4, y4, z4 = parameters[4]
        x5, y5, z5 = parameters[5]
        x6, y6, z6 = parameters[6]
        x7, y7, z7 = parameters[7]

        assert x2 == x0 == x4 == x6, f"must be a voxel."
        assert x1 == x3 == x5 == x7, f"must be a voxel."
        assert y0 == y1 == y4 == y5, f"must be a voxel."
        assert y2 == y3 == y6 == y7, f"must be a voxel."
        assert z0 == z1 == z2 == z3, f"must be a voxel."
        assert z4 == z5 == z6 == z7, f"must be a voxel."

        parameters = {
            'origin': (x0, y0, z0),
            'delta': (x1 - x0, y2 - y0, z4 - z0)
        }

        super().__init__(element_index, parameters, _map)
