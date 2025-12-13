# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from phyem.msehtt.static.mesh.great.elements.types.orthogonal_hexahedron import (
    MseHttGreatMeshOrthogonalHexahedronElement)


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

        check_array = np.sum(np.abs(np.array([
            [x2 - x0, x2 - x4, x2 - x6],
            [x1 - x3, x1 - x5, x1 - x7],
            [y0 - y1, y0 - y4, y0 - y5],
            [y2 - y3, y2 - y6, y2 - y7],
            [z0 - z1, z0 - z2, z0 - z3],
            [z4 - z5, z4 - z6, z4 - z7]]
        )))
        np.testing.assert_almost_equal(check_array, 0)

        parameters = {
            'origin': (x0, y0, z0),
            'delta': (x1 - x0, y2 - y0, z4 - z0)
        }

        super().__init__(element_index, parameters, _map)
