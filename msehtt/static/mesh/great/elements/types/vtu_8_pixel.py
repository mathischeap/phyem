# -*- coding: utf-8 -*-
r"""
"""
from phyem.msehtt.static.mesh.great.elements.types.orthogonal_rectangle import MseHttGreatMeshOrthogonalRectangleElement


class Vtu8Pixel(MseHttGreatMeshOrthogonalRectangleElement):
    r"""A pixel cell is actually an orthogonal rectangle cell."""
    def __init__(self, element_index, parameters, _map):
        r""""""
        x0, y0 = parameters[0]
        x1, y1 = parameters[1]
        x2, y2 = parameters[2]
        x3, y3 = parameters[3]

        assert x2 == x0 and y1 == y0 and x3 == x1 and y3 == y2, f"must be a pixel."

        parameters = {
            'origin': (x0, y0),
            'delta': (x1 - x0, y2 - y0)
        }

        super().__init__(element_index, parameters, _map)
