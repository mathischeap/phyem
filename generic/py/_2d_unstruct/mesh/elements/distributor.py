# -*- coding: utf-8 -*-
r"""
"""
from generic.py._2d_unstruct.mesh.elements.regular_triangle import RegularTriangle
from generic.py._2d_unstruct.mesh.elements.regular_quadrilateral import RegularQuadrilateral


def distributor(element_type, element_coordinates):
    """"""
    if element_type == 'regular triangle':
        return RegularTriangle(element_coordinates)
    elif element_type == 'regular quadrilateral':
        return RegularQuadrilateral(element_coordinates)
    else:
        raise Exception()
