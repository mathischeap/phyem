# -*- coding: utf-8 -*-
r"""
"""
from generic.py._2d_unstruct.mesh.elements.regular_triangle import RegularTriangle
from generic.py._2d_unstruct.mesh.elements.regular_quadrilateral import RegularQuadrilateral


def distributor(element_type):
    """"""
    if element_type == 'rt':
        return RegularTriangle
    elif element_type == 'rq':
        return RegularQuadrilateral
    else:
        raise Exception(f"element_type = {element_type} is not implemented.")
