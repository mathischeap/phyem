# -*- coding: utf-8 -*-
r"""
"""
from legacy.generic.py._2d_unstruct.mesh.elements.regular_triangle import RegularTriangle
from legacy.generic.py._2d_unstruct.mesh.elements.regular_quadrilateral import RegularQuadrilateral

_global_element_cache = {
    'rt': {},
    'rq': {},
}


def distributor(element_type):
    """"""
    if element_type == 'rt':
        return RegularTriangle
    elif element_type == 'rq':
        return RegularQuadrilateral
    else:
        raise Exception(f"element_type = {element_type} is not implemented.")


def distributor_with_cache(element_type, coordinates):
    """"""
    coordinates_key = str(coordinates)
    element_pool = _global_element_cache[element_type]
    if coordinates_key in element_pool:
        element = element_pool[coordinates_key]

    else:
        element = distributor(element_type)(coordinates)
        element_pool[coordinates_key] = element
    return element
