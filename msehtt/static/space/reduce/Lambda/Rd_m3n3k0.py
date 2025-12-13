# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.tools.quadrature import quadrature


def reduce_Lambda__m3n3k0(cf_t, tpm, degree, element_range=None):
    """Reduce target at time `t` to m3n3k0 space of degree ``degree`` on partial mesh ``tpm``."""

    elements = tpm.composition
    cochain = {}

    if element_range is None:
        ELEMENT_RANGE = elements
    else:
        ELEMENT_RANGE = element_range

    for e in ELEMENT_RANGE:
        element = elements[e]
        etype = element.etype
        if etype in ("orthogonal hexahedron", "unique msepy curvilinear hexahedron"):
            cochain[e] = ___330_msepy_quadrilateral___(element, cf_t, degree)
        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    return cochain


def ___330_msepy_quadrilateral___(element, cf_t, degree):
    """"""
    p, btype = element.degree_parser(degree)
    nodes = list()
    for _, bt in zip(p, btype):
        nodes.append(quadrature(_, bt).quad[0])
    xi, et, sg = np.meshgrid(*nodes, indexing='ij')
    xi = xi.ravel('F')
    et = et.ravel('F')
    sg = sg.ravel('F')
    x, y, z = element.ct.mapping(xi, et, sg)
    u = cf_t(x, y, z)[0]
    return u
