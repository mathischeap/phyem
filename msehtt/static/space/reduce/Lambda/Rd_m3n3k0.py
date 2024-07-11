# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.quadrature import quadrature


def reduce_Lambda__m3n3k0(cf_t, tpm, degree):
    """Reduce target at time `t` to m3n3k0 space of degree ``degree`` on partial mesh ``tpm``."""

    elements = tpm.composition
    cochain = {}
    for e in elements:
        element = elements[e]
        etype = element.etype
        if etype in ("orthogonal hexahedron", ):
            cochain[e] = ___330_msepy_quadrilateral___(element, cf_t, degree)
        else:
            raise NotImplementedError()
    return cochain


def ___330_msepy_quadrilateral___(element, cf_t, degree):
    """"""
    p, btype = element.degree_parser(degree)
    nodes = [quadrature(_, btype).quad[0] for _ in p]
    xi, et, sg = np.meshgrid(*nodes, indexing='ij')
    xi = xi.ravel('F')
    et = et.ravel('F')
    sg = sg.ravel('F')
    x, y, z = element.ct.mapping(xi, et, sg)
    u = cf_t(x, y, z)[0]
    return u
