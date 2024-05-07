# -*- coding: utf-8 -*-
"""
"""
import numpy as np
from tools.quadrature import quadrature


def reduce_Lambda__m2n2k0(cf_t, tpm, degree):
    """Reduce target at time `t` to m2n2k1 outer space of degree ``degree`` on partial mesh ``tpm``."""

    elements = tpm.composition
    cochain = {}
    for e in elements:
        element = elements[e]
        etype = element.etype
        if etype in ("orthogonal rectangle", "unique msepy curvilinear quadrilateral"):
            cochain[e] = ___220_msepy_quadrilateral___(element, cf_t, degree)
        else:
            raise NotImplementedError()
    return cochain


def ___220_msepy_quadrilateral___(element, cf_t, degree):
    """"""
    p, btype = element.degree_parser(degree)
    nodes = [quadrature(_, btype).quad[0] for _ in p]
    xi, et = np.meshgrid(*nodes, indexing='ij')
    xi = xi.ravel('F')
    et = et.ravel('F')
    x, y = element.ct.mapping(xi, et)
    u = cf_t(x, y)[0]
    return u
