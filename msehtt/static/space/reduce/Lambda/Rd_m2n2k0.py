# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.quadrature import quadrature


def reduce_Lambda__m2n2k0(cf_t, tpm, degree, element_range=None):
    r"""Reduce target at time `t` to m2n2k1 outer space of degree ``degree`` on partial mesh ``tpm``."""

    elements = tpm.composition
    cochain = {}

    if element_range is None:
        ELEMENT_RANGE = elements
    else:
        ELEMENT_RANGE = element_range

    for e in ELEMENT_RANGE:
        element = elements[e]
        etype = element.etype
        if etype in (
                9,
                "orthogonal rectangle",
                "unique msepy curvilinear quadrilateral",
                'unique curvilinear quad',
        ):
            cochain[e] = ___220_msepy_quadrilateral___(element, cf_t, degree)

        elif etype in (
                5,
                "unique msepy curvilinear triangle",
        ):
            cochain[e] = ___220_vtu_5___(element, cf_t, degree)

        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")

    return cochain


def ___220_msepy_quadrilateral___(element, cf_t, degree):
    r""""""
    p, btype = element.degree_parser(degree)
    nodes = [quadrature(_, btype).quad[0] for _ in p]
    xi, et = np.meshgrid(*nodes, indexing='ij')
    xi = xi.ravel('F')
    et = et.ravel('F')
    x, y = element.ct.mapping(xi, et)
    u = cf_t(x, y)[0]
    return u


def ___220_vtu_5___(element, cf_t, degree):
    r""""""
    p, btype = element.degree_parser(degree)
    nodes = [quadrature(_, btype).quad[0] for _ in p]
    xi, et = np.meshgrid(*nodes, indexing='ij')

    xi0 = xi[0, 0]
    et0 = et[0, 0]

    xi = xi[1:, :].ravel('F')
    et = et[1:, :].ravel('F')

    xi = np.concatenate([
        np.array([xi0]), xi
    ])

    et = np.concatenate([
        np.array([et0]), et
    ])

    x, y = element.ct.mapping(xi, et)
    u = cf_t(x, y)[0]
    return u
