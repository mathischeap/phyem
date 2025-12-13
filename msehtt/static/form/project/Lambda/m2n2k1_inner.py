# -*- coding: utf-8 -*-
r"""All the projections from m2n2k1_inner.
"""
from phyem.tools.quadrature import quadrature
from phyem.msehtt.static.space.reconstruct.Lambda.Rc_m2n2k1 import ___rc221i_msepy_quadrilateral___


def to__m2n2k0(ff, ft):
    r"""Into two 0-forms. Each refers to one of the components of the outer-1-form.

    Returns
    -------
    ff :
        From form.
    ft :
        From time.

    """
    tpm = ff.tpm
    elements = tpm.composition
    cochain_component_0 = {}
    cochain_component_1 = {}

    ff_cochain = ff[ft].cochain
    degree = ff.degree
    for e in elements:
        element = elements[e]
        etype = element.etype
        if etype in (
                9,
                "unique curvilinear quad",
                "orthogonal rectangle",
                "unique msepy curvilinear quadrilateral"
        ):
            cochain_component_0[e], cochain_component_1[e] = ___to_220_msepy_quadrilateral___(
                element, ff_cochain[e], degree
            )
        else:
            raise NotImplementedError()
    return cochain_component_0, cochain_component_1


def ___to_220_msepy_quadrilateral___(element, ff_local_cochain, degree):
    r""""""

    p, btype = element.degree_parser(degree)
    nodes = list()
    for _, bt in zip(p, btype):
        nodes.append(quadrature(_, bt).quad[0])
    _, _, u, v = ___rc221i_msepy_quadrilateral___(element, degree, ff_local_cochain, *nodes, ravel=True)
    return u, v
