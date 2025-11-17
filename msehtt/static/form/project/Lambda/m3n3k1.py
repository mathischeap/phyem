# -*- coding: utf-8 -*-
r"""All the projections from m3n3k1-forms.
"""
from phyem.tools.quadrature import quadrature
from phyem.msehtt.static.space.reconstruct.Lambda.Rc_m3n3k1 import ___rc331_orthogonal_hexahedron___


def to__m3n3k0(ff, ft):
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
    cochain_component_2 = {}

    ff_cochain = ff[ft].cochain
    degree = ff.degree
    for e in elements:
        element = elements[e]
        etype = element.etype
        if etype in ("orthogonal hexahedron", ):
            cochain_component_0[e], cochain_component_1[e], cochain_component_2[e] = (
                ___to_330_orthogonal_hexahedron___(
                    element, ff_cochain[e], degree
                )
            )
        else:
            raise NotImplementedError()
    return cochain_component_0, cochain_component_1, cochain_component_2


def ___to_330_orthogonal_hexahedron___(element, ff_local_cochain, degree):
    r""""""

    p, btype = element.degree_parser(degree)
    nodes = [quadrature(_, btype).quad[0] for _ in p]
    _, _, _, u, v, w = ___rc331_orthogonal_hexahedron___(element, degree, ff_local_cochain, *nodes, ravel=True)
    return u, v, w
