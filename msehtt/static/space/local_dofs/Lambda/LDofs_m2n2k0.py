# -*- coding: utf-8 -*-
r"""
"""
from tools.quadrature import quadrature


def LDofs_Lambda__m2n2k0(tpm, degree):
    r""""""
    LDofs = {}
    for e in tpm.composition:
        element = tpm.composition[e]
        etype = element.etype
        if etype in (
                'orthogonal rectangle',
                9,
        ):
            LDofs[e] = ___mm220_lDofs_quad___(element, degree)
        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    return LDofs


from msehtt.static.space.local_numbering.Lambda.ln_m2n2k0 import _ln_m2n2k0_msepy_quadrilateral_


def ___mm220_lDofs_quad___(element, degree):
    """"""
    p, dtype = element.degree_parser(degree)

    LN = _ln_m2n2k0_msepy_quadrilateral_(p)

    if dtype in ('Lobatto', ):
        nodes = quadrature(p, category=dtype).quad_nodes
    else:
        raise NotImplementedError(dtype)

    nodes0, nodes1 = nodes

    px, py = p

    lDofs_info = {}

    for j in range(py + 1):
        for i in range(px + 1):
            dof_local_numbering = LN[i, j]
            local_coo = (nodes0[i], nodes1[j])
            global_coo = element.ct.mapping(*local_coo)
            lDofs_info[dof_local_numbering] = [
                local_coo, global_coo
            ]

    return lDofs_info
