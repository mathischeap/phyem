# -*- coding: utf-8 -*-
r"""
"""
from tools.quadrature import quadrature


def LDofs_Lambda__m3n3k0(tpm, degree):
    r""""""
    LDofs = {}
    for e in tpm.composition:
        element = tpm.composition[e]
        etype = element.etype
        if etype in (
                'orthogonal hexahedron',
                'unique msepy curvilinear hexahedron',
        ):
            LDofs[e] = ___mm330_lDofs_hexahedron___(element, degree)
        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    return LDofs


from msehtt.static.space.local_numbering.Lambda.ln_m3n3k0 import _ln_m3n3k0_msepy_quadrilateral_


def ___mm330_lDofs_hexahedron___(element, degree):
    """"""
    p, dtype = element.degree_parser(degree)

    LN = _ln_m3n3k0_msepy_quadrilateral_(p)

    if dtype in ('Lobatto', ):
        nodes = quadrature(p, category=dtype).quad_nodes
    else:
        raise NotImplementedError(dtype)

    nodes0, nodes1, nodes2 = nodes

    px, py, pz = p

    lDofs_info = {}

    for k in range(pz + 1):
        for j in range(py + 1):
            for i in range(px + 1):
                dof_local_numbering = LN[i, j, k]
                local_coo = (nodes0[i], nodes1[j], nodes[k])
                global_coo = element.ct.mapping(*local_coo)
                lDofs_info[dof_local_numbering] = [
                    local_coo, global_coo
                ]

    return lDofs_info
