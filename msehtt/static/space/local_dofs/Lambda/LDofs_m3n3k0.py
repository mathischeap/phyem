# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.quadrature import quadrature


def LDofs_Lambda__m3n3k0(tpm, degree):
    r"""Return a dict, for example `D` whose
            keys:
                local element indices
            values:
                dictionaries

            So, for each key (each element), we have a dictionary. And this dictionary's keys are
            all local dof indices and values are the coordinate info of the local dof.
            For example,
            D[100] = {   # the local dof information of local element indexed 100.
                0: [local_coo_info, global_coo_info],
                1: ...,
                ...
            }
            So in the local element indexed 100, we have some local dofs locally labeled 0, 1, ...; and
            for the local dof #0, its coo_info in the reference element is `local_coo_info`, and its
            coo info in the physics domain is `global_coo_info`.

            If this dof is for a 0-form in m2n2, then it is like
            `local_coo_info = (-1, -1)` and `global_coo_info=(0, 0)`.
            It means this local dof is the top-left corner of the reference domain, and in the physical domain,
            it is at place (0, 0).

            Basically,

            we return `local_coo_info=[float, float, float]` and `global_coo_info=[float, float, float]`,
            as we are looking at a nodal-dof in 3d space.

    """
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


from phyem.msehtt.static.space.local_numbering.Lambda.ln_m3n3k0 import _ln_m3n3k0_msepy_quadrilateral_


def ___mm330_lDofs_hexahedron___(element, degree):
    """"""
    p, dtype = element.degree_parser(degree)

    LN = _ln_m3n3k0_msepy_quadrilateral_(p)

    nodes = quadrature(p, category=dtype).quad_nodes

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
