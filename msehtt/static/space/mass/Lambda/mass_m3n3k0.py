# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.src.config import COMM, MPI
from phyem.tools.quadrature import quadrature


def mass_Lambda__m3n3k0(tpm, degree, cochain):
    """"""
    elements = tpm.composition
    mass = list()
    for e in elements:
        element = elements[e]
        etype = element.etype
        if etype in (
                "orthogonal hexahedron",
                "unique msepy curvilinear hexahedron",
        ):
            element_mass = _mass330_msepy_quadrilateral_(
                element, cochain[e], degree)
            mass.append(element_mass)
        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")

    mass = sum(mass)
    return COMM.allreduce(mass, op=MPI.SUM)


from phyem.msehtt.static.space.reconstruct.Lambda.Rc_m3n3k0 import ___rc330_msepy_quadrilateral___


def _mass330_msepy_quadrilateral_(element, local_cochain, degree):
    """"""
    p = element.degree_parser(degree)[0]
    p = (p[0] + 2, p[1] + 2, p[2] + 2)
    nodes, weights = quadrature(p, 'Gauss').quad
    _, _, _, u = ___rc330_msepy_quadrilateral___(element, degree, local_cochain, *nodes, ravel=False)
    meshgrid = np.meshgrid(*nodes, indexing='ij')
    J = element.ct.Jacobian(*meshgrid)
    mass = np.einsum(
        'ij, i, j -> ',
        J * u, *weights,
        optimize='optimal'
    )
    return mass
