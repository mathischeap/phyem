# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from tools.quadrature import quadrature
from msehtt.static.form.addons.static import MseHttFormStaticCopy

from src.config import COMM


class PartialMesh_Elements_Compute(Frozen):
    """Compute something (other than cfl number) on this partial mesh of elements."""

    def __init__(self, elements):
        """"""
        self._elements = elements
        self._freeze()

    def a(self, A, B, C):
        """Compute trilinear a(A, B, C) : = <A x B, C>_{Omega} where Omega is the domain the elements cover.

        Thus, A, B, C must be three forms.

        Parameters
        ----------
        A
        B
        C

        Returns
        -------

        """
        assert (A.__class__ is MseHttFormStaticCopy and
                B.__class__ is MseHttFormStaticCopy and
                C.__class__ is MseHttFormStaticCopy), f"A, B, C must be msehtt static form static copy."

        if isinstance(A.degree, (float, int)):
            quad_degree_A = A.degree
        else:
            raise NotImplementedError(f"cannot find a quad degree from form-A degree = {A.degree}")
        if isinstance(B.degree, (float, int)):
            quad_degree_B = B.degree
        else:
            raise NotImplementedError(f"cannot find a quad degree from form-B degree = {B.degree}")
        if isinstance(C.degree, (float, int)):
            quad_degree_C = C.degree
        else:
            raise NotImplementedError(f"cannot find a quad degree from form-C degree = {C.degree}")

        quad_degree = int(max([quad_degree_A, quad_degree_B, quad_degree_C]) * 1.5) + 1

        if self._elements.mn == (2, 2):
            indicator = 'm2n2'
            quad = quadrature((quad_degree, quad_degree), category='Gauss')
        elif self._elements.mn == (3, 3):
            indicator = 'm3n3'
            quad = quadrature((quad_degree, quad_degree, quad_degree), category='Gauss')
        else:
            raise NotImplementedError()

        quad_nodes = quad.quad_nodes
        qw_ravel = quad.quad_weights_ravel
        metric_coo = [_.ravel('F') for _ in np.meshgrid(*quad_nodes, indexing='ij')]

        rA = A.reconstruct(*quad_nodes, ravel=True)[1]   # no need coo data
        rB = B.reconstruct(*quad_nodes, ravel=True)[1]   # no need coo data
        rC = C.reconstruct(*quad_nodes, ravel=True)[1]   # no need coo data

        indicator += '=' + str((len(rA), len(rB), len(rC)))

        rank_integral_values = list()
        for e in self._elements:
            element = self._elements[e]
            detJ = element.ct.Jacobian(*metric_coo)
            if indicator == 'm3n3=(3, 3, 3)':
                # A, B, C are all vectors.
                # A = [wx wy, wz]^T    B = [u v w]^T   C= [a b c]^T
                # A x B = [wy*w - wz*v   wz*u - wx*w   wx*v - wy*u]^T = [A0 B0 C0]^T
                # (A x B) dot C = A0*a + B0*b + C0*c
                wx, wy, wz = rA[0][e], rA[1][e], rA[2][e]
                u, v, w = rB[0][e], rB[1][e], rB[2][e]
                a, b, c = rC[0][e], rC[1][e], rC[2][e]
                A0a = (wy * w - wz * v) * a
                B0b = (wz * u - wx * w) * b
                C0c = (wx * v - wy * u) * c
                ABC = A0a + B0b + C0c
                integral_value_in_element_e = np.sum(ABC * qw_ravel * detJ)
            else:
                raise NotImplementedError(f"<AxB, C>-integral not implemented for indicator={indicator}")

            rank_integral_values.append(integral_value_in_element_e)

        rank_integral = sum(rank_integral_values)
        total_integral = COMM.allgather(rank_integral)
        return sum(total_integral)
