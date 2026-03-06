# -*- coding: utf-8 -*-
r"""Integrate something over this partial mesh of elements.

This is similar to method `compute`.
"""
import numpy as np

from phyem.tools.frozen import Frozen
from phyem.tools.quadrature import quadrature
from phyem.src.config import COMM, MPI
from phyem.msehtt.tools.local_scalar import MseHttLocalScalar
from phyem.msehtt.tools.vector.static.local import MseHttStaticLocalVector

from phyem.src.spaces.main import _degree_str_maker


class PartialMesh_Elements_Integrate(Frozen):
    """Integrate something on this partial mesh of elements."""

    def __init__(self, elements):
        self._elements = elements
        self._cache_0_ = {}
        self._cache_1_ = {}
        self._cache_3_ = {}
        self._freeze()

    def ABdC(self, A, B, C):
        r"""Integrate of {A * B * C} over these elements.

        CASE 1:
            A, B, C are all msehtt-static-forms.

        """
        if all([_.__class__.__name__ in ['MseHttFormStaticCopy', 'MseHtt_From_InterpolateCopy'] for _ in [A, B, C]]):
            # CASE 1
            return self._compute_msehtt_form_ABdC_(A, B, C)
        else:
            raise Exception(A.__class__.__name__, B.__class__.__name__, C.__class__.__name__)

    def _compute_msehtt_form_ABdC_(self, A, B, C):
        r"""Subroutine to compute msehtt-static form A, B, C.
        """
        # CASE 1
        At = A._f.space.dtype
        Bt = B._f.space.dtype
        Ct = C._f.space.dtype

        if (At, Bt, Ct) == ('2d-scalar', '2d-vector', '2d-scalar'):
            # Do integral of {A * (B · C)} over the partial mesh of elements.
            return self._compute_msehtt_form_sA_vB_dvC_(A, B, C, dimensions=2)
        else:
            raise NotImplementedError(f"{(At, Bt, Ct)}")

    def _compute_msehtt_form_sA_vB_dvC_(self, A, B, C, dimensions=2):
        r"""For msehtt forms, we compute integral of {A * (B · dC)} where A is scalar, B is a vector and
        C is a scalar. So dC becomes a vector (If C is inner, dC is grad of C, else dC is curl (perp grad) of C.)

        Parameters
        ----------
        A
        B
        C
        dimensions :
            If dimensions=2, we use m=n=2,
            If dimensions=3, we use m=n=3.

        """
        self._cache_1_ = {}

        from phyem.msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElement
        pA, _ = MseHttGreatMeshBaseElement.degree_parser(A.degree, m=dimensions, n=dimensions)
        pB, _ = MseHttGreatMeshBaseElement.degree_parser(B.degree, m=dimensions, n=dimensions)
        pC, _ = MseHttGreatMeshBaseElement.degree_parser(C.degree, m=dimensions, n=dimensions)

        max_pA = max(pA)
        max_pB = max(pB)
        max_pC = max(pC)
        quad_p = int(max([max_pA, max_pB, max_pC]) * 1.5) + 1
        if dimensions == 2:
            quad_p = (quad_p, quad_p)
        elif dimensions == 3:
            quad_p = (quad_p, quad_p, quad_p)
        else:
            raise Exception()

        quad = quadrature(quad_p, category='Gauss')

        quad_nodes = quad.quad_nodes
        qw_ravel = quad.quad_weights_ravel
        metric_coo = [_.ravel('F') for _ in np.meshgrid(*quad_nodes, indexing='ij')]

        rmA = A._f.reconstruction_matrix(*quad_nodes)
        rmB = B._f.reconstruction_matrix(*quad_nodes)

        E = C._f.incidence_matrix
        dC = C._f.d()
        rm_dc = dC.reconstruction_matrix(*quad_nodes)

        rmC = [{} for _ in range(len(rm_dc))]
        for i in E:
            ei = E[i]
            for j in range(len(rm_dc)):
                rmc = rm_dc[j][i]
                rmC[j][i] = rmc @ ei

        INTEGRAL = []
        for e in self._elements:
            element = self._elements[e]
            cache_key = element.metric_bf_cache_key()
            # print(cache_key)

            # ------------------------------------------------------------------------------------------
            if isinstance(cache_key, str) and cache_key in self._cache_1_:
                ARRAY = self._cache_1_[cache_key]
            else:
                detJ = element.ct.Jacobian(*metric_coo)

                W = rmA[0][e]
                if dimensions == 2:
                    # B = [u, v]
                    # C = [a, b]
                    u, v = rmB[0][e], rmB[1][e]
                    a, b = rmC[0][e], rmC[1][e]
                    w, c = None, None
                elif dimensions == 3:
                    # B = [u, v, w]
                    # C = [a, b, c]
                    u, v, w = rmB[0][e], rmB[1][e], rmB[2][e]
                    a, b, c = rmC[0][e], rmC[1][e], rmC[2][e]
                else:
                    raise Exception()

                if dimensions == 2:
                    # A = [W]
                    # B = [u, v]
                    # C = [a, b]
                    # int{W(ua + vb)} = int{Wua + Wvb}
                    # print(W.shape, u.shape, a.shape, qw_ravel.shape, detJ.shape)
                    ARRAY = np.einsum(
                        'li, lj, lk, l -> ijk', W, u, a, qw_ravel * detJ, optimize='optimal'
                    ) + np.einsum(
                        'li, lj, lk, l -> ijk', W, v, b, qw_ravel * detJ, optimize='optimal'
                    )
                elif dimensions == 3:
                    # A = [W]
                    # B = [u, v, w]
                    # C = [a, b, c]
                    # int{W(ua + vb + wc)} = int{Wua + Wvb + Wwc}
                    ARRAY = np.einsum(
                        'li, lj, lk, l -> ijk', W, u, a, qw_ravel * detJ, optimize='optimal'
                    ) + np.einsum(
                        'li, lj, lk, l -> ijk', W, v, b, qw_ravel * detJ, optimize='optimal'
                    ) + np.einsum(
                        'li, lj, lk, l -> ijk', W, w, c, qw_ravel * detJ, optimize='optimal'
                    )

                else:
                    raise Exception()

                if isinstance(cache_key, str):
                    self._cache_1_[cache_key] = ARRAY
                else:
                    pass

            # --------------------------------------------------------------------------------------------------
            c_A = A.cochain[e]
            c_B = B.cochain[e]
            c_C = C.cochain[e]
            integral = np.einsum('ijk, i, j, k ->', ARRAY, c_A, c_B, c_C, optimize='optimal')
            INTEGRAL.append(integral)

        INTEGRAL = sum(INTEGRAL)
        return COMM.allreduce(INTEGRAL, op=MPI.SUM)

    def ABC(self, A, B, C):
        r"""Integrate of {A * B * C} over these elements.

        CASE 1:
            A, B, C are all msehtt-static-forms.

        """
        if all([_.__class__.__name__ in ['MseHttFormStaticCopy', 'MseHtt_From_InterpolateCopy'] for _ in [A, B, C]]):
            # CASE 1
            return self._compute_msehtt_form_ABC_(A, B, C)
        else:
            raise Exception(A.__class__.__name__, B.__class__.__name__, C.__class__.__name__)

    def _compute_msehtt_form_ABC_(self, A, B, C):
        r"""Subroutine to compute msehtt-static form A, B, C.
        """
        # CASE 1
        At = A._f.space.dtype
        Bt = B._f.space.dtype
        Ct = C._f.space.dtype

        if (At, Bt, Ct) == ('2d-scalar', '2d-vector', '2d-vector'):
            # Do integral of {A * (B · C)} over the partial mesh of elements.
            return self._compute_msehtt_form_sA_vB_vC_(A, B, C, dimensions=2)
        elif (At, Bt, Ct) == ('2d-scalar', '2d-scalar', '2d-scalar'):
            # Do integral of {A * B * C} over the partial mesh of elements.
            return self._compute_msehtt_form_sA_sB_sC_(A, B, C, dimensions=2)
        elif (At, Bt, Ct) == ('2d-vector', '2d-scalar', '2d-vector'):
            # Do integral of {B * (A · C)} over the partial mesh of elements.
            return self._compute_msehtt_form_sA_vB_vC_(B, A, C, dimensions=2)
        elif (At, Bt, Ct) == ('2d-vector', '2d-vector', '2d-scalar'):
            # Do integral of {C * (A · B)} over the partial mesh of elements.
            return self._compute_msehtt_form_sA_vB_vC_(C, A, B, dimensions=2)
        elif (At, Bt, Ct) == ('3d-scalar', '3d-vector', '3d-vector'):
            # Do integral of {A * (B · C)} over the partial mesh of elements.
            return self._compute_msehtt_form_sA_vB_vC_(A, B, C, dimensions=3)
        elif (At, Bt, Ct) == ('3d-vector', '3d-scalar', '3d-vector'):
            # Do integral of {B * (A · C)} over the partial mesh of elements.
            return self._compute_msehtt_form_sA_vB_vC_(B, A, C, dimensions=3)
        elif (At, Bt, Ct) == ('3d-vector', '3d-vector', '3d-scalar'):
            # Do integral of {C * (A · B)} over the partial mesh of elements.
            return self._compute_msehtt_form_sA_vB_vC_(C, A, B, dimensions=3)
        else:
            raise NotImplementedError(f"{(At, Bt, Ct)}")

    def _compute_msehtt_form_sA_sB_sC_(self, A, B, C, dimensions=2):
        r"""For msehtt forms, we compute integral of {A * B * C}

        Parameters
        ----------
        A
        B
        C
        dimensions :
            If dimensions=2, we use m=n=2,
            If dimensions=3, we use m=n=3.

        """
        self._cache_3_ = {}

        from phyem.msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElement
        pA, _ = MseHttGreatMeshBaseElement.degree_parser(A.degree, m=dimensions, n=dimensions)
        pB, _ = MseHttGreatMeshBaseElement.degree_parser(B.degree, m=dimensions, n=dimensions)
        pC, _ = MseHttGreatMeshBaseElement.degree_parser(C.degree, m=dimensions, n=dimensions)

        max_pA = max(pA)
        max_pB = max(pB)
        max_pC = max(pC)
        quad_p = int(max([max_pA, max_pB, max_pC]) * 1.5) + 1
        if dimensions == 2:
            quad_p = (quad_p, quad_p)
        elif dimensions == 3:
            quad_p = (quad_p, quad_p, quad_p)
        else:
            raise Exception()

        quad = quadrature(quad_p, category='Gauss')

        quad_nodes = quad.quad_nodes
        qw_ravel = quad.quad_weights_ravel
        metric_coo = [_.ravel('F') for _ in np.meshgrid(*quad_nodes, indexing='ij')]

        rmA = A._f.reconstruction_matrix(*quad_nodes)
        rmB = B._f.reconstruction_matrix(*quad_nodes)
        rmC = C._f.reconstruction_matrix(*quad_nodes)

        INTEGRAL = []
        for e in self._elements:
            element = self._elements[e]
            cache_key = element.metric_bf_cache_key()

            # ------------------------------------------------------------------------------------------
            if isinstance(cache_key, str) and cache_key in self._cache_3_:
                ARRAY = self._cache_3_[cache_key]
            else:
                detJ = element.ct.Jacobian(*metric_coo)

                a = rmA[0][e]
                b = rmB[0][e]
                c = rmC[0][e]
                ARRAY = np.einsum(
                    'li, lj, lk, l -> ijk', a, b, c, qw_ravel * detJ, optimize='optimal'
                )

                if isinstance(cache_key, str):
                    self._cache_3_[cache_key] = ARRAY
                else:
                    pass

            # --------------------------------------------------------------------------------------------------
            c_A = A.cochain[e]
            c_B = B.cochain[e]
            c_C = C.cochain[e]
            integral = np.einsum('ijk, i, j, k ->', ARRAY, c_A, c_B, c_C, optimize='optimal')
            INTEGRAL.append(integral)

        INTEGRAL = sum(INTEGRAL)
        return COMM.allreduce(INTEGRAL, op=MPI.SUM)

    def _compute_msehtt_form_sA_vB_vC_(self, A, B, C, dimensions=2):
        r"""For msehtt forms, we compute integral of {A * (B · C)}

        Parameters
        ----------
        A
        B
        C
        dimensions :
            If dimensions=2, we use m=n=2,
            If dimensions=3, we use m=n=3.

        """
        self._cache_0_ = {}

        from phyem.msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElement
        pA, _ = MseHttGreatMeshBaseElement.degree_parser(A.degree, m=dimensions, n=dimensions)
        pB, _ = MseHttGreatMeshBaseElement.degree_parser(B.degree, m=dimensions, n=dimensions)
        pC, _ = MseHttGreatMeshBaseElement.degree_parser(C.degree, m=dimensions, n=dimensions)

        max_pA = max(pA)
        max_pB = max(pB)
        max_pC = max(pC)
        quad_p = int(max([max_pA, max_pB, max_pC]) * 1.5) + 1
        if dimensions == 2:
            quad_p = (quad_p, quad_p)
        elif dimensions == 3:
            quad_p = (quad_p, quad_p, quad_p)
        else:
            raise Exception()

        quad = quadrature(quad_p, category='Gauss')

        quad_nodes = quad.quad_nodes
        qw_ravel = quad.quad_weights_ravel
        metric_coo = [_.ravel('F') for _ in np.meshgrid(*quad_nodes, indexing='ij')]

        rmA = A._f.reconstruction_matrix(*quad_nodes)
        rmB = B._f.reconstruction_matrix(*quad_nodes)
        rmC = C._f.reconstruction_matrix(*quad_nodes)

        INTEGRAL = []
        for e in self._elements:
            element = self._elements[e]
            cache_key = element.metric_bf_cache_key()

            # ------------------------------------------------------------------------------------------
            if isinstance(cache_key, str) and cache_key in self._cache_0_:
                ARRAY = self._cache_0_[cache_key]
            else:
                detJ = element.ct.Jacobian(*metric_coo)

                W = rmA[0][e]
                if dimensions == 2:
                    # B = [u, v]
                    # C = [a, b]
                    u, v = rmB[0][e], rmB[1][e]
                    a, b = rmC[0][e], rmC[1][e]
                    w, c = None, None
                elif dimensions == 3:
                    # B = [u, v, w]
                    # C = [a, b, c]
                    u, v, w = rmB[0][e], rmB[1][e], rmB[2][e]
                    a, b, c = rmC[0][e], rmC[1][e], rmC[2][e]
                else:
                    raise Exception()

                if dimensions == 2:
                    # A = [W]
                    # B = [u, v]
                    # C = [a, b]
                    # int{W(ua + vb)} = int{Wua + Wvb}
                    # print(W.shape, u.shape, a.shape, qw_ravel.shape, detJ.shape)
                    ARRAY = np.einsum(
                        'li, lj, lk, l -> ijk', W, u, a, qw_ravel * detJ, optimize='optimal'
                    ) + np.einsum(
                        'li, lj, lk, l -> ijk', W, v, b, qw_ravel * detJ, optimize='optimal'
                    )
                elif dimensions == 3:
                    # A = [W]
                    # B = [u, v, w]
                    # C = [a, b, c]
                    # int{W(ua + vb + wc)} = int{Wua + Wvb + Wwc}
                    ARRAY = np.einsum(
                        'li, lj, lk, l -> ijk', W, u, a, qw_ravel * detJ, optimize='optimal'
                    ) + np.einsum(
                        'li, lj, lk, l -> ijk', W, v, b, qw_ravel * detJ, optimize='optimal'
                    ) + np.einsum(
                        'li, lj, lk, l -> ijk', W, w, c, qw_ravel * detJ, optimize='optimal'
                    )

                else:
                    raise Exception()

                if isinstance(cache_key, str):
                    self._cache_0_[cache_key] = ARRAY
                else:
                    pass

            # --------------------------------------------------------------------------------------------------
            c_A = A.cochain[e]
            c_B = B.cochain[e]
            c_C = C.cochain[e]
            integral = np.einsum('ijk, i, j, k ->', ARRAY, c_A, c_B, c_C, optimize='optimal')
            INTEGRAL.append(integral)

        INTEGRAL = sum(INTEGRAL)
        return COMM.allreduce(INTEGRAL, op=MPI.SUM)

    def AB(self, A, B, element_wise=False, about=None):
        r"""Integrate of {A * B} over these elements.
        """
        if about is None:
            pass
        else:
            if about == 1:
                return self.___A_ast_vector___(A, B)
            elif about == 0:
                return self.___A_ast_vector___(B, A)
            else:
                raise Exception()

        if element_wise:
            return self.___element_wise_AB___(A, B)
        else:
            pass

        # element_wise is False.
        space_A = A._f.space
        space_B = B._f.space

        if space_A is space_B:  # A and B are from the same space, we use the mass matrix to do it

            the_space = space_A

            if _degree_str_maker(A._f.degree) == _degree_str_maker(B._f.degree):

                M = the_space.mass_matrix(A._f.degree)[0]
                INTEGRAL = 0
                elements = A._f.tpm.composition
                for e in elements:
                    m = M[e]
                    INTEGRAL += np.einsum(
                        'i, ij, j ->',
                        A.cochain[e], m.toarray(), B.cochain[e], optimize='optimal'
                    )
                return COMM.allreduce(INTEGRAL, op=MPI.SUM)

            else:
                raise NotImplementedError(f"{A._f.degree} {B._f.degree}.")

        else:
            raise NotImplementedError(f"{space_A}, {space_B}.")


    def ___element_wise_AB___(self, A, B):
        r"""
        Return a dictionary whose keys are local element indices and values are the
        integration in these local elements.
        """
        space_A = A._f.space
        space_B = B._f.space

        if space_A is space_B:  # A and B are from the same space, we use the mass matrix to do it

            the_space = space_A

            if _degree_str_maker(A._f.degree) == _degree_str_maker(B._f.degree):

                M = the_space.mass_matrix(A._f.degree)[0]
                INTEGRAL = dict()
                elements = A._f.tpm.composition
                for e in elements:
                    m = M[e]
                    INTEGRAL[e] = np.einsum(
                        'i, ij, j ->',
                        A.cochain[e], m.toarray(), B.cochain[e], optimize='optimal'
                    )
                return MseHttLocalScalar(INTEGRAL)

            else:
                raise NotImplementedError(f"{A._f.degree} {B._f.degree}.")

        else:
            raise NotImplementedError(f"{space_A}, {space_B}.")


    def ___A_ast_vector___(self, A, B):
        r"""REturn a vector for: A is given, and we return a ditionary of vector D. And

        D[e] @ B.cochain[e] gives the integration of AB in the element #e.

        """
        space_A = A._f.space
        space_B = B._f.space

        if space_A is space_B:  # A and B are from the same space, we use the mass matrix to do it

            the_space = space_A

            if _degree_str_maker(A._f.degree) == _degree_str_maker(B._f.degree):

                M = the_space.mass_matrix(A._f.degree)[0]
                vector = dict()
                elements = A._f.tpm.composition
                for e in elements:
                    m = M[e]
                    vector[e] = np.einsum(
                        'i, ij -> j',
                        A.cochain[e], m.toarray(), optimize='optimal'
                    )

                return MseHttStaticLocalVector(vector, B._f.cochain.gathering_matrix)

            else:
                raise NotImplementedError(f"{A._f.degree} {B._f.degree}.")

        else:
            raise NotImplementedError(f"{space_A}, {space_B}.")
