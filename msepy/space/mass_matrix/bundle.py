# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from scipy.sparse import csr_matrix, bmat

from phyem.tools.frozen import Frozen
from phyem.tools.quadrature import Quadrature
from phyem.src.config import _setting


class MsePyMassMatrixBundle(Frozen):
    """"""

    def __init__(self, space):
        """Store required info."""
        self._space = space
        self._k = space.abstract.k
        self._n = space.abstract.n  # manifold dimensions
        self._m = space.abstract.m  # dimensions of the embedding space.
        self._orientation = space.abstract.orientation
        self._cache = dict()
        self._freeze()

    def __call__(self, degree, quad=None):
        """Making the local numbering for degree."""
        key = f"{degree}" + str(quad)

        if quad is None:
            _P_ = self._space[degree].p

            quad_degrees = list()
            quad_types = list()
            for _pi_ in _P_:

                is_linear = self._space.mesh.elements._is_linear()
                if is_linear:  # ALL elements are linear.
                    high_accuracy = _setting['high_accuracy']
                    if high_accuracy:
                        quad_degree = [p + 1 for p in _pi_]
                        quad_type = 'Gauss'
                        # +1 for conservation
                    else:
                        quad_degree = [p for p in _pi_]
                        # + 0 for much sparser matrices.
                        quad_type = self._space[degree].ntype
                else:
                    quad_degree = [p + 2 for p in _pi_]
                    quad_type = 'Gauss'

                quad_degrees.append(
                    quad_degree
                )
                quad_types.append(
                    quad_type
                )

            quad_degrees = np.array(quad_degrees)
            quad_degree = np.max(quad_degrees, axis=0)

            if all([_ == quad_types[0] for _ in quad_types]):
                quad_type = quad_types[0]
            else:
                quad_type = 'Gauss'

            quad = (quad_degree, quad_type)

        else:
            raise NotImplementedError()

        m = self._m
        n = self._n
        k = self._k

        if key in self._cache:
            M = self._cache[key]
        else:
            if m == 2 and n == 2 and k == 1:  # for k == 0 and k == 1.
                method_name = f"_m{m}_n{n}_k{k}_{self._orientation}"
            else:
                method_name = f"_m{m}_n{n}_k{k}"
            M = getattr(self, method_name)(degree, quad)
            M = self._space.mesh.elements._index_mapping.distribute_according_to_reference_elements_dict(M)
            self._cache[key] = M

        return M

    def _m3_n3_k3(self, degree, quad):
        """"""
        quad_degree, quad_type = quad
        quad = Quadrature(quad_degree, category=quad_type)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et_sg, BF = self._space.basis_functions[degree](*quad_nodes)
        detJM = self._space.mesh.ct.Jacobian(*xi_et_sg)

        bf0, bf1, bf2 = BF
        M = dict()
        for re in detJM:  # go through all reference elements
            _1_over_det_jm = np.reciprocal(detJM[re])
            metric = _1_over_det_jm * quad_weights
            M_re = np.einsum(
                'im, jm, m -> ij',
                bf0, bf0, metric,
                optimize='optimal',
            )
            M0 = csr_matrix(M_re)

            M_re = np.einsum(
                'im, jm, m -> ij',
                bf1, bf1, metric,
                optimize='optimal',
            )
            M1 = csr_matrix(M_re)

            M_re = np.einsum(
                'im, jm, m -> ij',
                bf2, bf2, metric,
                optimize='optimal',
            )
            M2 = csr_matrix(M_re)

            M[re] = bmat(
                [
                    (M0, None, None),
                    (None, M1, None),
                    (None, None, M2)
                ], format='csr'
            )

        return M

    @staticmethod
    def _einsum_helper(metric, bfO, bfS):
        """"""
        M = np.einsum('m, im, jm -> ij', metric, bfO, bfS, optimize='optimal')
        return csr_matrix(M)

    def _m3_n3_k2(self, degree, quad):
        """"""
        quad_degree, quad_type = quad
        quad = Quadrature(quad_degree, category=quad_type)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et_sg, BF = self._space.basis_functions[degree](*quad_nodes)
        JM = self._space.mesh.ct.Jacobian_matrix(*xi_et_sg)
        detJM = self._space.mesh.ct.Jacobian(*xi_et_sg, JM=JM)
        iJM = self._space.mesh.ct.inverse_Jacobian_matrix(*xi_et_sg, JM=JM)
        G = self._space.mesh.ct.inverse_metric_matrix(*xi_et_sg, iJM=iJM)
        del JM, iJM

        M = dict()
        for re in detJM:
            det_jm = detJM[re]
            g = G[re]
            reference_mtype = detJM.get_mtype_of_reference_element(re)

            if reference_mtype[:6] == 'Linear':
                g00 = quad_weights * det_jm * g[1][1]*g[2][2]
                g11 = quad_weights * det_jm * g[2][2]*g[0][0]
                g22 = quad_weights * det_jm * g[0][0]*g[1][1]
                g12_20_g10_22 = None
                g10_21_g11_20 = None
                g20_01_g21_00 = None
            else:
                g00 = quad_weights * det_jm * (g[1][1]*g[2][2]-g[1][2]*g[2][1])
                g11 = quad_weights * det_jm * (g[2][2]*g[0][0]-g[2][0]*g[0][2])
                g22 = quad_weights * det_jm * (g[0][0]*g[1][1]-g[0][1]*g[1][0])
                g12_20_g10_22 = quad_weights * det_jm * (g[1][2] * g[2][0] - g[1][0] * g[2][2])
                g10_21_g11_20 = quad_weights * det_jm * (g[1][0] * g[2][1] - g[1][1] * g[2][0])
                g20_01_g21_00 = quad_weights * det_jm * (g[2][0] * g[0][1] - g[2][1] * g[0][0])

            Mre = list()

            for i in range(3):

                bf = BF[i]

                if reference_mtype[:6] == 'Linear':
                    M00 = self._einsum_helper(g00, bf[0], bf[0])
                    M11 = self._einsum_helper(g11, bf[1], bf[1])
                    M22 = self._einsum_helper(g22, bf[2], bf[2])
                    M01 = None
                    M02 = None
                    M10 = None
                    M12 = None
                    M20 = None
                    M21 = None
                else:
                    M00 = self._einsum_helper(g00, bf[0], bf[0])
                    M11 = self._einsum_helper(g11, bf[1], bf[1])
                    M22 = self._einsum_helper(g22, bf[2], bf[2])
                    M01 = self._einsum_helper(g12_20_g10_22, bf[0], bf[1])
                    M02 = self._einsum_helper(g10_21_g11_20, bf[0], bf[2])
                    M12 = self._einsum_helper(g20_01_g21_00, bf[1], bf[2])
                    M10 = M01.T
                    M20 = M02.T
                    M21 = M12.T

                Mre.append(bmat(
                    [
                        (M00, M01, M02),
                        (M10, M11, M12),
                        (M20, M21, M22)
                    ], format='csr'
                ))

            M[re] = bmat(
                [
                    (Mre[0], None, None),
                    (None, Mre[1], None),
                    (None, None, Mre[2])
                ], format='csr'
            )

        return M

    def _m3_n3_k1(self, degree, quad):
        """"""
        quad_degree, quad_type = quad
        quad = Quadrature(quad_degree, category=quad_type)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et_sg, BF = self._space.basis_functions[degree](*quad_nodes)
        JM = self._space.mesh.ct.Jacobian_matrix(*xi_et_sg)
        detJM = self._space.mesh.ct.Jacobian(*xi_et_sg, JM=JM)
        iJM = self._space.mesh.ct.inverse_Jacobian_matrix(*xi_et_sg, JM=JM)
        G = self._space.mesh.ct.inverse_metric_matrix(*xi_et_sg, iJM=iJM)
        del JM, iJM

        M = dict()
        for re in detJM:
            det_jm = detJM[re]
            g = G[re]
            reference_mtype = detJM.get_mtype_of_reference_element(re)

            g00 = quad_weights * det_jm * g[0][0]
            g11 = quad_weights * det_jm * g[1][1]
            g22 = quad_weights * det_jm * g[2][2]
            if reference_mtype[:6] == 'Linear':
                g01 = None
                g02 = None
                g12 = None
            else:
                g01 = quad_weights * det_jm * g[0][1]
                g02 = quad_weights * det_jm * g[0][2]
                g12 = quad_weights * det_jm * g[1][2]

            Mre = list()
            for i in range(3):
                bf = BF[i]

                M00 = self._einsum_helper(g00, bf[0], bf[0])
                M11 = self._einsum_helper(g11, bf[1], bf[1])
                M22 = self._einsum_helper(g22, bf[2], bf[2])

                if reference_mtype[:6] == 'Linear':
                    M01 = None
                    M02 = None
                    M10 = None
                    M12 = None
                    M20 = None
                    M21 = None
                else:
                    M01 = self._einsum_helper(g01, bf[0], bf[1])
                    M02 = self._einsum_helper(g02, bf[0], bf[2])
                    M12 = self._einsum_helper(g12, bf[1], bf[2])
                    M10 = M01.T
                    M20 = M02.T
                    M21 = M12.T

                Mre.append(bmat(
                    [
                        (M00, M01, M02),
                        (M10, M11, M12),
                        (M20, M21, M22)
                    ], format='csr'
                ))

            M[re] = bmat(
                [
                    (Mre[0], None, None),
                    (None, Mre[1], None),
                    (None, None, Mre[2])
                ], format='csr'
            )

        return M

    def _m3_n3_k0(self, degree, quad):
        """mass matrix of 0-form on 3-manifold in 3d space"""
        quad_degree, quad_type = quad
        quad = Quadrature(quad_degree, category=quad_type)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et_sg, BF = self._space.basis_functions[degree](*quad_nodes)
        detJM = self._space.mesh.ct.Jacobian(*xi_et_sg)
        bf0, bf1, bf2 = BF
        M = dict()
        for re in detJM:  # go through all reference elements
            det_jm = detJM[re]
            metric = det_jm * quad_weights
            M_re = np.einsum(
                'im, jm, m -> ij',
                bf0, bf0, metric,
                optimize='optimal',
            )
            M0 = csr_matrix(M_re)

            M_re = np.einsum(
                'im, jm, m -> ij',
                bf1, bf1, metric,
                optimize='optimal',
            )
            M1 = csr_matrix(M_re)

            M_re = np.einsum(
                'im, jm, m -> ij',
                bf2, bf2, metric,
                optimize='optimal',
            )
            M2 = csr_matrix(M_re)

            M[re] = bmat(
                [
                    (M0, None, None),
                    (None, M1, None),
                    (None, None, M2)
                ], format='csr'
            )
        return M

    def _m2_n2_k0(self, degree, quad):
        """"""
        quad_degree, quad_type = quad
        quad = Quadrature(quad_degree, category=quad_type)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et, BF = self._space.basis_functions[degree](*quad_nodes)
        detJM = self._space.mesh.ct.Jacobian(*xi_et)
        bf0, bf1 = BF
        M = dict()
        for re in detJM:  # go through all reference elements
            det_jm = detJM[re]
            metric = det_jm * quad_weights
            M_re = np.einsum(
                'im, jm, m -> ij',
                bf0, bf0, metric,
                optimize='optimal',
                        )
            M0 = csr_matrix(M_re)
            M_re = np.einsum(
                'im, jm, m -> ij',
                bf1, bf1, metric,
                optimize='optimal',
            )
            M1 = csr_matrix(M_re)

            M[re] = bmat(
                [
                    (M0, None),
                    (None, M1)
                ], format='csr'
            )
        return M

    def _m2_n2_k2(self, degree, quad):
        """"""
        quad_degree, quad_type = quad
        quad = Quadrature(quad_degree, category=quad_type)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et, BF = self._space.basis_functions[degree](*quad_nodes)
        detJM = self._space.mesh.ct.Jacobian(*xi_et)
        bf0, bf1 = BF
        M = dict()
        for re in detJM:  # go through all reference elements
            det_jm = np.reciprocal(detJM[re])
            metric = det_jm * quad_weights
            M_re = np.einsum(
                'im, jm, m -> ij',
                bf0, bf0, metric,
                optimize='optimal',
                        )
            M0 = csr_matrix(M_re)
            M_re = np.einsum(
                'im, jm, m -> ij',
                bf1, bf1, metric,
                optimize='optimal',
            )
            M1 = csr_matrix(M_re)

            M[re] = bmat(
                [
                    (M0, None),
                    (None, M1)
                ], format='csr'
            )

        return M

    def _m2_n2_k1_inner(self, degree, quad):
        """mass matrix of inner 1-form on 2-manifold in 2d space"""
        quad_degree, quad_type = quad
        quad = Quadrature(quad_degree, category=quad_type)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et, BF = self._space.basis_functions[degree](*quad_nodes)
        JM = self._space.mesh.ct.Jacobian_matrix(*xi_et)
        detJM = self._space.mesh.ct.Jacobian(*xi_et, JM=JM)
        iJM = self._space.mesh.ct.inverse_Jacobian_matrix(*xi_et, JM=JM)
        G = self._space.mesh.ct.inverse_metric_matrix(*xi_et, iJM=iJM)
        del JM, iJM
        M = dict()
        for re in detJM:
            det_jm = detJM[re]
            g = G[re]
            reference_mtype = detJM.get_mtype_of_reference_element(re)

            g00 = quad_weights * det_jm * g[0][0]
            g11 = quad_weights * det_jm * g[1][1]

            if reference_mtype[:6] == 'Linear':
                g01 = None
            else:
                g01 = quad_weights * det_jm * g[0][1]

            Mre = list()
            for i in range(2):
                bf = BF[i]

                M00 = self._einsum_helper(g00, bf[0], bf[0])
                M11 = self._einsum_helper(g11, bf[1], bf[1])

                if reference_mtype[:6] == 'Linear':
                    M01 = None
                    M10 = None
                else:
                    M01 = self._einsum_helper(g01, bf[0], bf[1])
                    M10 = M01.T

                Mre.append(bmat(
                    [
                        (M00, M01),
                        (M10, M11)
                    ], format='csr'
                ))

            M[re] = bmat(
                [
                    (Mre[0], None),
                    (None, Mre[1])
                ], format='csr'
            )

        return M

    def _m2_n2_k1_outer(self, degree, quad):
        """mass matrix of outer 1-form on 2-manifold in 2d space"""
        quad_degree, quad_type = quad
        quad = Quadrature(quad_degree, category=quad_type)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et, BF = self._space.basis_functions[degree](*quad_nodes)
        JM = self._space.mesh.ct.Jacobian_matrix(*xi_et)
        detJM = self._space.mesh.ct.Jacobian(*xi_et, JM=JM)
        iJM = self._space.mesh.ct.inverse_Jacobian_matrix(*xi_et, JM=JM)
        G = self._space.mesh.ct.inverse_metric_matrix(*xi_et, iJM=iJM)
        del JM, iJM
        M = dict()
        for re in detJM:
            det_jm = detJM[re]
            g = G[re]
            reference_mtype = detJM.get_mtype_of_reference_element(re)

            g00 = quad_weights * det_jm * g[1][1]
            g11 = quad_weights * det_jm * g[0][0]
            if reference_mtype[:6] == 'Linear':
                g01 = None
            else:
                g01 = quad_weights * det_jm * g[1][0]

            Mre = list()
            for i in range(2):
                bf = BF[i]

                M00 = self._einsum_helper(g00, bf[0], bf[0])
                M11 = self._einsum_helper(g11, bf[1], bf[1])

                if reference_mtype[:6] == 'Linear':
                    M01 = None
                    M10 = None
                else:
                    M01 = - self._einsum_helper(g01, bf[0], bf[1])
                    M10 = M01.T

                Mre.append(bmat(
                    [
                        (M00, M01),
                        (M10, M11)
                    ], format='csr'
                ))

            M[re] = bmat(
                [
                    (Mre[0], None),
                    (None, Mre[1])
                ], format='csr'
            )

        return M

    def _m1_n1_k0(self, degree, quad):
        """"""
        quad_degree, quad_type = quad
        quad = Quadrature(quad_degree, category=quad_type)
        quad_nodes, quad_weights = quad.quad
        xi_et, bf = self._space.basis_functions[degree](quad_nodes)
        detJM = self._space.mesh.ct.Jacobian(*xi_et)
        bf = bf[0]
        M = dict()
        for re in detJM:  # go through all reference elements
            det_jm = detJM[re]
            M_re = np.einsum(
                'im, jm, m -> ij',
                bf, bf, det_jm * quad_weights,
                optimize='optimal',
                        )
            M[re] = csr_matrix(M_re)
        return M

    def _m1_n1_k1(self, degree, quad):
        """"""
        quad_degree, quad_type = quad
        quad = Quadrature(quad_degree, category=quad_type)
        quad_nodes, quad_weights = quad.quad
        xi_et, bf = self._space.basis_functions[degree](quad_nodes)
        detJM = self._space.mesh.ct.Jacobian(*xi_et)
        bf = bf[0]
        M = dict()
        for re in detJM:  # go through all reference elements
            reciprocal_det_jm = np.reciprocal(detJM[re])
            M_re = np.einsum(
                'im, jm, m -> ij',
                bf, bf, reciprocal_det_jm * quad_weights,
                optimize='optimal',
                        )
            M[re] = csr_matrix(M_re)
        return M
