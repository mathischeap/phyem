# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from scipy.sparse import csr_matrix, bmat, lil_matrix

from phyem.tools.frozen import Frozen
from phyem.tools.quadrature import Quadrature
from phyem.src.config import _setting


class MsePyMassMatrixLambda(Frozen):
    """"""

    def __init__(self, space):
        """Store required info."""
        self._space = space
        self._mesh = space.mesh
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
            is_linear = self._space.mesh.elements._is_linear()
            if is_linear:  # ALL elements are linear.
                high_accuracy = _setting['high_accuracy']
                if high_accuracy:
                    quad_degree = [p + 1 for p in self._space[degree].p]
                    # +1 for conservation
                    quad_type = 'Gauss'
                else:
                    quad_degree = [p for p in self._space[degree].p]
                    # + 0 for much sparser matrices.
                    quad_type = self._space[degree].ntype
                quad = (quad_degree, quad_type)
            else:
                quad_degree = [p + 2 for p in self._space[degree].p]
                quad = (quad_degree, 'Gauss')

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
            elif m == 2 and n == 1 and k == 1:
                method_name = f"_m{m}_n{n}_k{k}_{self._orientation}"
            else:
                method_name = f"_m{m}_n{n}_k{k}"
            M = getattr(self, method_name)(degree, quad)

            if isinstance(M, tuple):
                indicator, M = M
                if indicator == 'unique dict':
                    pass
                else:
                    raise NotImplementedError()
            else:
                M = self._space.mesh.elements._index_mapping.distribute_according_to_reference_elements_dict(M)
            self._cache[key] = M

        return M

    def _m3_n3_k3(self, degree, quad):
        """mass matrix of 3-form on 3-manifold in 3d space"""
        quad_degree, quad_type = quad
        quad = Quadrature(quad_degree, category=quad_type)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et_sg, bf = self._space.basis_functions[degree](*quad_nodes)
        detJM = self._space.mesh.ct.Jacobian(*xi_et_sg)
        bf = bf[0]
        M = dict()
        for re in detJM:  # go through all reference elements
            _1_over_det_jm = np.reciprocal(detJM[re])
            M_re = np.einsum(
                'im, jm, m -> ij',
                bf, bf, _1_over_det_jm * quad_weights,
                optimize='optimal',
            )
            M[re] = csr_matrix(M_re)
        return M

    def _m3_n3_k2(self, degree, quad):
        """mass matrix of 2-form on 3-manifold in 3d space"""
        quad_degree, quad_type = quad
        quad = Quadrature(quad_degree, category=quad_type)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et_sg, bf = self._space.basis_functions[degree](*quad_nodes)
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
                M00 = self._einsum_helper(quad_weights * det_jm * g[1][1]*g[2][2], bf[0], bf[0])
                M11 = self._einsum_helper(quad_weights * det_jm * g[2][2]*g[0][0], bf[1], bf[1])
                M22 = self._einsum_helper(quad_weights * det_jm * g[0][0]*g[1][1], bf[2], bf[2])
                M01 = None
                M02 = None
                M10 = None
                M12 = None
                M20 = None
                M21 = None
            else:
                M00 = self._einsum_helper(quad_weights * det_jm * (g[1][1]*g[2][2]-g[1][2]*g[2][1]), bf[0], bf[0])
                M11 = self._einsum_helper(quad_weights * det_jm * (g[2][2]*g[0][0]-g[2][0]*g[0][2]), bf[1], bf[1])
                M22 = self._einsum_helper(quad_weights * det_jm * (g[0][0]*g[1][1]-g[0][1]*g[1][0]), bf[2], bf[2])
                g12_20_g10_22 = g[1][2] * g[2][0] - g[1][0] * g[2][2]
                g10_21_g11_20 = g[1][0] * g[2][1] - g[1][1] * g[2][0]
                g20_01_g21_00 = g[2][0] * g[0][1] - g[2][1] * g[0][0]
                M01 = self._einsum_helper(quad_weights * det_jm * g12_20_g10_22, bf[0], bf[1])
                M02 = self._einsum_helper(quad_weights * det_jm * g10_21_g11_20, bf[0], bf[2])
                M12 = self._einsum_helper(quad_weights * det_jm * g20_01_g21_00, bf[1], bf[2])
                M10 = M01.T
                M20 = M02.T
                M21 = M12.T

            M[re] = bmat(
                [
                    (M00, M01, M02),
                    (M10, M11, M12),
                    (M20, M21, M22)
                ], format='csr'
            )

        return M

    def _m3_n3_k1(self, degree, quad):
        """mass matrix of 1-form on 3-manifold in 3d space"""
        quad_degree, quad_type = quad
        quad = Quadrature(quad_degree, category=quad_type)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et_sg, bf = self._space.basis_functions[degree](*quad_nodes)
        JM = self._space.mesh.ct.Jacobian_matrix(*xi_et_sg)
        detJM = self._space.mesh.ct.Jacobian(*xi_et_sg, JM=JM)
        iJM = self._space.mesh.ct.inverse_Jacobian_matrix(*xi_et_sg, JM=JM)
        G = self._space.mesh.ct.inverse_metric_matrix(*xi_et_sg, iJM=iJM)
        del JM, iJM
        M = dict()
        for re in detJM:
            det_jm = detJM[re]
            g = G[re]
            M00 = self._einsum_helper(quad_weights * det_jm * g[0][0], bf[0], bf[0])
            M11 = self._einsum_helper(quad_weights * det_jm * g[1][1], bf[1], bf[1])
            M22 = self._einsum_helper(quad_weights * det_jm * g[2][2], bf[2], bf[2])
            reference_mtype = detJM.get_mtype_of_reference_element(re)

            if reference_mtype[:6] == 'Linear':
                M01 = None
                M02 = None
                M10 = None
                M12 = None
                M20 = None
                M21 = None
            else:
                M01 = self._einsum_helper(quad_weights * det_jm * g[0][1], bf[0], bf[1])
                M02 = self._einsum_helper(quad_weights * det_jm * g[0][2], bf[0], bf[2])
                M12 = self._einsum_helper(quad_weights * det_jm * g[1][2], bf[1], bf[2])
                M10 = M01.T
                M20 = M02.T
                M21 = M12.T

            M[re] = bmat(
                [
                    (M00, M01, M02),
                    (M10, M11, M12),
                    (M20, M21, M22)
                ], format='csr'
            )
        return M

    def _m3_n3_k0(self, degree, quad):
        """mass matrix of 0-form on 3-manifold in 3d space"""
        quad_degree, quad_type = quad
        quad = Quadrature(quad_degree, category=quad_type)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et_sg, bf = self._space.basis_functions[degree](*quad_nodes)
        detJM = self._space.mesh.ct.Jacobian(*xi_et_sg)
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

    def _m2_n2_k0(self, degree, quad):
        """mass matrix of 0-form on 2-manifold in 2d space"""
        quad_degree, quad_type = quad
        quad = Quadrature(quad_degree, category=quad_type)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et, bf = self._space.basis_functions[degree](*quad_nodes)
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

    def _m2_n2_k2(self, degree, quad):
        """mass matrix of 2-form on 2-manifold in 2d space"""
        quad_degree, quad_type = quad
        quad = Quadrature(quad_degree, category=quad_type)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et, bf = self._space.basis_functions[degree](*quad_nodes)
        detJM = self._space.mesh.ct.Jacobian(*xi_et)
        bf = bf[0]
        M = dict()
        for re in detJM:  # go through all reference elements
            det_jm = np.reciprocal(detJM[re])
            M_re = np.einsum(
                'im, jm, m -> ij',
                bf, bf, det_jm * quad_weights,
                optimize='optimal',
            )
            M[re] = csr_matrix(M_re)
        return M

    def _m2_n2_k1_inner(self, degree, quad):
        """mass matrix of inner 1-form on 2-manifold in 2d space"""
        quad_degree, quad_type = quad
        quad = Quadrature(quad_degree, category=quad_type)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et, bf = self._space.basis_functions[degree](*quad_nodes)
        JM = self._space.mesh.ct.Jacobian_matrix(*xi_et)
        detJM = self._space.mesh.ct.Jacobian(*xi_et, JM=JM)
        iJM = self._space.mesh.ct.inverse_Jacobian_matrix(*xi_et, JM=JM)
        G = self._space.mesh.ct.inverse_metric_matrix(*xi_et, iJM=iJM)
        del JM, iJM
        M = dict()
        for re in detJM:
            det_jm = detJM[re]
            g = G[re]
            M00 = self._einsum_helper(quad_weights * det_jm * g[0][0], bf[0], bf[0])
            M11 = self._einsum_helper(quad_weights * det_jm * g[1][1], bf[1], bf[1])
            reference_mtype = detJM.get_mtype_of_reference_element(re)

            if reference_mtype[:6] == 'Linear':
                M01 = None
                M10 = None
            else:
                M01 = self._einsum_helper(quad_weights * det_jm * g[0][1], bf[0], bf[1])
                M10 = M01.T

            M[re] = bmat(
                [
                    (M00, M01),
                    (M10, M11)
                ], format='csr'
            )
        return M

    @staticmethod
    def _einsum_helper(metric, bfO, bfS):
        """"""
        M = np.einsum('m, im, jm -> ij', metric, bfO, bfS, optimize='optimal')
        return csr_matrix(M)

    def _m2_n2_k1_outer(self, degree, quad):
        """mass matrix of outer 1-form on 2-manifold in 2d space"""
        quad_degree, quad_type = quad
        quad = Quadrature(quad_degree, category=quad_type)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et, bf = self._space.basis_functions[degree](*quad_nodes)
        JM = self._space.mesh.ct.Jacobian_matrix(*xi_et)
        detJM = self._space.mesh.ct.Jacobian(*xi_et, JM=JM)
        iJM = self._space.mesh.ct.inverse_Jacobian_matrix(*xi_et, JM=JM)
        G = self._space.mesh.ct.inverse_metric_matrix(*xi_et, iJM=iJM)
        del JM, iJM
        M = dict()
        for re in detJM:
            det_jm = detJM[re]
            g = G[re]
            M00 = self._einsum_helper(quad_weights * det_jm * g[1][1], bf[0], bf[0])
            M11 = self._einsum_helper(quad_weights * det_jm * g[0][0], bf[1], bf[1])
            reference_mtype = detJM.get_mtype_of_reference_element(re)

            if reference_mtype[:6] == 'Linear':
                M01 = None
                M10 = None
            else:
                M01 = - self._einsum_helper(quad_weights * det_jm * g[1][0], bf[0], bf[1])
                M10 = M01.T

            M[re] = bmat(
                [
                    (M00, M01),
                    (M10, M11)
                ], format='csr'
            )
        return M

    def _m2_n1_k1_outer(self, degree, quad):
        """"""
        quad_degree, quad_type = quad
        if isinstance(quad_degree, int):
            pass
        else:
            quad_degree = max(quad_degree)
        quad = Quadrature(quad_degree, category=quad_type)
        quad_nodes, quad_weights = quad.quad
        xi_et, bfs = self._space.basis_functions[degree](quad_nodes)

        from msepy.main import base
        meshes = base['meshes']
        boundary_sym = self._mesh.abstract.boundary()._sym_repr
        boundary_section = None
        for sym in meshes:
            if sym == boundary_sym:
                boundary_section = meshes[sym]
                break
            else:
                pass
        assert boundary_section is not None, f"must have found a boundary section."

        p = self._space[degree].p
        nWE, nNS = p
        local_indices = {
            (0, 0): (0, nNS),
            (0, 1): (nNS, 2 * nNS),
            (1, 0): (2 * nNS, 2 * nNS + nWE),
            (1, 1): (2 * nNS + nWE, 2 * nNS + 2 * nWE),
        }

        M = {}
        for i in range(self._mesh.elements._num):
            M[i] = lil_matrix(
                (2 * (nWE + nNS), 2 * (nWE + nNS))
            )
        faces = boundary_section.faces
        for i in faces:
            face = faces[i]
            m, n, element = face._m, face._n, face._element
            ct = face.ct
            quad_nodes = xi_et[(m, n)]
            bf = bfs[(m, n)]
            JM = ct.Jacobian_matrix(quad_nodes)
            Jacobian = np.sqrt(JM[0]**2 + JM[1]**2)
            reciprocal_det_jm = np.reciprocal(Jacobian)
            M_face = np.einsum(
                'im, jm, m -> ij',
                bf, bf, reciprocal_det_jm * quad_weights,
                optimize='optimal',
            )

            if m == 0 and n == 1:  # south
                M_face = - M_face
            elif m == 1 and n == 0:  # West
                M_face = - M_face
            else:
                pass

            i0, i1 = local_indices[(m, n)]
            M[element][i0:i1, i0:i1] = M_face

        for i in M:
            # noinspection PyUnresolvedReferences
            M[i] = M[i].tocsr()

        return 'unique dict', M

    def _m1_n1_k0(self, degree, quad):
        """mass matrix of 0-form on 1-manifold in 1d space"""
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
        """mass matrix of 1-form on 1-manifold in 1d space"""
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
