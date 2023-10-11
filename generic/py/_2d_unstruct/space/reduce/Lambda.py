# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from tools.quadrature import Quadrature


class ReduceLambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._mesh = space.mesh
        self._indicator = space.abstract.indicator
        self._k = space.abstract.k
        self._orientation = space.abstract.orientation
        self._cache221 = {}
        self._cache222 = {}
        self._cache_k0 = {}
        self._freeze()

    def __call__(self, target, t, degree):
        """We reduce the `target` at `t` to a form in space of `degree`.

        Parameters
        ----------
        target
        t
        degree

        Returns
        -------

        """
        k = self._k
        orientation = self._orientation
        if k == 1:
            return getattr(self, f'_k{k}_{orientation}')(target, t, degree)
        else:
            return getattr(self, f'_k{k}')(target, t, degree)

    def _k0_read_cache(self, degree):
        if degree in self._cache_k0:
            return self._cache_k0[degree]
        else:
            nodes = self._space[degree].nodes
            p = self._space[degree].p
            full_local_numbering = np.arange((p+1) * (p+1)).reshape((p+1, p+1), order='F')
            using_dofs = np.concatenate(
                [np.array([0, ]), full_local_numbering[1:, :].ravel('F')]
            )
            xi, et = np.meshgrid(*nodes, indexing='ij')
            xi = xi.ravel('F')
            et = et.ravel('F')
            self._cache_k0[degree] = using_dofs, xi, et
            return using_dofs, xi, et

    def _k0(self, target, t, degree):
        """0-form on 1-manifold in 1d space."""
        using_dofs, xi, et = self._k0_read_cache(degree)
        xy = self._mesh.ct.mapping(xi, et)
        local_cochain = dict()
        for index in xy:
            x, y = xy[index]
            value = target(t, x, y)[0]  # the scalar evaluated at time `t`.
            ele_type = self._mesh[index].type
            if ele_type == 'q':
                pass
            elif ele_type == 't':
                value = value[using_dofs]
            else:
                raise Exception()
            local_cochain[index] = value
        return local_cochain

    def _k0_local(self, func, degree, element_range):
        """0-form on 1-manifold in 1d space."""
        using_dofs, xi, et = self._k0_read_cache(degree)
        xy = self._mesh.ct.mapping(xi, et, element_range=element_range)
        local_cochain = dict()
        for index in xy:
            x, y = xy[index]
            value = func(x, y)  # the scalar evaluated at time `t`.
            ele_type = self._mesh[index].type
            if ele_type == 'q':
                pass
            elif ele_type == 't':
                value = value[using_dofs]
            else:
                raise Exception()
            local_cochain[index] = value

        return local_cochain

    def _k1_inner(self, target, t, degree):
        """"""
        p = self._space[degree].p
        quad_degree = [p + 2, p + 2]

        # dx edge cochain, x-axis direction component.
        xi, et, edge_size_d, quad_weights = self._k1_preparation('x', degree, quad_degree)
        xy = self._mesh.ct.mapping(xi, et)
        J = self._mesh.ct.Jacobian_matrix(xi, et)
        cochain_local_dx = dict()

        for index in J:
            x, y = xy[index]
            u, v = target(t, x, y)
            Je = J[index]
            J00, J01 = Je[0]
            J10, J11 = Je[1]

            vdx = np.einsum('ij, ij -> ij', J00, u, optimize='optimal') + \
                np.einsum('ij, ij -> ij', J10, v, optimize='optimal')

            cochain_local_dx[index] = np.einsum(
                'ij, i, j -> j',
                vdx, quad_weights[0], edge_size_d*0.5,
                optimize='optimal'
            )

        # dy edge cochain, y-axis direction component.
        q_pp = self._k1_preparation('y', degree, quad_degree)
        xi_q, et_q, edge_size_d_q, quad_weights_q = q_pp
        xy_q = self._mesh.ct.mapping(xi_q, et_q)
        J_q = self._mesh.ct.Jacobian_matrix(xi, et_q)

        t_pp = self._k1_preparation('y', degree, quad_degree, triangle_y=True)
        xi_t, et_t, edge_size_d_t, quad_weights_t = t_pp
        xy_t = self._mesh.ct.mapping(xi_t, et_t)
        J_t = self._mesh.ct.Jacobian_matrix(xi_t, et_t)

        cochain_local_dy = dict()
        for index in J:
            element = self._mesh[index]
            if element.type == 'q':
                x, y = xy_q[index]
                Je = J_q[index]
                quad_weights = quad_weights_q
                edge_size_d = edge_size_d_q
            elif element.type == 't':
                x, y = xy_t[index]
                Je = J_t[index]
                quad_weights = quad_weights_t
                edge_size_d = edge_size_d_t
            else:
                raise Exception()

            u, v = target(t, x, y)
            J00, J01 = Je[0]
            J10, J11 = Je[1]

            vdy = np.einsum('ij, ij -> ij', J01, u, optimize='optimal') + \
                np.einsum('ij, ij -> ij', J11, v, optimize='optimal')

            cochain_local_dy[index] = np.einsum(
                'ij, i, j -> j',
                vdy, quad_weights[1], edge_size_d*0.5,
                optimize='optimal'
            )

        # time to merge the two cochain components
        cochain_local = dict()
        csm = self._space.basis_functions.csm(degree)

        for index in cochain_local_dx:
            _dx = cochain_local_dx[index]
            _dy = cochain_local_dy[index]
            _ = np.concatenate(
                [_dx, _dy]
            )
            if index in csm:
                _ = csm[index] @ _
            else:
                pass
            cochain_local[index] = _

        return cochain_local

    def _k1_inner_local(self, func_x, func_y, degree, element_range):
        """"""
        p = self._space[degree].p
        quad_degree = [p + 2, p + 2]

        # dx edge cochain, x-axis direction component.
        xi, et, edge_size_d, quad_weights = self._k1_preparation('x', degree, quad_degree)
        xy = self._mesh.ct.mapping(xi, et, element_range=element_range)
        J = self._mesh.ct.Jacobian_matrix(xi, et, element_range=element_range)
        cochain_local_dx = dict()

        for index in J:
            x, y = xy[index]
            u, v = func_x(x, y), func_y(x, y)
            Je = J[index]
            J00, J01 = Je[0]
            J10, J11 = Je[1]

            vdx = np.einsum('ij, ij -> ij', J00, u, optimize='optimal') + \
                np.einsum('ij, ij -> ij', J10, v, optimize='optimal')

            cochain_local_dx[index] = np.einsum(
                'ij, i, j -> j',
                vdx, quad_weights[0], edge_size_d*0.5,
                optimize='optimal'
            )

        # dy edge cochain, y-axis direction component.
        q_pp = self._k1_preparation('y', degree, quad_degree)
        xi_q, et_q, edge_size_d_q, quad_weights_q = q_pp
        xy_q = self._mesh.ct.mapping(xi_q, et_q, element_range=element_range)
        J_q = self._mesh.ct.Jacobian_matrix(xi, et_q, element_range=element_range)

        t_pp = self._k1_preparation('y', degree, quad_degree, triangle_y=True)
        xi_t, et_t, edge_size_d_t, quad_weights_t = t_pp
        xy_t = self._mesh.ct.mapping(xi_t, et_t, element_range=element_range)
        J_t = self._mesh.ct.Jacobian_matrix(xi_t, et_t, element_range=element_range)

        cochain_local_dy = dict()
        for index in J:
            element = self._mesh[index]
            if element.type == 'q':
                x, y = xy_q[index]
                Je = J_q[index]
                quad_weights = quad_weights_q
                edge_size_d = edge_size_d_q
            elif element.type == 't':
                x, y = xy_t[index]
                Je = J_t[index]
                quad_weights = quad_weights_t
                edge_size_d = edge_size_d_t
            else:
                raise Exception()

            u, v = func_x(x, y), func_y(x, y)
            J00, J01 = Je[0]
            J10, J11 = Je[1]

            vdy = np.einsum('ij, ij -> ij', J01, u, optimize='optimal') + \
                np.einsum('ij, ij -> ij', J11, v, optimize='optimal')

            cochain_local_dy[index] = np.einsum(
                'ij, i, j -> j',
                vdy, quad_weights[1], edge_size_d*0.5,
                optimize='optimal'
            )

        # time to merge the two cochain components
        cochain_local = dict()

        for index in cochain_local_dx:
            _dx = cochain_local_dx[index]
            _dy = cochain_local_dy[index]
            cochain_local[index] = np.concatenate(
                [_dx, _dy]
            )
        return cochain_local

    def _k1_preparation(self, d_, degree, quad_degree, triangle_y=False):
        """"""
        key = d_ + str(degree) + str(quad_degree) + str(triangle_y)
        if key in self._cache221:
            data = self._cache221[key]
        else:
            nodes = self._space[degree].nodes
            p = self._space[degree].p
            qp = quad_degree
            quad_nodes, quad_weights = Quadrature(qp, category='Gauss').quad
            p_x, p_y = qp
            edges_size = [nodes[i][1:] - nodes[i][:-1] for i in range(2)]
            cell_nodes = [(0.5 * (edges_size[i][np.newaxis, :]) * (quad_nodes[i][:, np.newaxis] + 1)
                           + nodes[i][:-1]).ravel('F') for i in range(2)]

            if d_ == 'x':
                quad_xi = np.tile(cell_nodes[0], p + 1).reshape(
                    (p_x + 1, p * (p + 1)), order='F')
                quad_eta = np.repeat(nodes[1][np.newaxis, :], p, axis=0).ravel('F')
                quad_eta = quad_eta[np.newaxis, :].repeat(p_x + 1, axis=0)
                ES = np.tile(edges_size[0], p + 1)
                data = quad_xi, quad_eta, ES, quad_weights

            elif d_ == 'y':
                quad_xi = np.tile(nodes[0], p)[np.newaxis, :].repeat(p_y + 1, axis=0)
                quad_eta = np.repeat(cell_nodes[1].reshape(
                    (p_y + 1, p), order='F'), p + 1, axis=1)
                ES = np.repeat(edges_size[1], p + 1)
                if triangle_y:
                    dy_local_numbering = np.arange((p+1) * p).reshape((p+1, p), order='F')
                    using_dofs = dy_local_numbering[1:, :].ravel('F')
                    quad_xi = quad_xi[:, using_dofs]
                    quad_eta = quad_eta[:, using_dofs]
                    ES = ES[using_dofs]
                else:
                    pass
                data = quad_xi, quad_eta, ES, quad_weights
            else:
                raise Exception()

            self._cache221[key] = data
        return data

    def _k1_outer(self, target, t, degree):
        """"""
        p = self._space[degree].p
        quad_degree = [p + 2, p + 2]

        # dx edge cochain, x-axis direction component.
        xi, et, edge_size_d, quad_weights = self._k1_preparation('x', degree, quad_degree)
        xy = self._mesh.ct.mapping(xi, et)
        J = self._mesh.ct.Jacobian_matrix(xi, et)
        cochain_local_dx = dict()
        for index in J:
            x, y = xy[index]
            u, v = target(t, x, y)
            j = J[index]
            J00, J01 = j[0]
            J10, J11 = j[1]

            vdx = + np.einsum('ij, ij -> ij', J00, v, optimize='optimal') \
                - np.einsum('ij, ij -> ij', J10, u, optimize='optimal')

            cochain_local_dx[index] = np.einsum(
                'ij, i, j -> j',
                vdx, quad_weights[0], edge_size_d*0.5,
                optimize='optimal'
            )

        # dy edge cochain, y-axis direction component.
        q_pp = self._k1_preparation('y', degree, quad_degree)
        xi_q, et_q, edge_size_d_q, quad_weights_q = q_pp
        xy_q = self._mesh.ct.mapping(xi_q, et_q)
        J_q = self._mesh.ct.Jacobian_matrix(xi, et_q)

        t_pp = self._k1_preparation('y', degree, quad_degree, triangle_y=True)
        xi_t, et_t, edge_size_d_t, quad_weights_t = t_pp
        xy_t = self._mesh.ct.mapping(xi_t, et_t)
        J_t = self._mesh.ct.Jacobian_matrix(xi_t, et_t)

        cochain_local_dy = dict()
        for index in J:
            element = self._mesh[index]
            if element.type == 'q':
                x, y = xy_q[index]
                Je = J_q[index]
                quad_weights = quad_weights_q
                edge_size_d = edge_size_d_q
            elif element.type == 't':
                x, y = xy_t[index]
                Je = J_t[index]
                quad_weights = quad_weights_t
                edge_size_d = edge_size_d_t
            else:
                raise Exception()

            u, v = target(t, x, y)
            J00, J01 = Je[0]
            J10, J11 = Je[1]

            vdy = - np.einsum('ij, ij -> ij', J01, v, optimize='optimal') \
                + np.einsum('ij, ij -> ij', J11, u, optimize='optimal')

            cochain_local_dy[index] = np.einsum(
                'ij, i, j -> j',
                vdy, quad_weights[1], edge_size_d*0.5,
                optimize='optimal'
            )

        # time to merge the two cochain components
        cochain_local = dict()
        csm = self._space.basis_functions.csm(degree)

        for index in cochain_local_dx:
            _dy = cochain_local_dy[index]
            _dx = cochain_local_dx[index]

            _ = np.concatenate(
                [_dy, _dx]
            )
            if index in csm:
                _ = csm[index] @ _
            else:
                pass
            cochain_local[index] = _

        return cochain_local

    def _k1_outer_local(self, func_x, func_y, degree, element_range):
        """"""
        p = self._space[degree].p
        quad_degree = [p + 2, p + 2]

        # dx edge cochain, x-axis direction component.
        xi, et, edge_size_d, quad_weights = self._k1_preparation('x', degree, quad_degree)
        xy = self._mesh.ct.mapping(xi, et, element_range=element_range)
        J = self._mesh.ct.Jacobian_matrix(xi, et, element_range=element_range)
        cochain_local_dx = dict()
        for index in J:
            x, y = xy[index]
            u, v = func_x(x, y), func_y(x, y)
            j = J[index]
            J00, J01 = j[0]
            J10, J11 = j[1]

            vdx = + np.einsum('ij, ij -> ij', J00, v, optimize='optimal') \
                - np.einsum('ij, ij -> ij', J10, u, optimize='optimal')

            cochain_local_dx[index] = np.einsum(
                'ij, i, j -> j',
                vdx, quad_weights[0], edge_size_d*0.5,
                optimize='optimal'
            )

        # dy edge cochain, y-axis direction component.
        q_pp = self._k1_preparation('y', degree, quad_degree)
        xi_q, et_q, edge_size_d_q, quad_weights_q = q_pp
        xy_q = self._mesh.ct.mapping(xi_q, et_q, element_range=element_range)
        J_q = self._mesh.ct.Jacobian_matrix(xi, et_q, element_range=element_range)

        t_pp = self._k1_preparation('y', degree, quad_degree, triangle_y=True)
        xi_t, et_t, edge_size_d_t, quad_weights_t = t_pp
        xy_t = self._mesh.ct.mapping(xi_t, et_t, element_range=element_range)
        J_t = self._mesh.ct.Jacobian_matrix(xi_t, et_t, element_range=element_range)

        cochain_local_dy = dict()
        for index in J:
            element = self._mesh[index]
            if element.type == 'q':
                x, y = xy_q[index]
                Je = J_q[index]
                quad_weights = quad_weights_q
                edge_size_d = edge_size_d_q
            elif element.type == 't':
                x, y = xy_t[index]
                Je = J_t[index]
                quad_weights = quad_weights_t
                edge_size_d = edge_size_d_t
            else:
                raise Exception()

            u, v = func_x(x, y), func_y(x, y)
            J00, J01 = Je[0]
            J10, J11 = Je[1]

            vdy = - np.einsum('ij, ij -> ij', J01, v, optimize='optimal') \
                + np.einsum('ij, ij -> ij', J11, u, optimize='optimal')

            cochain_local_dy[index] = np.einsum(
                'ij, i, j -> j',
                vdy, quad_weights[1], edge_size_d*0.5,
                optimize='optimal'
            )

        # time to merge the two cochain components
        cochain_local = dict()

        for index in cochain_local_dx:
            _dy = cochain_local_dy[index]
            _dx = cochain_local_dx[index]
            cochain_local[index] = np.concatenate(
                [_dy, _dx]
            )
        return cochain_local

    def _k2(self, target, t, degree):
        """"""
        p = self._space[degree].p
        quad_degree = [p + 2, p + 2]
        xi, et, volume, quad_weights = self._k2_preparation(degree, quad_degree)
        xy = self._mesh.ct.mapping(xi, et)
        J = self._mesh.ct.Jacobian(xi, et)
        cochain_local = dict()
        for index in xy:
            x, y = xy[index]
            f = target(t, x, y)[0]
            Je = J[index]
            cochain_local[index] = np.einsum(
                'ijk, ijk, i, j, k -> i',
                f, Je, volume, quad_weights[0], quad_weights[1],
                optimize='optimal',
            )
        return cochain_local

    def _k2_local(self, func, degree, element_range):
        """"""
        p = self._space[degree].p
        quad_degree = [p + 2, p + 2]
        xi, et, volume, quad_weights = self._k2_preparation(degree, quad_degree)
        xy = self._mesh.ct.mapping(xi, et, element_range=element_range)
        J = self._mesh.ct.Jacobian(xi, et, element_range=element_range)
        cochain_local = dict()
        for index in xy:
            x, y = xy[index]
            f = func(x, y)
            Je = J[index]
            cochain_local[index] = np.einsum(
                'ijk, ijk, i, j, k -> i',
                f, Je, volume, quad_weights[0], quad_weights[1],
                optimize='optimal',
            )
        return cochain_local

    def _k2_preparation(self, degree, quad_degree):
        """"""
        key = str(degree) + str(quad_degree)
        if key in self._cache222:
            data = self._cache222[key]
        else:
            p = self._space[degree].p
            quad_degree = quad_degree
            nodes = self._space[degree].nodes
            num_basis = self._space.num_local_dofs(degree)['q']
            quad_nodes, quad_weights = Quadrature(quad_degree, category='Gauss').quad
            magic_factor = 0.25
            xi = np.zeros((num_basis, quad_degree[0] + 1, quad_degree[1] + 1))
            et = np.zeros((num_basis, quad_degree[0] + 1, quad_degree[1] + 1))
            volume = np.zeros(num_basis)
            for j in range(p):
                for i in range(p):
                    m = i + j*p
                    xi[m, ...] = (quad_nodes[0][:, np.newaxis].repeat(quad_degree[1] + 1, axis=1) + 1) \
                        * (nodes[0][i+1]-nodes[0][i])/2 + nodes[0][i]
                    et[m, ...] = (quad_nodes[1][np.newaxis, :].repeat(quad_degree[0] + 1, axis=0) + 1) \
                        * (nodes[1][j+1]-nodes[1][j])/2 + nodes[1][j]
                    volume[m] = (nodes[0][i+1]-nodes[0][i]) \
                        * (nodes[1][j+1]-nodes[1][j]) * magic_factor
            data = xi, et, volume, quad_weights
            self._cache222[key] = data
        return data
