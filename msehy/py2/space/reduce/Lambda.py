# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from tools.quadrature import Quadrature


class MseHyPy2SpaceReduceLambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._mesh = space.mesh
        self._space = space
        self._cache221 = {}
        self._cache222 = {}
        self._freeze()

    def __call__(self, cf, t, g, degree, **kwargs):
        """"""
        g = self._mesh._pg(g)
        abs_sp = self._space.abstract
        k = abs_sp.k
        orientation = abs_sp.orientation
        if k == 1:
            return getattr(self, f'_k{k}_{orientation}')(cf, t, g, degree, **kwargs)
        else:
            return getattr(self, f'_k{k}')(cf, t, g, degree, **kwargs)

    def _k0(self, cf, t, generation, degree):
        """0-form on 1-manifold in 1d space."""
        nodes = self._space[degree].nodes
        px, py = self._space[degree].p
        assert px == py
        full_local_numbering = np.arange((px+1) * (py+1)).reshape((px+1, py+1), order='F')
        using_dofs = np.concatenate(
            [np.array([0, ]), full_local_numbering[1:, :].ravel('F')]
        )
        xi, et = np.meshgrid(*nodes, indexing='ij')
        xi = xi.ravel('F')
        et = et.ravel('F')
        representative = self._mesh[generation]
        xy = representative.ct.mapping(xi, et)
        func = cf[t]
        local_cochain = dict()
        for e in xy:
            fc = representative[e]
            x, y = xy[e]
            region = fc.region
            value = func[region](x, y)[0]  # the scalar evaluated at time `t`.
            if fc._type == 'q':
                pass
            else:
                value = value[using_dofs]
            local_cochain[e] = value

        return local_cochain

    def _k1_inner(self, cf, t, generation, degree, quad_degree=None):
        """"""
        if quad_degree is None:
            quad_degree = [p + 2 for p in self._space[degree].p]
        else:
            pass

        representative = self._mesh[generation]
        func = cf[t]

        # dx edge cochain, x-axis direction component.
        xi, et, edge_size_d, quad_weights = self._k1_preparation('x', degree, quad_degree)
        xy = representative.ct.mapping(xi, et)
        J = representative.ct.Jacobian_matrix(xi, et)
        cochain_local_dx = dict()

        for e in J:
            fc = representative[e]
            region = fc.region
            x, y = xy[e]
            u, v = func[region](x, y)
            Je = J[e]
            J00, J01 = Je[0]
            J10, J11 = Je[1]

            if not isinstance(J10, np.ndarray) and J10 == 0:
                vdx = np.einsum('ij, ij -> ij', J00, u, optimize='optimal')
            else:
                vdx = np.einsum('ij, ij -> ij', J00, u, optimize='optimal') + \
                      np.einsum('ij, ij -> ij', J10, v, optimize='optimal')

            cochain_local_dx[e] = np.einsum(
                'ij, i, j -> j',
                vdx, quad_weights[0], edge_size_d*0.5,
                optimize='optimal'
            )

        # dy edge cochain, y-axis direction component.
        q_pp = self._k1_preparation('y', degree, quad_degree)
        xi_q, et_q, edge_size_d_q, quad_weights_q = q_pp
        xy_q = representative.ct.mapping(xi_q, et_q)
        J_q = representative.ct.Jacobian_matrix(xi, et_q)

        t_pp = self._k1_preparation('y', degree, quad_degree, triangle_y=True)
        xi_t, et_t, edge_size_d_t, quad_weights_t = t_pp
        xy_t = representative.ct.mapping(xi_t, et_t)
        J_t = representative.ct.Jacobian_matrix(xi_t, et_t)

        cochain_local_dy = dict()
        for e in J:
            fc = representative[e]
            if fc._type == 'q':
                x, y = xy_q[e]
                Je = J_q[e]
                quad_weights = quad_weights_q
                edge_size_d = edge_size_d_q
            elif fc._type == 't':
                x, y = xy_t[e]
                Je = J_t[e]
                quad_weights = quad_weights_t
                edge_size_d = edge_size_d_t
            else:
                raise Exception()

            region = fc.region
            u, v = func[region](x, y)
            J00, J01 = Je[0]
            J10, J11 = Je[1]

            if not isinstance(J01, np.ndarray) and J01 == 0:
                vdy = np.einsum('ij, ij -> ij', J11, v, optimize='optimal')
            else:
                vdy = np.einsum('ij, ij -> ij', J01, u, optimize='optimal') + \
                      np.einsum('ij, ij -> ij', J11, v, optimize='optimal')

            cochain_local_dy[e] = np.einsum(
                'ij, i, j -> j',
                vdy, quad_weights[1], edge_size_d*0.5,
                optimize='optimal'
            )

        # time to merge the two cochain components
        cochain_local = dict()
        for e in cochain_local_dx:
            cochain_local[e] = np.concatenate(
                [cochain_local_dx[e], cochain_local_dy[e]]
            )
        return cochain_local

    def _k1_outer(self, cf, t, generation, degree, quad_degree=None):
        """"""
        if quad_degree is None:
            quad_degree = [p + 2 for p in self._space[degree].p]
        else:
            pass

        representative = self._mesh[generation]
        func = cf[t]

        # dx edge cochain, x-axis direction component.
        xi, et, edge_size_d, quad_weights = self._k1_preparation('x', degree, quad_degree)
        xy = representative.ct.mapping(xi, et)
        J = representative.ct.Jacobian_matrix(xi, et)
        cochain_local_dx = dict()
        for e in J:
            fc = representative[e]
            region = fc.region
            x, y = xy[e]
            u, v = func[region](x, y)
            Je = J[e]
            J00, J01 = Je[0]
            J10, J11 = Je[1]

            if not isinstance(J10, np.ndarray) and J10 == 0:
                vdx = np.einsum('ij, ij -> ij', J00, v, optimize='optimal')
            else:
                vdx = + np.einsum('ij, ij -> ij', J00, v, optimize='optimal') \
                      - np.einsum('ij, ij -> ij', J10, u, optimize='optimal')

            cochain_local_dx[e] = np.einsum(
                'ij, i, j -> j',
                vdx, quad_weights[0], edge_size_d*0.5,
                optimize='optimal'
            )

        # dy edge cochain, y-axis direction component.
        q_pp = self._k1_preparation('y', degree, quad_degree)
        xi_q, et_q, edge_size_d_q, quad_weights_q = q_pp
        xy_q = representative.ct.mapping(xi_q, et_q)
        J_q = representative.ct.Jacobian_matrix(xi, et_q)

        t_pp = self._k1_preparation('y', degree, quad_degree, triangle_y=True)
        xi_t, et_t, edge_size_d_t, quad_weights_t = t_pp
        xy_t = representative.ct.mapping(xi_t, et_t)
        J_t = representative.ct.Jacobian_matrix(xi_t, et_t)

        cochain_local_dy = dict()
        for e in J:
            fc = representative[e]
            if fc._type == 'q':
                x, y = xy_q[e]
                Je = J_q[e]
                quad_weights = quad_weights_q
                edge_size_d = edge_size_d_q
            elif fc._type == 't':
                x, y = xy_t[e]
                Je = J_t[e]
                quad_weights = quad_weights_t
                edge_size_d = edge_size_d_t
            else:
                raise Exception()

            region = fc.region
            u, v = func[region](x, y)
            J00, J01 = Je[0]
            J10, J11 = Je[1]

            if not isinstance(J01, np.ndarray) and J01 == 0:
                vdy = np.einsum('ij, ij -> ij', J11, u, optimize='optimal')
            else:
                vdy = - np.einsum('ij, ij -> ij', J01, v, optimize='optimal') \
                      + np.einsum('ij, ij -> ij', J11, u, optimize='optimal')

            cochain_local_dy[e] = np.einsum(
                'ij, i, j -> j',
                vdy, quad_weights[1], edge_size_d*0.5,
                optimize='optimal'
            )

        # time to merge the two cochain components
        cochain_local = dict()
        for e in cochain_local_dx:
            cochain_local[e] = np.concatenate(
                [cochain_local_dy[e], cochain_local_dx[e]]
            )
        return cochain_local

    def _k1_preparation(self, d_, degree, quad_degree, triangle_y=False):
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
                quad_xi = np.tile(cell_nodes[0], p[1] + 1).reshape(
                    (p_x + 1, p[0] * (p[1] + 1)), order='F')
                quad_eta = np.repeat(nodes[1][np.newaxis, :], p[0], axis=0).ravel('F')
                quad_eta = quad_eta[np.newaxis, :].repeat(p_x + 1, axis=0)
                ES = np.tile(edges_size[0], p[1] + 1)
                data = quad_xi, quad_eta, ES, quad_weights

            elif d_ == 'y':
                quad_xi = np.tile(nodes[0], p[1])[np.newaxis, :].repeat(p_y + 1, axis=0)
                quad_eta = np.repeat(cell_nodes[1].reshape(
                    (p_y + 1, p[1]), order='F'), p[0] + 1, axis=1)
                ES = np.repeat(edges_size[1], p[0] + 1)
                if triangle_y:
                    px, py = p
                    assert px == py
                    dy_local_numbering = np.arange((px+1) * py).reshape((px+1, py), order='F')
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

    def _k2(self, cf, t, generation, degree, quad_degree=None):
        """"""
        if quad_degree is None:
            quad_degree = [p + 2 for p in self._space[degree].p]
        else:
            pass
        xi, et, volume, quad_weights = self._k2_preparation(degree, quad_degree)

        representative = self._mesh[generation]
        xy = representative.ct.mapping(xi, et)
        J = representative.ct.Jacobian(xi, et)
        func = cf[t]
        cochain_local = dict()
        for e in xy:
            fc = representative[e]
            x, y = xy[e]
            region = fc.region
            f = func[region](x, y)[0]
            Je = J[e]
            cochain_local[e] = np.einsum(
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
            for j in range(p[1]):
                for i in range(p[0]):
                    m = i + j*p[0]
                    xi[m, ...] = (quad_nodes[0][:, np.newaxis].repeat(quad_degree[1] + 1, axis=1) + 1) \
                        * (nodes[0][i+1]-nodes[0][i])/2 + nodes[0][i]
                    et[m, ...] = (quad_nodes[1][np.newaxis, :].repeat(quad_degree[0] + 1, axis=0) + 1) \
                        * (nodes[1][j+1]-nodes[1][j])/2 + nodes[1][j]
                    volume[m] = (nodes[0][i+1]-nodes[0][i]) \
                        * (nodes[1][j+1]-nodes[1][j]) * magic_factor
            data = xi, et, volume, quad_weights
            self._cache222[key] = data
        return data
