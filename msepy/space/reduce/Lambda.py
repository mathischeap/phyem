# -*- coding: utf-8 -*-
r"""
"""

import numpy as np

from phyem.tools.frozen import Frozen
from phyem.tools.quadrature import Quadrature


class MsePySpaceReduceLambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._mesh = space.mesh
        self._space = space
        self._cache221 = {}
        self._cache222 = {}
        self._cache331 = {}
        self._cache332 = {}
        self._cache333 = {}
        self._freeze()

    def __call__(self, cf, t, degree, **kwargs):
        """"""
        abs_sp = self._space.abstract
        m = abs_sp.m
        n = abs_sp.n
        k = abs_sp.k
        orientation = abs_sp.orientation
        if m == n == 2 and k == 1:
            return getattr(self, f'_m{m}_n{n}_k{k}_{orientation}')(cf, t, degree, **kwargs)
        else:
            return getattr(self, f'_m{m}_n{n}_k{k}')(cf, t, degree, **kwargs)

    def _m1_n1_k0(self, cf, t, degree):
        """"""
        nodes = self._space[degree].nodes
        nodes = nodes[0]
        x = self._mesh.ct.mapping(nodes)[0]
        local_cochain = []
        for ri in cf.field:
            scalar = cf.field[ri][t]  # the scalar evaluated at time `t`.
            start, end = self._mesh.elements._elements_in_region(ri)
            x_region = x[..., start:end]
            local_cochain_region = scalar(x_region)[0]
            local_cochain.append(local_cochain_region)

        local_cochain = np.concatenate(local_cochain, axis=1).T

        return local_cochain

    def _m1_n1_k1(self, cf, t, degree, quad_degree=None):
        """"""
        nodes = self._space[degree].nodes
        nodes = nodes[0]
        edges = self._space[degree].edges
        edges = edges[0]

        p = self._space[degree].p
        p = p[0]

        if quad_degree is None:
            quad_degree = p + 2
        else:
            pass

        quad = Quadrature(quad_degree).quad  # using Gauss quadrature by default.
        quad_nodes, quad_weights = quad

        quad_nodes = (quad_nodes[:, np.newaxis].repeat(p, axis=1) + 1) * edges / 2 + nodes[:-1]

        x = self._mesh.ct.mapping(quad_nodes)[0]
        J = self._mesh.ct.Jacobian(quad_nodes)

        local_cochain = []

        for ri in cf.field:
            scalar = cf.field[ri][t]  # the scalar evaluated at time `t`.
            start, end = self._mesh.elements._elements_in_region(ri)
            x_region = x[..., start:end]
            quad_values = scalar(x_region)[0]
            metric = J(range(start, end))
            local_cochain_region = np.einsum(
                'ijk, i, kij, j -> kj',
                quad_values,
                quad_weights,
                metric,
                edges * 0.5,
                optimize='optimal',
            )
            local_cochain.append(local_cochain_region)

        local_cochain = np.concatenate(local_cochain, axis=0)

        return local_cochain

    def _m2_n2_k0(self, cf, t, degree):
        """0-form on 1-manifold in 1d space."""
        nodes = self._space[degree].nodes
        xi, et = np.meshgrid(*nodes, indexing='ij')
        xi = xi.ravel('F')
        et = et.ravel('F')
        x, y = self._mesh.ct.mapping(xi, et)
        local_cochain = []
        for ri in cf.field:
            scalar = cf.field[ri][t]  # the scalar evaluated at time `t`.
            start, end = self._mesh.elements._elements_in_region(ri)
            x_region = x[..., start:end]
            y_region = y[..., start:end]
            local_cochain_region = scalar(x_region, y_region)[0]

            local_cochain.append(local_cochain_region)

        local_cochain = np.concatenate(local_cochain, axis=1).T

        return local_cochain

    def _m2_n2_k1_inner(self, cf, t, degree, quad_degree=None):
        """"""
        if quad_degree is None:
            quad_degree = [p + 2 for p in self._space[degree].p]
        else:
            pass

        # dx edge cochain, x-axis direction component.
        xi, et, edge_size_d, quad_weights = self._n2_k1_preparation('x', degree, quad_degree)
        x, y = self._mesh.ct.mapping(xi, et)
        J = self._mesh.ct.Jacobian_matrix(xi, et)
        u, v = cf[t](x, y, axis=-1)
        u = J.split(u, axis=2)
        v = J.split(v, axis=2)
        cochain_local_dx = list()

        for ci in J.cache_indices:
            Jci = J.get_data_of_cache_index(ci)
            J00, J01 = Jci[0]
            J10, J11 = Jci[1]

            if not isinstance(J10, np.ndarray) and J10 == 0:
                vdx = np.einsum('ij, ijk -> ijk', J00, u[ci], optimize='optimal')
            else:
                vdx = np.einsum('ij, ijk -> ijk', J00, u[ci], optimize='optimal') + \
                      np.einsum('ij, ijk -> ijk', J10, v[ci], optimize='optimal')

            cochain_local_dx.append(
                np.einsum(
                    'ijk, i, j -> kj',
                    vdx, quad_weights[0], edge_size_d*0.5,
                    optimize='optimal'
                )
            )
        cochain_local_dx = J.merge(cochain_local_dx, axis=0)

        # dy edge cochain, y-axis direction component.
        xi, et, edge_size_d, quad_weights = self._n2_k1_preparation('y', degree, quad_degree)
        x, y = self._mesh.ct.mapping(xi, et)
        J = self._mesh.ct.Jacobian_matrix(xi, et)
        u, v = cf[t](x, y, axis=-1)
        u = J.split(u, axis=2)
        v = J.split(v, axis=2)
        cochain_local_dy = list()

        for ci in J.cache_indices:
            Jci = J.get_data_of_cache_index(ci)
            J00, J01 = Jci[0]
            J10, J11 = Jci[1]

            if not isinstance(J01, np.ndarray) and J01 == 0:
                vdy = np.einsum('ij, ijk -> ijk', J11, v[ci], optimize='optimal')
            else:
                vdy = np.einsum('ij, ijk -> ijk', J01, u[ci], optimize='optimal') + \
                      np.einsum('ij, ijk -> ijk', J11, v[ci], optimize='optimal')

            cochain_local_dy.append(
                np.einsum(
                    'ijk, i, j -> kj',
                    vdy, quad_weights[1], edge_size_d*0.5,
                    optimize='optimal'
                )
            )
        cochain_local_dy = J.merge(cochain_local_dy, axis=0)

        # time to merge the two cochain components
        cochain_local = np.hstack((cochain_local_dx, cochain_local_dy))

        return cochain_local

    def _m2_n2_k1_outer(self, cf, t, degree, quad_degree=None):
        """"""
        if quad_degree is None:
            quad_degree = [p + 2 for p in self._space[degree].p]
        else:
            pass

        # dx edge cochain, x-axis direction component.
        xi, et, edge_size_d, quad_weights = self._n2_k1_preparation('x', degree, quad_degree)
        x, y = self._mesh.ct.mapping(xi, et)
        J = self._mesh.ct.Jacobian_matrix(xi, et)
        u, v = cf[t](x, y, axis=-1)
        u = J.split(u, axis=2)
        v = J.split(v, axis=2)
        cochain_local_dx = list()

        for ci in J.cache_indices:
            Jci = J.get_data_of_cache_index(ci)
            J00, J01 = Jci[0]
            J10, J11 = Jci[1]

            if not isinstance(J10, np.ndarray) and J10 == 0:
                vdx = np.einsum('ij, ijk -> ijk', J00, v[ci], optimize='optimal')
            else:
                vdx = + np.einsum('ij, ijk -> ijk', J00, v[ci], optimize='optimal') \
                      - np.einsum('ij, ijk -> ijk', J10, u[ci], optimize='optimal')

            cochain_local_dx.append(
                np.einsum(
                    'ijk, i, j -> kj',
                    vdx, quad_weights[0], edge_size_d*0.5,
                    optimize='optimal'
                )
            )
        cochain_local_dx = J.merge(cochain_local_dx, axis=0)

        # dy edge cochain, y-axis direction component.
        xi, et, edge_size_d, quad_weights = self._n2_k1_preparation('y', degree, quad_degree)
        x, y = self._mesh.ct.mapping(xi, et)
        J = self._mesh.ct.Jacobian_matrix(xi, et)
        u, v = cf[t](x, y, axis=-1)
        u = J.split(u, axis=2)
        v = J.split(v, axis=2)
        cochain_local_dy = list()

        for ci in J.cache_indices:
            Jci = J.get_data_of_cache_index(ci)
            J00, J01 = Jci[0]
            J10, J11 = Jci[1]

            if not isinstance(J01, np.ndarray) and J01 == 0:
                vdy = np.einsum('ij, ijk -> ijk', J11, u[ci], optimize='optimal')
            else:
                vdy = - np.einsum('ij, ijk -> ijk', J01, v[ci], optimize='optimal') \
                      + np.einsum('ij, ijk -> ijk', J11, u[ci], optimize='optimal')

            cochain_local_dy.append(
                np.einsum(
                    'ijk, i, j -> kj',
                    vdy, quad_weights[1], edge_size_d*0.5,
                    optimize='optimal'
                )
            )
        cochain_local_dy = J.merge(cochain_local_dy, axis=0)

        # time to merge the two cochain components
        cochain_local = np.hstack((cochain_local_dy, cochain_local_dx))

        return cochain_local

    def _n2_k1_preparation(self, d_, degree, quad_degree):
        key = d_ + str(degree) + str(quad_degree)
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
                data = quad_xi, quad_eta, ES, quad_weights
            else:
                raise Exception()

            self._cache221[key] = data
        return data

    def _m2_n2_k2(self, cf, t, degree, quad_degree=None):
        """"""
        if quad_degree is None:
            quad_degree = [p + 2 for p in self._space[degree].p]
        else:
            pass
        xi, et, volume, quad_weights = self._preparation_m2n2k2(degree, quad_degree)
        x, y = self._mesh.ct.mapping(xi, et)
        J = self._mesh.ct.Jacobian(xi, et)
        u = cf[t](x, y, axis=-1)[0]
        u = J.split(u, axis=-1)
        cochain_local = list()
        for ci in J.cache_indices:
            Jci = J.get_data_of_cache_index(ci)
            uci = u[ci]

            cochain_local.append(
                np.einsum(
                    'ijkm, ijk, j, k, i -> mi',
                    uci, Jci, quad_weights[0], quad_weights[1], volume,
                    optimize='optimal',
                )
            )
        cochain_local = J.merge(cochain_local, axis=0)

        return cochain_local

    def _preparation_m2n2k2(self, degree, quad_degree):
        """"""
        key = str(degree) + str(quad_degree)
        if key in self._cache222:
            data = self._cache222[key]
        else:
            p = self._space[degree].p
            quad_degree = quad_degree
            nodes = self._space[degree].nodes
            num_basis = self._space.num_local_dofs(degree)
            quad_nodes, quad_weights = Quadrature(quad_degree).quad
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

    def _m2_n1_k1(self, cf, t, degree, quad_degree=None):
        """"""
        mesh = self._mesh
        num_elements = mesh.elements._num
        p = self._space[degree].p
        nWE, nNS = p

        N_s, N_e = 0, nNS
        S_s, S_e = nNS, 2 * nNS
        W_s, W_e = 2 * nNS, 2 * nNS + nWE
        E_s, E_e = 2 * nNS + nWE, 2 * nNS + 2 * nWE

        if quad_degree is None:

            qd_NS = nNS + 2
            qd_WE = nWE + 2

        else:
            raise NotImplementedError()

        quad = Quadrature(qd_NS).quad  # using Gauss quadrature by default.
        qn_NS, qw_NS = quad

        quad = Quadrature(qd_WE).quad  # using Gauss quadrature by default.
        qn_WE, qw_WE = quad

        nodes = self._space[degree].nodes
        nodes_WE, nodes_NS = nodes
        edges = self._space[degree].edges
        edges_WE, edges_NS = edges

        qn_NS = (qn_NS[:, np.newaxis].repeat(nNS, axis=1) + 1) * edges_NS / 2 + nodes_NS[:-1]
        qn_WE = (qn_WE[:, np.newaxis].repeat(nWE, axis=1) + 1) * edges_WE / 2 + nodes_WE[:-1]

        cochain = np.zeros(
            [num_elements, 2 * (nNS + nWE)],
            dtype=float
        )

        from msepy.main import base
        meshes = base['meshes']
        boundary_sym = mesh.abstract.boundary()._sym_repr
        boundary_section = None
        for sym in meshes:
            if sym == boundary_sym:
                boundary_section = meshes[sym]
                break
            else:
                pass
        assert boundary_section is not None, f"must have found a boundary section."

        faces = boundary_section.faces

        for i in faces:  # go through all trace elements on the boundary.
            face = faces[i]
            m, n = face._m, face._n
            element = face._element
            ct = face.ct
            region = mesh.elements[element].region

            scalar = cf.field[region]

            if m == 0 and n == 0:   # North side
                x, y = ct.mapping(qn_NS)
                JM = face.ct.Jacobian_matrix(qn_NS)
                edges = edges_NS
                qw = qw_NS

            elif m == 0 and n == 1:  # South
                x, y = ct.mapping(qn_NS)
                JM = face.ct.Jacobian_matrix(qn_NS)
                edges = edges_NS
                qw = qw_NS

            elif m == 1 and n == 0:  # West
                x, y = ct.mapping(qn_WE)
                JM = face.ct.Jacobian_matrix(qn_WE)
                edges = edges_WE
                qw = qw_WE

            elif m == 1 and n == 1:  # East
                x, y = ct.mapping(qn_WE)
                JM = face.ct.Jacobian_matrix(qn_WE)
                edges = edges_WE
                qw = qw_WE

            else:
                raise Exception

            scalar_value = scalar(t, x, y)[0]
            Jacobian = np.sqrt(JM[0]**2 + JM[1]**2)

            local_cochain_element = np.einsum(
                'ik, i, ik, k -> k',
                scalar_value,
                qw,
                Jacobian,
                edges * 0.5,
                optimize='optimal',
            )

            if m == 0 and n == 0:   # North side
                cochain[element, N_s:N_e] = local_cochain_element
            elif m == 0 and n == 1:  # South
                cochain[element, S_s:S_e] = local_cochain_element
            elif m == 1 and n == 0:  # West
                cochain[element, W_s:W_e] = local_cochain_element
            elif m == 1 and n == 1:  # East
                cochain[element, E_s:E_e] = local_cochain_element
            else:
                raise Exception

        return cochain

    def _m3_n3_k0(self, cf, t, degree):
        """0-form on 3-manifold in 3d space."""
        nodes = self._space[degree].nodes
        xi, et, sg = np.meshgrid(*nodes, indexing='ij')
        xi = xi.ravel('F')
        et = et.ravel('F')
        sg = sg.ravel('F')
        x, y, z = self._mesh.ct.mapping(xi, et, sg)
        local_cochain = []
        for ri in cf.field:
            scalar = cf.field[ri][t]  # the scalar evaluated at time `t`.
            start, end = self._mesh.elements._elements_in_region(ri)
            x_region = x[..., start:end]
            y_region = y[..., start:end]
            z_region = z[..., start:end]
            local_cochain_region = scalar(x_region, y_region, z_region)[0]

            local_cochain.append(local_cochain_region)

        local_cochain = np.concatenate(local_cochain, axis=1).T

        return local_cochain

    def _m3_n3_k1(self, cf, t, degree, quad_degree=None):
        """1-form on 3-manifold in 3d space."""
        if quad_degree is None:
            quad_degree = [p + 2 for p in self._space[degree].p]
        else:
            pass

        xi, eta, sigma, edge_size_dxi, quad_weights = \
            self._m3n3k1_preparation('x', degree, quad_degree)
        coo_dx = self._mesh.ct.mapping(xi, eta, sigma)
        Jx = self._mesh.ct.Jacobian_matrix(xi, eta, sigma)

        xi, eta, sigma, edge_size_deta, quad_weights = \
            self._m3n3k1_preparation('y', degree, quad_degree)
        coo_dy = self._mesh.ct.mapping(xi, eta, sigma)
        Jy = self._mesh.ct.Jacobian_matrix(xi, eta, sigma)

        xi, eta, sigma, edge_size_dsigma, quad_weights = \
            self._m3n3k1_preparation('z', degree, quad_degree)
        coo_dz = self._mesh.ct.mapping(xi, eta, sigma)
        Jz = self._mesh.ct.Jacobian_matrix(xi, eta, sigma)

        del xi, eta, sigma
        edge_size = (edge_size_dxi, edge_size_deta, edge_size_dsigma)

        ux, vx, wx = cf[t](*coo_dx, axis=-1)
        uy, vy, wy = cf[t](*coo_dy, axis=-1)
        uz, vz, wz = cf[t](*coo_dz, axis=-1)
        ux, uy, uz = ux.transpose(2, 0, 1), uy.transpose(2, 0, 1), uz.transpose(2, 0, 1)
        vx, vy, vz = vx.transpose(2, 0, 1), vy.transpose(2, 0, 1), vz.transpose(2, 0, 1)
        wx, wy, wz = wx.transpose(2, 0, 1), wy.transpose(2, 0, 1), wz.transpose(2, 0, 1)
        ux, vx, wx = Jx.split(ux, axis=0), Jx.split(vx, axis=0), Jx.split(wx, axis=0)
        uy, vy, wy = Jy.split(uy, axis=0), Jy.split(vy, axis=0), Jy.split(wy, axis=0)
        uz, vz, wz = Jz.split(uz, axis=0), Jz.split(vz, axis=0), Jz.split(wz, axis=0)
        local_dx, local_dy, local_dz = list(), list(), list()

        for ci in Jx.cache_indices:
            jx = Jx.get_data_of_cache_index(ci)
            jy = Jy.get_data_of_cache_index(ci)
            jz = Jz.get_data_of_cache_index(ci)
            jx = (jx[0][0], jx[1][0], jx[2][0])
            jy = (jy[0][1], jy[1][1], jy[2][1])
            jz = (jz[0][2], jz[1][2], jz[2][2])

            local_dx.append(
                np.einsum(
                    'eij, i, j -> ej',
                    jx[0] * ux[ci] + jx[1] * vx[ci] + jx[2] * wx[ci],
                    quad_weights[0],
                    edge_size[0]*0.5,
                    optimize='optimal',
                )
            )
            local_dy.append(
                np.einsum(
                    'eij, i, j -> ej',
                    jy[0] * uy[ci] + jy[1] * vy[ci] + jy[2] * wy[ci],
                    quad_weights[1],
                    edge_size[1]*0.5,
                    optimize='optimal',
                    )
            )
            local_dz.append(
                np.einsum(
                    'eij, i, j -> ej',
                    jz[0] * uz[ci] + jz[1] * vz[ci] + jz[2] * wz[ci],
                    quad_weights[2],
                    edge_size[2]*0.5,
                    optimize='optimal',
                    )
            )

        local_dx = Jx.merge(local_dx, axis=0)
        local_dy = Jy.merge(local_dy, axis=0)
        local_dz = Jz.merge(local_dz, axis=0)

        local = np.hstack((local_dx, local_dy, local_dz))

        return local

    def _m3n3k1_preparation(self, d_, degree, quad_degree):
        """
        Parameters
        ----------
        d_ : str, optional
            'x', 'y' or 'z'.
        """
        key = d_ + str(degree) + str(quad_degree)

        if key in self._cache331:
            data = self._cache331[key]
        else:

            p = self._space[degree].p
            quad_nodes, quad_weights = Quadrature(quad_degree, category='Gauss').quad
            quad_num_nodes = [len(quad_nodes_i) for quad_nodes_i in quad_nodes]
            nodes = self._space[degree].nodes

            sbn0 = nodes[0]
            sbn1 = nodes[1]
            sbn2 = nodes[2]

            if d_ == 'x':
                a = sbn0[1:] - sbn0[:-1]
                a = a.ravel('F')
                b = (p[1]+1)*(p[2]+1)
                edge_size_x = np.tile(a, b)
                snb_x = b * p[0]
                D = quad_nodes[0][:, np.newaxis].repeat(snb_x, axis=1) + 1
                assert np.shape(D)[1] == len(edge_size_x)
                xi1 = D * edge_size_x / 2
                xi2 = np.tile(sbn0[:-1], b)
                xi = xi1 + xi2
                eta = np.tile(np.tile(sbn1[:, np.newaxis].repeat(quad_num_nodes[0], axis=1).T,
                                      (p[0], 1)).reshape((quad_num_nodes[0], p[0]*(p[1]+1)), order='F'),
                              (1, p[2]+1))
                sigma = sbn2.repeat(p[0]*(p[1]+1))[np.newaxis, :].repeat(
                    quad_num_nodes[0], axis=0)
                data = [xi, eta, sigma, edge_size_x, quad_weights]

            elif d_ == 'y':
                edge_size_y = np.tile(np.repeat((sbn1[1:] - sbn1[:-1]),
                                                p[0]+1), p[2]+1)
                xi = np.tile(sbn0, p[1]*(p[2]+1))[np.newaxis, :].repeat(
                    quad_num_nodes[1], axis=0)
                snb_y = (p[0] + 1) * p[1] * (p[2] + 1)
                eta1 = (quad_nodes[1][:, np.newaxis].repeat(snb_y, axis=1) + 1) * edge_size_y / 2
                eta2 = np.tile(np.repeat(sbn1[:-1], (p[0]+1)), (p[2]+1))
                eta = eta1 + eta2
                sigma = sbn2.repeat(p[1]*(p[0]+1))[np.newaxis, :].repeat(
                    quad_num_nodes[1], axis=0)
                data = [xi, eta, sigma, edge_size_y, quad_weights]

            elif d_ == 'z':
                edge_size_z = np.repeat((sbn2[1:] - sbn2[:-1]),
                                        p[0]+1).repeat(p[1]+1)
                xi = np.tile(sbn0, (p[1]+1)*(p[2]))[np.newaxis, :].repeat(
                    quad_num_nodes[2], axis=0)
                eta = np.tile(np.repeat(sbn1, (p[0]+1)), p[2])[np.newaxis, :].repeat(
                    quad_num_nodes[2], axis=0)
                snb_z = (p[0] + 1) * (p[1] + 1) * p[2]
                sigma1 = (quad_nodes[2][:, np.newaxis].repeat(snb_z, axis=1) + 1) * edge_size_z / 2
                sigma2 = sbn2[:-1].repeat((p[0]+1)*(p[1]+1))
                sigma = sigma1 + sigma2
                data = [xi, eta, sigma, edge_size_z, quad_weights]

            else:
                raise Exception()

            self._cache331[key] = data

        return data

    def _m3_n3_k2(self, cf, t, degree, quad_degree=None):
        """2-form on 3-manifold in 3d space."""
        if quad_degree is None:
            quad_degree = [p + 2 for p in self._space[degree].p]
        else:
            pass

        quad_nodes, quad_weights = Quadrature(quad_degree, category='Gauss').quad
        key = str(degree) + str(quad_degree)

        if key in self._cache332:
            data = self._cache332[key]

        else:
            num_basis_components = self._space[degree].num_local_dof_components
            p = self._space[degree].p
            nodes = self._space[degree].nodes

            # dy dz face ________________________________________________________________________
            xi = np.zeros((num_basis_components[0], quad_degree[1] + 1, quad_degree[2] + 1))
            et = np.zeros((num_basis_components[0], quad_degree[1] + 1, quad_degree[2] + 1))
            si = np.zeros((num_basis_components[0], quad_degree[1] + 1, quad_degree[2] + 1))
            area_dydz = np.zeros((num_basis_components[0]))
            for k in range(p[2]):
                for j in range(p[1]):
                    for i in range(p[0]+1):
                        m = i + j*(p[0]+1) + k*(p[0]+1)*p[1]
                        xi[m, ...] = np.ones((quad_degree[1] + 1, quad_degree[2] + 1)) * nodes[0][i]
                        et[m, ...] = (quad_nodes[1][:, np.newaxis].repeat(quad_degree[2] + 1, axis=1) + 1) * (
                                nodes[1][j+1]-nodes[1][j]) / 2 + nodes[1][j]
                        si[m, ...] = (quad_nodes[2][np.newaxis, :].repeat((quad_degree[1] + 1), axis=0) + 1) * (
                                nodes[2][k+1]-nodes[2][k]) / 2 + nodes[2][k]
                        area_dydz[m] = (nodes[2][k + 1] - nodes[2][k]) * (nodes[1][j+1]-nodes[1][j])
            coo_x = self._mesh.ct.mapping(xi, et, si)
            Jx = self._mesh.ct.Jacobian_matrix(xi, et, si)
            # dz dx face _________________________________________________________________________
            xi = np.zeros((num_basis_components[1], quad_degree[0] + 1, quad_degree[2] + 1))
            et = np.zeros((num_basis_components[1], quad_degree[0] + 1, quad_degree[2] + 1))
            si = np.zeros((num_basis_components[1], quad_degree[0] + 1, quad_degree[2] + 1))
            area_dzdx = np.zeros((num_basis_components[1]))
            for k in range(p[2]):
                for j in range(p[1]+1):
                    for i in range(p[0]):
                        m = i + j*p[0] + k*(p[1]+1)*p[0]
                        xi[m, ...] = (quad_nodes[0][:, np.newaxis].repeat(quad_degree[2] + 1, axis=1) + 1) * (
                                nodes[0][i+1]-nodes[0][i]) / 2 + nodes[0][i]
                        et[m, ...] = np.ones((quad_degree[0] + 1, quad_degree[2] + 1)) * nodes[1][j]
                        si[m, ...] = (quad_nodes[2][np.newaxis, :].repeat(quad_degree[0] + 1, axis=0) + 1) * (
                                nodes[2][k+1]-nodes[2][k]) / 2 + nodes[2][k]
                        area_dzdx[m] = (nodes[2][k + 1] - nodes[2][k]) * (nodes[0][i + 1] - nodes[0][i])
            coo_y = self._mesh.ct.mapping(xi, et, si)
            Jy = self._mesh.ct.Jacobian_matrix(xi, et, si)
            # dx dy face _________________________________________________________________________
            xi = np.zeros((num_basis_components[2], quad_degree[0] + 1, quad_degree[1] + 1))
            et = np.zeros((num_basis_components[2], quad_degree[0] + 1, quad_degree[1] + 1))
            si = np.zeros((num_basis_components[2], quad_degree[0] + 1, quad_degree[1] + 1))
            area_dxdy = np.zeros((num_basis_components[2]))
            for k in range(p[2]+1):
                for j in range(p[1]):
                    for i in range(p[0]):
                        m = i + j*p[0] + k*p[1]*p[0]
                        xi[m, ...] = (quad_nodes[0][:, np.newaxis].repeat(quad_degree[1] + 1, axis=1) + 1) * (
                                nodes[0][i+1]-nodes[0][i]) / 2 + nodes[0][i]
                        et[m, ...] = (quad_nodes[1][np.newaxis, :].repeat(quad_degree[0] + 1, axis=0) + 1) * (
                                nodes[1][j+1]-nodes[1][j]) / 2 + nodes[1][j]
                        si[m, ...] = np.ones((quad_degree[0] + 1, quad_degree[1] + 1)) * nodes[2][k]
                        area_dxdy[m] = (nodes[1][j + 1] - nodes[1][j]) * (nodes[0][i+1]-nodes[0][i])
            coo_z = self._mesh.ct.mapping(xi, et, si)
            Jz = self._mesh.ct.Jacobian_matrix(xi, et, si)
            # ===================================================================================

            data = (coo_x, Jx, area_dydz,
                    coo_y, Jy, area_dzdx,
                    coo_z, Jz, area_dxdy)

            self._cache332[key] = data

        # --------------------------------------------------------
        coo_x, Jx, area_dydz, \
            coo_y, Jy, area_dzdx, \
            coo_z, Jz, area_dxdy \
            = data

        # dx-perp face ------------------------------------------
        u, v, w = cf[t](*coo_x, axis=-1)
        u, v, w = u.transpose(3, 0, 1, 2), v.transpose(3, 0, 1, 2), w.transpose(3, 0, 1, 2)
        u, v, w = Jx.split(u, axis=0), Jx.split(v, axis=0), Jx.split(w, axis=0)
        local_dydz = list()
        for ci in Jx.cache_indices:
            jx = Jx.get_data_of_cache_index(ci)

            Jx_0 = jx[1][1]*jx[2][2] - jx[1][2]*jx[2][1]
            Jx_1 = jx[2][1]*jx[0][2] - jx[2][2]*jx[0][1]
            Jx_2 = jx[0][1]*jx[1][2] - jx[0][2]*jx[1][1]

            uvw_dydz = Jx_0*u[ci] + Jx_1*v[ci] + Jx_2*w[ci]

            local_dydz.append(
                self._m3n3k2_einsum(
                    uvw_dydz, quad_weights[1], quad_weights[2], area_dydz
                )
            )

        # dy-perp face -----------------------------------------
        u, v, w = cf[t](*coo_y, axis=-1)
        u, v, w = u.transpose(3, 0, 1, 2), v.transpose(3, 0, 1, 2), w.transpose(3, 0, 1, 2)
        u, v, w = Jy.split(u, axis=0), Jy.split(v, axis=0), Jy.split(w, axis=0)
        local_dzdx = list()
        for ci in Jy.cache_indices:
            jy = Jy.get_data_of_cache_index(ci)

            Jy_0 = jy[1][2]*jy[2][0] - jy[1][0]*jy[2][2]
            Jy_1 = jy[2][2]*jy[0][0] - jy[2][0]*jy[0][2]
            Jy_2 = jy[0][2]*jy[1][0] - jy[0][0]*jy[1][2]

            uvw_dzdx = Jy_0*u[ci] + Jy_1*v[ci] + Jy_2*w[ci]

            local_dzdx.append(
                self._m3n3k2_einsum(
                    uvw_dzdx, quad_weights[0], quad_weights[2], area_dzdx
                )
            )

        # dz-perp face ----------------------------------------
        u, v, w = cf[t](*coo_z, axis=-1)
        u, v, w = u.transpose(3, 0, 1, 2), v.transpose(3, 0, 1, 2), w.transpose(3, 0, 1, 2)
        u, v, w = Jz.split(u, axis=0), Jz.split(v, axis=0), Jz.split(w, axis=0)
        local_dxdy = list()
        for ci in Jz.cache_indices:
            jz = Jz.get_data_of_cache_index(ci)

            Jz_0 = jz[1][0]*jz[2][1] - jz[1][1]*jz[2][0]
            Jz_1 = jz[2][0]*jz[0][1] - jz[2][1]*jz[0][0]
            Jz_2 = jz[0][0]*jz[1][1] - jz[0][1]*jz[1][0]

            uvw_dxdy = Jz_0*u[ci] + Jz_1*v[ci] + Jz_2*w[ci]

            local_dxdy.append(
                self._m3n3k2_einsum(
                    uvw_dxdy, quad_weights[0], quad_weights[1], area_dxdy
                )
            )

        # ====================================================
        local_dydz = Jx.merge(local_dydz, axis=0)
        local_dzdx = Jy.merge(local_dzdx, axis=0)
        local_dxdy = Jz.merge(local_dxdy, axis=0)

        local = np.hstack((local_dydz, local_dzdx, local_dxdy))

        return local

    @staticmethod
    def _m3n3k2_einsum(uvw, quad_weights_1, quad_weights_2, area):
        """ """
        return np.einsum(
            'ijkl, kl, j -> ij',
            uvw,
            np.tensordot(quad_weights_1, quad_weights_2, axes=0),
            area*0.25,
            optimize='optimal',
        )

    def _m3_n3_k3(self, cf, t, degree, quad_degree=None):
        """3-form on 3-manifold in 3d space."""
        if quad_degree is None:
            quad_degree = [p + 2 for p in self._space[degree].p]
        else:
            pass

        quad_nodes, quad_weights = Quadrature(quad_degree, category='Gauss').quad

        key = str(degree) + str(quad_degree)
        if key in self._cache333:
            data = self._cache333[key]

        else:
            p = self._space[degree].p
            nodes = self._space[degree].nodes
            num_basis = self._space[degree].num_local_dofs
            xi = np.zeros((num_basis, quad_degree[0] + 1, quad_degree[1] + 1, quad_degree[2] + 1))
            et = np.zeros((num_basis, quad_degree[0] + 1, quad_degree[1] + 1, quad_degree[2] + 1))
            si = np.zeros((num_basis, quad_degree[0] + 1, quad_degree[1] + 1, quad_degree[2] + 1))
            volume = np.zeros(num_basis)

            for k in range(p[2]):
                for j in range(p[1]):
                    for i in range(p[0]):
                        m = i + j*p[0] + k*p[0]*p[1]
                        xi[m, ...] = (quad_nodes[0][:, np.newaxis].repeat(
                            quad_degree[1] + 1, axis=1
                        )[:, :, np.newaxis].repeat(quad_degree[2] + 1, axis=2) + 1) \
                            * (nodes[0][i+1]-nodes[0][i])/2 + nodes[0][i]

                        et[m, ...] = (quad_nodes[1][np.newaxis, :].repeat(
                            quad_degree[0] + 1, axis=0
                        )[:, :, np.newaxis].repeat(quad_degree[2] + 1, axis=2) + 1) \
                            * (nodes[1][j+1]-nodes[1][j])/2 + nodes[1][j]

                        si[m, ...] = (quad_nodes[2][np.newaxis, :].repeat(
                            quad_degree[1] + 1, axis=0
                        )[np.newaxis, :, :].repeat(quad_degree[0] + 1, axis=0) + 1) \
                            * (nodes[2][k+1]-nodes[2][k])/2 + nodes[2][k]

                        volume[m] = (nodes[0][i+1]-nodes[0][i]) \
                            * (nodes[1][j+1]-nodes[1][j]) \
                            * (nodes[2][k+1]-nodes[2][k])

            data = xi, et, si, volume * 0.125
            self._cache333[key] = data

        xi, et, si, volume = data

        detJ = self._mesh.ct.Jacobian(xi, et, si)
        xyz = self._mesh.ct.mapping(xi, et, si)
        f = cf[t](*xyz, axis=-1)[0]
        f = f.transpose(4, 0, 1, 2, 3)
        f = detJ.split(f, axis=0)
        local = list()
        for ci in detJ.cache_indices:
            det_j = detJ.get_data_of_cache_index(ci)

            local.append(
                np.einsum(
                    'ecijk, i, j, k, c -> ec',
                    f[ci]*det_j,
                    quad_weights[0], quad_weights[1], quad_weights[2],
                    volume,
                    optimize='optimal',
                )
            )

        local = detJ.merge(local, axis=0)

        return local
