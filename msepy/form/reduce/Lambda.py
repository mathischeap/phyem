# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""

import numpy as np

from tools.frozen import Frozen
from tools.quadrature import Quadrature


class MsePyReduceLambda(Frozen):
    """"""

    def __init__(self, f):
        """"""
        self._f = f
        self._mesh = f.mesh
        self._space = f.space
        self._cache1 = {}
        self._freeze()

    def __call__(self, t, update_cochain, **kwargs):
        """"""
        abs_sp = self._space.abstract
        m = abs_sp.m
        n = abs_sp.n
        k = abs_sp.k
        orientation = abs_sp.orientation

        if m == n == 2 and k == 1:
            return getattr(self, f'_m{m}_n{n}_k{k}_{orientation}')(t, update_cochain, **kwargs)
        else:
            return getattr(self, f'_m{m}_n{n}_k{k}')(t, update_cochain, **kwargs)

    def _m1_n1_k0(self, t, update_cochain):
        """"""
        cf = self._f.cf
        nodes = self._space[self._f.degree].nodes
        nodes = nodes[0]
        x = self._f.mesh.ct.mapping(nodes)[0]
        local_cochain = []
        for ri in cf.field:
            scalar = cf.field[ri][t]  # the scalar evaluated at time `t`.
            start, end = self._f.mesh.elements._elements_in_region(ri)
            x_region = x[..., start:end]
            local_cochain_region = scalar(x_region)[0]
            local_cochain.append(local_cochain_region)

        local_cochain = np.concatenate(local_cochain, axis=1).T

        if update_cochain:
            self._f[t].cochain = local_cochain
        else:
            pass
        return local_cochain

    def _m1_n1_k1(self, t, update_cochain, quad_degree=None):
        """"""
        cf = self._f.cf
        nodes = self._space[self._f.degree].nodes
        nodes = nodes[0]
        edges = self._space[self._f.degree].edges
        edges = edges[0]

        p = self._space[self._f.degree].p
        p = p[0]

        if quad_degree is None:
            quad_degree = p + 2
        else:
            pass

        quad = Quadrature(quad_degree).quad  # using Gauss quadrature by default.
        quad_nodes, quad_weights = quad

        quad_nodes = (quad_nodes[:, np.newaxis].repeat(p, axis=1) + 1) * edges / 2 + nodes[:-1]

        x = self._f.mesh.ct.mapping(quad_nodes)[0]
        J = self._f.mesh.ct.Jacobian(quad_nodes)

        local_cochain = []

        for ri in cf.field:
            scalar = cf.field[ri][t]  # the scalar evaluated at time `t`.
            start, end = self._f.mesh.elements._elements_in_region(ri)
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

        if update_cochain:
            self._f[t].cochain = local_cochain
        else:
            pass
        return local_cochain

    def _m2_n2_k0(self, t, update_cochain):
        """0-form on 1-manifold in 1d space."""
        cf = self._f.cf
        nodes = self._space[self._f.degree].nodes
        xi, et = np.meshgrid(*nodes, indexing='ij')
        xi = xi.ravel('F')
        et = et.ravel('F')
        x, y = self._f.mesh.ct.mapping(xi, et)
        local_cochain = []
        for ri in cf.field:
            scalar = cf.field[ri][t]  # the scalar evaluated at time `t`.
            start, end = self._f.mesh.elements._elements_in_region(ri)
            x_region = x[..., start:end]
            y_region = y[..., start:end]
            local_cochain_region = scalar(x_region, y_region)[0]

            local_cochain.append(local_cochain_region)

        local_cochain = np.concatenate(local_cochain, axis=1).T

        if update_cochain:
            self._f[t].cochain = local_cochain
        else:
            pass
        return local_cochain

    def _m2_n2_k1_inner(self, t, update_cochain, quad_degree=None):
        """"""
        if quad_degree is None:
            quad_degree = [p + 2 for p in self._space[self._f.degree].p]
        else:
            pass
        xi_x, et_x, edge_size_d_xi, quad_weights = self._n2_k1_preparation('x', quad_degree)
        xi_y, et_y, edge_size_d_et, quad_weights = self._n2_k1_preparation('y', quad_degree)

        # dx edge cochain, x-axis direction component.
        x, y = self._mesh.ct.mapping(xi_x, et_x)
        # u, v = self._f.cf[t](x, y)
        J = self._mesh.ct.Jacobian_matrix(xi_x, et_x)
        # x = J.split(x, axis=2)
        # y = J.split(y, axis=2)
        for ci in J.cache_indices:
            Jci = J.get_data_of_cache_index(ci)
            J0, J1 = Jci
            J00, J01 = J0
            J10, J11 = J1



    def _n2_k1_preparation(self, d_, quad_degree):
        key = d_ + str(quad_degree)
        if key in self._cache1:
            data = self._cache1[key]
        else:
            nodes = self._space[self._f.degree].nodes
            p = self._space[self._f.degree].p
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

            self._cache1[key] = data

        return data