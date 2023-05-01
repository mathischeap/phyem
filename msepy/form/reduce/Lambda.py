

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
        for ri in cf:
            scalar = cf[ri][t]  # the scalar evaluated at time `t`.
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

        for ri in cf:
            scalar = cf[ri][t]  # the scalar evaluated at time `t`.
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
