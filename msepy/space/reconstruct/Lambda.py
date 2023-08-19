# -*- coding: utf-8 -*-
r"""
"""

import numpy as np

from tools.frozen import Frozen


class MsePySpaceReconstructLambda(Frozen):
    """Reconstruct over all mesh-elements."""

    def __init__(self, space):
        """"""
        self._mesh = space.mesh
        self._space = space
        self._freeze()

    def __call__(self, local_cochain, degree, *meshgrid, **kwargs):
        """Reconstruct using cochain at time `t` on the mesh grid of `meshgrid_xi_et_sg`."""
        abs_sp = self._space.abstract
        m = abs_sp.m
        n = abs_sp.n
        k = abs_sp.k
        orientation = abs_sp.orientation

        if m == n == 2 and k == 1:
            return getattr(self, f'_m{m}_n{n}_k{k}_{orientation}')(local_cochain, degree, *meshgrid, **kwargs)
        else:
            return getattr(self, f'_m{m}_n{n}_k{k}')(local_cochain, degree, *meshgrid, **kwargs)

    def _m1_n1_k0(self, local_cochain, degree, *meshgrid_xi):
        """"""
        xi, bf = self._space.basis_functions[degree](*meshgrid_xi)
        bf = bf[0]   # bf, value
        xi = xi[0]
        x = self._mesh.ct.mapping(xi)
        v = np.einsum('ij, ei -> ej', bf, local_cochain, optimize='optimal')
        return (x[0].T, ), (v, )  # here x is already in a tuple, like (x-coo, )

    def _m1_n1_k1(self, local_cochain, degree, *meshgrid_xi):
        """"""
        xi, bf = self._space.basis_functions[degree](*meshgrid_xi)
        bf = bf[0]   # bf, value
        xi = xi[0]
        x = self._mesh.ct.mapping(xi)
        iJ = self._mesh.ct.inverse_Jacobian(xi)
        cochain_batches = iJ.split(local_cochain, axis=0)
        value_batches = list()
        for ci, cochain_batch in enumerate(cochain_batches):
            iJ_ci = iJ.get_data_of_cache_index(ci)
            v = np.einsum('ij, ei -> ej', bf * iJ_ci, cochain_batch, optimize='optimal')
            value_batches.append(v)
        v = iJ.merge(value_batches, axis=0)
        return (x[0].T, ), (v, )  # here x is already in a tuple, like (x-coo, )

    def _m2_n2_k0(self, local_cochain, degree, *meshgrid_xi_et, ravel=False):
        """"""
        n = 2
        xi, et = meshgrid_xi_et
        shape: list = [len(xi), len(et)]
        xi_et, bf = self._space.basis_functions[degree](*meshgrid_xi_et)

        bf = bf[0]   # bf, value
        xy = self._mesh.ct.mapping(*xi_et)
        x, y = xy
        x, y = x.T, y.T

        v = np.einsum('ij, ei -> ej', bf, local_cochain, optimize='optimal')

        if ravel:
            pass
        else:
            cache = [[] for _ in range(n+1)]  # + 1 because it is a scalar
            for i in range(len(x)):
                cache[0].append(x[i].reshape(shape, order='F'))
                cache[1].append(y[i].reshape(shape, order='F'))
                cache[2].append(v[i].reshape(shape, order='F'))
            x, y, v = [np.array(cache[_]) for _ in range(n+1)]

        return (x, y), (v, )

    def _m2_n2_k1_inner(self, local_cochain, degree, *meshgrid_xi_et, ravel=False):
        """"""
        n = 2
        xi, et = meshgrid_xi_et
        shape: list = [len(xi), len(et)]
        xi_et, bf = self._space.basis_functions[degree](*meshgrid_xi_et)

        num_components = self._space.num_local_dof_components(degree)
        local_0 = local_cochain[:, :num_components[0]]
        local_1 = local_cochain[:, num_components[0]:]

        xy = self._mesh.ct.mapping(*xi_et)
        x, y = xy
        x, y = x.T, y.T

        u = np.einsum('ij, ki -> kj', bf[0], local_0, optimize='optimal')
        v = np.einsum('ij, ki -> kj', bf[1], local_1, optimize='optimal')

        iJ = self._mesh.ct.inverse_Jacobian_matrix(*xi_et)
        u = iJ.split(u, axis=0)
        v = iJ.split(v, axis=0)

        vx, vy = list(), list()
        for ci in iJ.cache_indices:

            iJci = iJ.get_data_of_cache_index(ci)
            iJ0, iJ1 = iJci
            iJ00, iJ01 = iJ0
            iJ10, iJ11 = iJ1

            uci = u[ci]
            vci = v[ci]

            if not isinstance(iJ10, np.ndarray) and iJ10 == 0:
                v0 = uci * iJ00
            else:
                v0 = uci * iJ00 + vci * iJ10

            if not isinstance(iJ01, np.ndarray) and iJ01 == 0:
                v1 = vci * iJ11
            else:
                v1 = uci * iJ01 + vci * iJ11

            vx.append(v0)
            vy.append(v1)

        vx = iJ.merge(vx, axis=0)
        vy = iJ.merge(vy, axis=0)

        if ravel:
            pass
        else:
            cache = [[] for _ in range(n+2)]  # + 2 because it is a vector
            for i in range(len(x)):
                cache[0].append(x[i].reshape(shape, order='F'))
                cache[1].append(y[i].reshape(shape, order='F'))
                cache[2].append(vx[i].reshape(shape, order='F'))
                cache[3].append(vy[i].reshape(shape, order='F'))
            x, y, vx, vy = [np.array(cache[_]) for _ in range(n+2)]

        return (x, y), (vx, vy)

    def _m2_n2_k1_outer(self, local_cochain, degree, *meshgrid_xi_et, ravel=False):
        """"""
        n = 2
        xi, et = meshgrid_xi_et
        shape: list = [len(xi), len(et)]
        xi_et, bf = self._space.basis_functions[degree](*meshgrid_xi_et)

        num_components = self._space.num_local_dof_components(degree)
        local_0 = local_cochain[:, :num_components[0]]
        local_1 = local_cochain[:, num_components[0]:]

        xy = self._mesh.ct.mapping(*xi_et)
        x, y = xy
        x, y = x.T, y.T

        u = np.einsum('ij, ki -> kj', bf[0], local_0, optimize='optimal')
        v = np.einsum('ij, ki -> kj', bf[1], local_1, optimize='optimal')

        iJ = self._mesh.ct.inverse_Jacobian_matrix(*xi_et)
        u = iJ.split(u, axis=0)
        v = iJ.split(v, axis=0)

        vx, vy = list(), list()
        for ci in iJ.cache_indices:

            iJci = iJ.get_data_of_cache_index(ci)
            iJ0, iJ1 = iJci
            iJ00, iJ01 = iJ0
            iJ10, iJ11 = iJ1

            uci = u[ci]
            vci = v[ci]

            if not isinstance(iJ01, np.ndarray) and iJ01 == 0:
                v0 = + uci * iJ11
            else:
                v0 = + uci * iJ11 - vci * iJ01

            if not isinstance(iJ10, np.ndarray) and iJ10 == 0:
                v1 = + vci * iJ00
            else:
                v1 = - uci * iJ10 + vci * iJ00

            vx.append(v0)
            vy.append(v1)

        vx = iJ.merge(vx, axis=0)
        vy = iJ.merge(vy, axis=0)

        if ravel:
            pass
        else:
            cache = [[] for _ in range(n+2)]  # + 2 because it is a vector
            for i in range(len(x)):
                cache[0].append(x[i].reshape(shape, order='F'))
                cache[1].append(y[i].reshape(shape, order='F'))
                cache[2].append(vx[i].reshape(shape, order='F'))
                cache[3].append(vy[i].reshape(shape, order='F'))
            x, y, vx, vy = [np.array(cache[_]) for _ in range(n+2)]

        return (x, y), (vx, vy)

    def _m2_n2_k2(self, local_cochain, degree, *meshgrid_xi_et, ravel=False):
        """"""
        n = 2
        xi, et = meshgrid_xi_et
        shape: list = [len(xi), len(et)]
        xi_et, bf = self._space.basis_functions[degree](*meshgrid_xi_et)
        xy = self._mesh.ct.mapping(*xi_et)
        x, y = xy
        x, y = x.T, y.T
        bf = bf[0]
        iJ = self._mesh.ct.inverse_Jacobian(*xi_et)
        local_cochain = iJ.split(local_cochain, axis=0)
        v = list()
        for ci in iJ.cache_indices:
            iJci = iJ.get_data_of_cache_index(ci)
            cci = local_cochain[ci]

            v.append(
                np.einsum(
                    'mi, j, ij -> mj',
                    cci, iJci, bf,
                    optimize='optimal',
                )
            )
        v = iJ.merge(v, axis=0)
        if ravel:
            pass
        else:
            cache = [[] for _ in range(n+1)]  # + 1 because it is a scalar
            for i in range(len(x)):
                cache[0].append(x[i].reshape(shape, order='F'))
                cache[1].append(y[i].reshape(shape, order='F'))
                cache[2].append(v[i].reshape(shape, order='F'))
            x, y, v = [np.array(cache[_]) for _ in range(n+1)]

        return (x, y), (v, )

    def _m3_n3_k0(self, local_cochain, degree, *meshgrid_xi_et_sg, ravel=False):
        """0-form on 3-manifold in 3d space"""
        n = 3
        xi, et, sg = meshgrid_xi_et_sg
        shape: list = [len(xi), len(et), len(sg)]
        xi_et_sg, bf = self._space.basis_functions[degree](*meshgrid_xi_et_sg)
        bf = bf[0]   # bf, value
        xyz = self._mesh.ct.mapping(*xi_et_sg)
        x, y, z = xyz
        x, y, z = x.T, y.T, z.T

        v = np.einsum('ij, ei -> ej', bf, local_cochain, optimize='optimal')

        if ravel:
            pass
        else:
            cache = [[] for _ in range(n+1)]  # + 1 because it is a scalar
            for i in range(len(x)):
                cache[0].append(x[i].reshape(shape, order='F'))
                cache[1].append(y[i].reshape(shape, order='F'))
                cache[2].append(z[i].reshape(shape, order='F'))
                cache[3].append(v[i].reshape(shape, order='F'))
            x, y, z, v = [np.array(cache[_]) for _ in range(n+1)]

        return (x, y, z), (v, )

    def _m3_n3_k1(self, local_cochain, degree, *meshgrid_xi_et_sg, ravel=False):
        """1-form on 3-manifold in 3d space"""
        n = 3
        xi, et, sg = meshgrid_xi_et_sg
        shape: list = [len(xi), len(et), len(sg)]
        xi_et_sg, bfs = self._space.basis_functions[degree](*meshgrid_xi_et_sg)
        xyz = self._mesh.ct.mapping(*xi_et_sg)
        x, y, z = xyz
        x, y, z = x.T, y.T, z.T

        num_components = self._space.num_local_dof_components(degree)
        local_0 = local_cochain[:, :num_components[0]]
        local_1 = local_cochain[:, num_components[0]:num_components[0]+num_components[1]]
        local_2 = local_cochain[:, -num_components[2]:]

        u = np.einsum('ij, ki -> kj', bfs[0], local_0, optimize='optimal')
        v = np.einsum('ij, ki -> kj', bfs[1], local_1, optimize='optimal')
        w = np.einsum('ij, ki -> kj', bfs[2], local_2, optimize='optimal')

        iJ = self._mesh.ct.inverse_Jacobian_matrix(*xi_et_sg)
        u = iJ.split(u, axis=0)
        v = iJ.split(v, axis=0)
        w = iJ.split(w, axis=0)

        vx, vy, vz = list(), list(), list()
        for ci in iJ.cache_indices:

            iJci = iJ.get_data_of_cache_index(ci)
            iJ0, iJ1, iJ2 = iJci
            iJ00, iJ01, iJ02 = iJ0
            iJ10, iJ11, iJ12 = iJ1
            iJ20, iJ21, iJ22 = iJ2

            uci = u[ci]
            vci = v[ci]
            wci = w[ci]

            v0 = uci * iJ00 + vci * iJ10 + wci * iJ20
            v1 = uci * iJ01 + vci * iJ11 + wci * iJ21
            v2 = uci * iJ02 + vci * iJ12 + wci * iJ22

            vx.append(v0)
            vy.append(v1)
            vz.append(v2)

        vx = iJ.merge(vx, axis=0)
        vy = iJ.merge(vy, axis=0)
        vz = iJ.merge(vz, axis=0)

        if ravel:
            pass
        else:
            cache = [[] for _ in range(n+3)]  # + 3 because it is a scalar
            for i in range(len(x)):
                cache[0].append(x[i].reshape(shape, order='F'))
                cache[1].append(y[i].reshape(shape, order='F'))
                cache[2].append(z[i].reshape(shape, order='F'))
                cache[3].append(vx[i].reshape(shape, order='F'))
                cache[4].append(vy[i].reshape(shape, order='F'))
                cache[5].append(vz[i].reshape(shape, order='F'))
            x, y, z, vx, vy, vz = [np.array(cache[_]) for _ in range(n+3)]

        return (x, y, z), (vx, vy, vz)

    def _m3_n3_k2(self, local_cochain, degree, *meshgrid_xi_et_sg, ravel=False):
        """2-form on 3-manifold in 3d space"""
        n = 3
        xi, et, sg = meshgrid_xi_et_sg
        shape: list = [len(xi), len(et), len(sg)]
        xi_et_sg, bfs = self._space.basis_functions[degree](*meshgrid_xi_et_sg)
        xyz = self._mesh.ct.mapping(*xi_et_sg)
        x, y, z = xyz
        x, y, z = x.T, y.T, z.T

        num_components = self._space.num_local_dof_components(degree)
        local_0 = local_cochain[:, :num_components[0]]
        local_1 = local_cochain[:, num_components[0]:num_components[0]+num_components[1]]
        local_2 = local_cochain[:, -num_components[2]:]

        u = np.einsum('ij, ki -> kj', bfs[0], local_0, optimize='optimal')
        v = np.einsum('ij, ki -> kj', bfs[1], local_1, optimize='optimal')
        w = np.einsum('ij, ki -> kj', bfs[2], local_2, optimize='optimal')

        iJ = self._mesh.ct.inverse_Jacobian_matrix(*xi_et_sg)
        u = iJ.split(u, axis=0)
        v = iJ.split(v, axis=0)
        w = iJ.split(w, axis=0)

        vx, vy, vz = list(), list(), list()
        for ci in iJ.cache_indices:

            ij = iJ.get_data_of_cache_index(ci)

            uci = u[ci]
            vci = v[ci]
            wci = w[ci]

            v0 = \
                uci*(ij[1][1]*ij[2][2] - ij[1][2]*ij[2][1]) + \
                vci*(ij[2][1]*ij[0][2] - ij[2][2]*ij[0][1]) + \
                wci*(ij[0][1]*ij[1][2] - ij[0][2]*ij[1][1])

            v1 = \
                uci*(ij[1][2]*ij[2][0] - ij[1][0]*ij[2][2]) + \
                vci*(ij[2][2]*ij[0][0] - ij[2][0]*ij[0][2]) + \
                wci*(ij[0][2]*ij[1][0] - ij[0][0]*ij[1][2])

            v2 = \
                uci*(ij[1][0]*ij[2][1] - ij[1][1]*ij[2][0]) + \
                vci*(ij[2][0]*ij[0][1] - ij[2][1]*ij[0][0]) + \
                wci*(ij[0][0]*ij[1][1] - ij[0][1]*ij[1][0])

            vx.append(v0)
            vy.append(v1)
            vz.append(v2)

        vx = iJ.merge(vx, axis=0)
        vy = iJ.merge(vy, axis=0)
        vz = iJ.merge(vz, axis=0)

        if ravel:
            pass
        else:
            cache = [[] for _ in range(n+3)]  # + 3 because it is a scalar
            for i in range(len(x)):
                cache[0].append(x[i].reshape(shape, order='F'))
                cache[1].append(y[i].reshape(shape, order='F'))
                cache[2].append(z[i].reshape(shape, order='F'))
                cache[3].append(vx[i].reshape(shape, order='F'))
                cache[4].append(vy[i].reshape(shape, order='F'))
                cache[5].append(vz[i].reshape(shape, order='F'))
            x, y, z, vx, vy, vz = [np.array(cache[_]) for _ in range(n+3)]

        return (x, y, z), (vx, vy, vz)

    def _m3_n3_k3(self, local_cochain, degree, *meshgrid_xi_et_sg, ravel=False):
        """3-form on 3-manifold in 3d space"""
        n = 3
        xi, et, sg = meshgrid_xi_et_sg
        shape: list = [len(xi), len(et), len(sg)]
        xi_et_sg, bf = self._space.basis_functions[degree](*meshgrid_xi_et_sg)
        bf = bf[0]   # bf, value
        xyz = self._mesh.ct.mapping(*xi_et_sg)
        x, y, z = xyz
        x, y, z = x.T, y.T, z.T

        iJ = self._mesh.ct.inverse_Jacobian(*xi_et_sg)
        local_cochain = iJ.split(local_cochain, axis=0)
        v = list()
        for ci in iJ.cache_indices:
            iJci = iJ.get_data_of_cache_index(ci)
            cci = local_cochain[ci]

            v.append(
                np.einsum(
                    'mi, j, ij -> mj',
                    cci, iJci, bf,
                    optimize='optimal',
                )
            )
        v = iJ.merge(v, axis=0)

        if ravel:
            pass
        else:
            cache = [[] for _ in range(n+1)]  # + 1 because it is a scalar
            for i in range(len(x)):
                cache[0].append(x[i].reshape(shape, order='F'))
                cache[1].append(y[i].reshape(shape, order='F'))
                cache[2].append(z[i].reshape(shape, order='F'))
                cache[3].append(v[i].reshape(shape, order='F'))
            x, y, z, v = [np.array(cache[_]) for _ in range(n+1)]

        return (x, y, z), (v, )
