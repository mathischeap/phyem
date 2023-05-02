

import numpy as np

from tools.frozen import Frozen


class MsePyRootFormReconstructLambda(Frozen):
    """"""

    def __init__(self, rf, t):
        """"""
        self._f = rf
        self._mesh = rf.mesh
        self._space = rf.space
        self._t = t
        self._freeze()

    def __call__(self, *meshgrid, **kwargs):
        """Reconstruct using cochain at time `t` on the mesh grid of `meshgrid_xi_et_sg`."""
        abs_sp = self._space.abstract
        m = abs_sp.m
        n = abs_sp.n
        k = abs_sp.k
        orientation = abs_sp.orientation

        if m == n == 2 and k == 1:
            return getattr(self, f'_m{m}_n{n}_k{k}_{orientation}')(*meshgrid, **kwargs)
        else:
            return getattr(self, f'_m{m}_n{n}_k{k}')(*meshgrid, **kwargs)

    def _m1_n1_k0(self, *meshgrid_xi):
        """"""
        t = self._t
        xi, bf = self._f._evaluate_bf_on(*meshgrid_xi)
        bf = bf[0]   # bf, value
        xi = xi[0]
        x = self._mesh.ct.mapping(xi)
        cochain = self._f.cochain[t].local
        v = np.einsum('ij, ei -> ej', bf, cochain, optimize='optimal')
        return x, (v, )  # here x is already in a tuple, like (x-coo, )

    def _m1_n1_k1(self, *meshgrid_xi):
        """"""
        t = self._t
        xi, bf = self._f._evaluate_bf_on(*meshgrid_xi)
        bf = bf[0]   # bf, value
        xi = xi[0]
        x = self._mesh.ct.mapping(xi)
        iJ = self._mesh.ct.inverse_Jacobian(xi)
        cochain = self._f.cochain[t].local
        cochain_batches = iJ.split(cochain, axis=0)
        value_batches = list()
        for ci, cochain_batch in enumerate(cochain_batches):
            iJ_ci = iJ.get_data_of_cache_index(ci)
            v = np.einsum('ij, ei -> ej', bf * iJ_ci, cochain_batch, optimize='optimal')
            value_batches.append(v)
        v = iJ.merge(value_batches, axis=0)
        return x, (v, )  # here x is already in a tuple, like (x-coo, )

    def _m2_n2_k0(self, *meshgrid_xi_et, ravel=False):
        """"""
        n = 2
        t = self._t
        xi, et = meshgrid_xi_et
        shape: list = [len(xi), len(et)]
        xi_et, bf = self._f._evaluate_bf_on(*meshgrid_xi_et)

        bf = bf[0]   # bf, value
        xy = self._mesh.ct.mapping(*xi_et)
        cochain = self._f.cochain[t].local
        v = np.einsum('ij, ei -> ej', bf, cochain, optimize='optimal')

        x, y = xy
        x, y = x.T, y.T
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
