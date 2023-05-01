

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

    def __call__(self, *meshgrid_xi_et_sg):
        """Reconstruct using cochain at time `t` on the mesh grid of `meshgrid_xi_et_sg`."""
        abs_sp = self._space.abstract
        m = abs_sp.m
        n = abs_sp.n
        k = abs_sp.k
        orientation = abs_sp.orientation

        if m == n == 2 and k == 1:
            return getattr(self, f'_m{m}_n{n}_k{k}_{orientation}')(*meshgrid_xi_et_sg)
        else:
            return getattr(self, f'_m{m}_n{n}_k{k}')(*meshgrid_xi_et_sg)

    def _m1_n1_k0(self, *meshgrid_xi_et_sg):
        """"""
        t = self._t
        xi, bf = self._f._evaluate_bf_on(*meshgrid_xi_et_sg)
        bf = bf[0]   # bf, value
        xi = xi[0]
        x = self._mesh.ct.mapping(xi)
        cochain = self._f.cochain[t].local
        v = np.einsum('ij, ei -> ej', bf, cochain, optimize='optimal')
        return x, (v, )

    def _m1_n1_k1(self, *meshgrid_xi_et_sg):
        """"""
        t = self._t
        xi, bf = self._f._evaluate_bf_on(*meshgrid_xi_et_sg)
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
        return x, (v, )
