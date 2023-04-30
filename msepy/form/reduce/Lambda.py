

import numpy as np

from tools.frozen import Frozen


class MsePyReduceLambda(Frozen):
    """"""

    def __init__(self, f):
        """"""
        self._f = f
        self._mesh = f.mesh
        self._space = f.space
        self._freeze()

    def __call__(self, t, update_cochain):
        """"""
        abs_sp = self._space.abstract
        m = abs_sp.m
        n = abs_sp.n
        k = abs_sp.k
        orientation = abs_sp.orientation

        if m == n == 2 and k == 1:
            return getattr(self, f'_m{m}_n{n}_k{k}_{orientation}')(t, update_cochain)
        else:
            return getattr(self, f'_m{m}_n{n}_k{k}')(t, update_cochain)

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
