# -*- coding: utf-8 -*-
r"""
"""
from numpy import sin, cos, pi, zeros_like

from phyem.tools.frozen import Frozen
from phyem.tools.functions.time_space._3d.wrappers.vector import T3dVector
from phyem.tools.functions.time_space._3d.wrappers.scalar import T3dScalar


class MHD3_Helicity_Conservation_test1(Frozen):
    """See Section 4.2 of [Kaibo Hu, Young-Ju Lee, Jinchao Xu, Helicity-conservative finite element discretization
    for incompressible MHD systems, JCP 436 (2021) 110284]

    It was used for conservation tests.
    """

    def __init__(self):
        """"""
        self._u_init_ = None
        self._w_init_ = None
        self._B_init_ = None
        self._j_init_ = None
        self._freeze()

    def _u1_(self, t, x, y, z):
        """"""
        return - sin(pi * ( x - 1/2)) * cos(pi * (y - 1/2)) * z * (z-1)

    def _u2_(self, t, x, y, z):
        """"""
        return cos(pi * (x - 1/2)) * sin(pi * (y - 1/2)) * z * (z - 1)

    def _u3_(self, t, x, y, z):
        return zeros_like(x)

    @property
    def u_initial_condition(self):
        if self._u_init_ is None:
            self._u_init_ = T3dVector(self._u1_, self._u2_, self._u3_)
        return self._u_init_

    @property
    def w_initial_condition(self):
        if self._w_init_ is None:
            self._w_init_ = self.u_initial_condition.curl
        return self._w_init_

    def _B0_(self, t, x, y, z):
        """"""
        return - sin(pi * x) * cos(pi * y)

    def _B1_(self, t, x, y, z):
        return cos(pi * x) * sin(pi * y)

    @property
    def B_initial_condition(self):
        if self._B_init_ is None:
            self._B_init_ = T3dVector(self._B0_, self._B1_, 0)
        return self._B_init_

    @property
    def j_initial_condition(self):
        if self._j_init_ is None:
            self._j_init_ = self.B_initial_condition.curl
        return self._j_init_

    @property
    def g(self):
        return T3dScalar(0)

    @property
    def f(self):
        return T3dVector(0, 0, 0)

    @property
    def m(self):
        return T3dVector(0, 0, 0)


if __name__ == '__main__':
    condition = MHD3_Helicity_Conservation_test1()

    c = 1

    from scipy.integrate import nquad

    u = condition.u_initial_condition
    B = condition.B_initial_condition

    us = u.dot(u)
    Bs = B.dot(B)

    from functools import partial

    us = partial(us._s_, 0)

    u_energy = 0.5 * nquad(us, [[0, 1], [0, 1], [0, 1]])[0]
    # print(u_energy)

    Bs = partial(Bs._s_, 0)

    B_energy = 0.5 * c * nquad(Bs, [[0, 1], [0, 1], [0, 1]])[0]
    # print(B_energy)

    print(u_energy + B_energy)
