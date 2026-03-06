r""""""

from phyem.tools.frozen import Frozen
from phyem.msehtt.tools.linear_system.static.global_.solvers.scipy_ import spsolve


class General_Global_Linear_System(Frozen):
    r""""""
    def __init__(self, A, b):
        r""""""
        self._A = A
        self._b = b
        self._freeze()

    def solve(self, scheme, x0=None, **kwargs):
        r""""""
        if scheme == 'direct':
            return spsolve(self._A, self._b, **kwargs)
        else:
            assert x0 is not None
            raise NotImplementedError()
