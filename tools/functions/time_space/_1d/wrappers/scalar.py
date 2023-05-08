

import sys

if './' not in sys.path:
    sys.path.append('/')

from tools.frozen import Frozen
from functools import partial
from tools.numerical.time_space._1d.partial_derivative_as_functions import \
    NumericalPartialDerivativeTxFunctions


class T1dScalar(Frozen):
    """"""

    def __init__(self, s):
        """"""
        self._s_ = s
        self.__NPD__ = None
        self._freeze()

    def __call__(self, t, x):
        return [self._s_(t, x),]

    def __getitem__(self, t):
        """return functions evaluated at time `t`."""
        return partial(self, t)

    @property
    def _NPD_(self):
        """"""
        if self.__NPD__ is None:
            self.__NPD__ = NumericalPartialDerivativeTxFunctions(self._s_)
        return self.__NPD__

    @property
    def time_derivative(self):
        """"""
        ps_pt = self._NPD_('t')
        return self.__class__(ps_pt)
        # return findiff(1, )

    @property
    def derivative(self):
        """"""
        ps_px = self._NPD_('x')
        return self.__class__(ps_px)


if __name__ == '__main__':
    # python ./tools/functions/_1d/wrappers/scalar.py
    def f(t, x):
        return x + 2 * t

    import numpy as np

    s = T1dScalar(f)
    print(s.time_derivative(1, np.array([1, 2, 3])))
