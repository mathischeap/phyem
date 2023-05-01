

import numpy as np
import sys
if './' not in sys.path:
    sys.path.append('./')

from tools.frozen import Frozen
from tools.matplot.plot import plot


class MsePyRootFormVisualizeMatplot(Frozen):
    """"""
    def __init__(self, rf):
        """"""
        self._f = rf
        self._mesh = rf.mesh
        self._freeze()

    def __call__(self, *args, **kwargs):
        """"""
        abs_sp = self._f.space.abstract
        m = abs_sp.m
        n = abs_sp.n
        k = abs_sp.k
        orientation = abs_sp.orientation

        if m == n == 2 and k == 1:
            return getattr(self, f'_m{m}_n{n}_k{k}_{orientation}')(*args, **kwargs)
        else:
            return getattr(self, f'_m{m}_n{n}_k{k}')(*args, **kwargs)

    def _m1_n1_k0(
            self, sampling_factor=1,
            figsize=(10, 6),
            color='k',
            **kwargs
    ):
        """"""
        samples = 500 * sampling_factor
        samples = int((np.ceil(samples / self._mesh.elements._num))**(1/self._mesh.m))
        if samples > 100:
            samples = 100
        elif samples < 5:
            samples = 5
        else:
            samples = int(samples)

        linspace = np.linspace(-1, 1, samples)
        t = self._f.visualize._t
        x, v = self._f.reconstruct[t](linspace)

        x = x[0].T
        v = v[0]
        num_lines = len(x)  # also num elements
        return plot(x, v, num_lines=num_lines, colors=color, xlabel='$x$', labels=False, styles='-',
             figsize=figsize, **kwargs)

    def _m1_n1_k1(self, *args, **kwargs):
        """"""
        return self._m1_n1_k0(*args, **kwargs)
