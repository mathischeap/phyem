# -*- coding: utf-8 -*-
r"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""
import sys

if './' not in sys.path:
    sys.path.append('./')

from tools.frozen import Frozen
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class RandomRefiningStrengthFunction(Frozen):

    def __init__(self, bounds, factor=6):
        """Make a random rsf in domain = ([xl, xu], [yl, yu]).

        And bounds = ([xl, xu], [yl, yu])

        Parameters
        ----------
        bounds

        Returns
        -------

        """
        x_bounds, y_bounds = bounds
        x_lower, x_upper = x_bounds
        y_lower, y_upper = y_bounds
        assert x_lower < x_upper and y_lower < y_upper
        a = np.random.rand(factor**2) * 2 - 1
        a = a.reshape((factor, factor))
        x = y = np.linspace(-1, 1, factor)

        self._interp = RegularGridInterpolator((x, y), a)

    def __call__(self, x, y):
        """"""
        return self._interp((x, y))


if __name__ == '__main__':
    # python msehy/py2/tools/random_refining_strength_function.py
    f = RandomRefiningStrengthFunction(([0, 1], [0, 1]))
    print(f(np.array([0,1,1]), np.array([0,1,1])))
