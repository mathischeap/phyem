# -*- coding: utf-8 -*-
r"""

"""
import numpy as np
from abc import ABC


class CartPolSwitcher(ABC):
    """
    A polar <-> Cartesian coordinates switcher.

    .. doctest::

        >>> CartPolSwitcher.cart2pol(2,2)
        (2.8284271247461903, 0.7853981633974483)
        >>> CartPolSwitcher.pol2cart(1, np.pi/4)
        (0.7071067811865476, 0.7071067811865476)
    """

    @classmethod
    def cart2pol(cls, x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return rho, phi

    @classmethod
    def pol2cart(cls, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y


if __name__ == "__main__":
    import doctest
    doctest.testmod()
