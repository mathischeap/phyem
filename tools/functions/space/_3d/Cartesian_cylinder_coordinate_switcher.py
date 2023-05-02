# -*- coding: utf-8 -*-
import numpy as np
from abc import ABC


class CartCylSwitcher(ABC):
    """A cylinder <-> Cartesian coordinate switcher."""
    @classmethod
    def cart2cyl(cls, x, y, z):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return rho, phi, z

    @classmethod
    def cyl2cart(cls, rho, phi, z):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y, z
