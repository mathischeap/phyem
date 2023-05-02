# -*- coding: utf-8 -*-
import numpy as np
from abc import ABC


class CartSphSwitcher(ABC):
    """A spherical <-> Cartesian coordinate switcher."""
    @classmethod
    def cart2sph(cls, x, y, z):
        hxy = np.hypot(x, y)
        r = np.hypot(hxy, z)
        el = np.arctan2(z, hxy)
        az = np.arctan2(y, x)
        return az, el, r

    @classmethod
    def sph2cart(cls, az, el, r):
        r_cos_theta = r * np.cos(el)
        x = r_cos_theta * np.cos(az)
        y = r_cos_theta * np.sin(az)
        z = r * np.sin(el)
        return x, y, z
