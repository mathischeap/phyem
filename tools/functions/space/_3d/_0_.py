# -*- coding: utf-8 -*-
import numpy as np


def _0_(x, y, z):
    assert np.shape(x) == np.shape(y) == np.shape(z)
    return np.zeros(np.shape(x))
