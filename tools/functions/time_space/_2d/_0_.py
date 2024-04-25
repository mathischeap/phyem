# -*- coding: utf-8 -*-
import numpy as np


def _0t_(t, x, y):
    """
    A function always returns ``0``.

    :param x:
    :param y:
    :return:

    .. doctest::

        >>> _0t_(100, 100, 1000)
        0.0
    """
    assert np.shape(x) == np.shape(y)
    return np.zeros_like(x) + 0 * t


if __name__ == "__main__":
    import doctest
    doctest.testmod()
