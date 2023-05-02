# -*- coding: utf-8 -*-
import numpy as np


def _0_(x, y):
    """
    A function always returns ``0``.

    :param x:
    :param y:
    :return:

    .. doctest::

        >>> _0_(100,1000)
        array(0.)
    """
    assert np.shape(x) == np.shape(y)
    return np.zeros(np.shape(x))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
