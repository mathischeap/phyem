# -*- coding: utf-8 -*-
"""2D functions."""
import numpy as np


def angle(origin, pt):
    """
    compute angle between the vector from origin to pt and the x-direction vector.

    For example:

    .. doctest::

        >>> angle((0,0), (1,1)) # should return pi/4
        0.7853981633974484
    """
    x1, y1 = (1, 0)
    x2, y2 = (pt[0] - origin[0], pt[1] - origin[1])
    inner_product = x1 * x2 + y1 * y2
    len1 = np.hypot(x1, y1)
    len2 = np.hypot(x2, y2)
    if y2 < 0:
        return float(2 * np.pi - np.arccos(inner_product / (len1 * len2)))
    else:
        return float(np.arccos(inner_product / (len1 * len2)))


if __name__ == "__main__":

    print(
        angle((0, 0), (1, 0)),
        angle((0, 0), (0, 1)),
        angle((0, 0), (-1, 0)),
        angle((0, 0), (0, -1)),
    )

    import doctest
    doctest.testmod()