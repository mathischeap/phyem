# -*- coding: utf-8 -*-


def __det__(a, b):
    return a[0] * b[1] - a[1] * b[0]


def find_line_intersection(a1, a2, b1, b2):
    """

    :param a1:
    :param a2:
    :param b1:
    :param b2:
    :return:

    .. doctest::

        >>> a1, a2, b1, b2 = (0,0), (1,0), (0,1), (1,1)
        >>> find_line_intersection(a1, a2, b1, b2)
        >>> a1, a2, b1, b2 = (0,0), (1,1), (1,0), (0,1)
        >>> find_line_intersection(a1, a2, b1, b2)
        (0.5, 0.5)
        >>> a1, a2, b1, b2 = (0,0), (1,0), (0,1), (1,2)
        >>> find_line_intersection(a1, a2, b1, b2)
        (-1.0, 0.0)
        >>> find_line_intersection(a1, a2, b2, b1)
        (-1.0, -0.0)
    """
    line1, line2 = (a1, a2), (b1, b2)
    xdiff = (a1[0] - a2[0], b1[0] - b2[0])
    ydiff = (a1[1] - a2[1], b1[1] - b2[1])

    div = __det__(xdiff, ydiff)
    if div == 0:
        return None

    d = (__det__(*line1), __det__(*line2))
    x = __det__(d, xdiff) / div
    y = __det__(d, ydiff) / div
    return x, y


if __name__ == "__main__":
    import doctest
    doctest.testmod()
