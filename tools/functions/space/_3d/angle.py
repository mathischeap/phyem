# -*- coding: utf-8 -*-
"""3D functions."""
import numpy as np


def angle_between_two_vectors(v1, v2):
    """Compute the angle between the angle between v1 and v2 (can be
    of any dimension).

    :param v1:
    :param v2:
    :type v1: tuple, list, `np.array`
    :type v2: tuple, list, `np.array`
    :return: a float between [0, pi]
    """
    dot_product = sum([a*b for a, b in zip(v1, v2)])
    len_v1 = np.sqrt(np.sum([a**2 for a in v1]))
    len_v2 = np.sqrt(np.sum([a**2 for a in v2]))

    _ = dot_product / (len_v1 * len_v2)

    if _ > 1:
        _ = 1
    elif _ < -1:
        _ = -1
    else:
        pass

    return np.arccos(_)
