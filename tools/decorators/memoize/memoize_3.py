# -*- coding: utf-8 -*-
r"""

"""


def memoize3(f):
    """ Memoization decorator for a function taking a single argument.

    - ``+``: Very fast.
    - ``-``: Can not be used for methods in classes.
    - ``-``: for single input functions.
    - ``-``: Can not be used for numpy.ndarray inputs.
    """

    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret

    return memodict().__getitem__
