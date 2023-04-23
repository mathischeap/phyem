# -*- coding: utf-8 -*-


def memoize4(f):
    """ Memoization decorator for a function taking one or more arguments.

    - ``+``: Very fast.
    - ``+``: Can be used for multiple inputs functions.
    - ``-``: Can not be used for methods in classes.
    - ``-``: Can not be used for numpy.ndarray inputs.
    """

    class memodict(dict):
        def __getitem__(self, *key):
            return dict.__getitem__(self, key)

        def __missing__(self, key):
            ret = self[key] = f(*key)
            return ret

    return memodict().__getitem__
