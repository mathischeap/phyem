# -*- coding: utf-8 -*-
r"""

"""
import functools


def memoize1(func):
    """ Generally speaking, the best one is this one.

    - ``+``: Can be used for frozen object.
    - ``+``: Can be used for numpy.ndarray inputs. But normally when we can have numpy.ndarray as input,
        we do not use @memoize because storing the input may need a lot of memory.
    - ``-``: it is relatively slower than _auxiliaries.
    - ``-``: kwargs seem not to cached in keys at all.
    """
    cache = func.cache = dict()

    @functools.wraps(func)
    def memoized_func(*args, **kwargs):
        key = str(args) + str(kwargs)
        try:
            return cache[key]
        except KeyError:
            cache[key] = func(*args, **kwargs)
            return cache[key]

    return memoized_func
