# -*- coding: utf-8 -*-
r"""

"""
import functools


class memoize2(object):
    """ Cache the return value of a method in class.

    This class is meant to be used as a decorator of methods. The return value from
    a given method invocation will be cached on the instance whose method was
    invoked. All arguments passed to a method decorated with memoize must be
    hashable.

    If a memoized method is invoked directly on its class the result will not be
    cached. Instead of the method will be invoked like a static method:
    ::

       class Obj(object):
           @memoize2
           def add_to(self, arg):
               return self + arg

    ``Obj.add_to(1)`` # not enough arguments; ``Obj.add_to(1, 2)`` # returns 3, result is not cached

    Script borrowed from here:
    MIT Licensed, attributed to Daniel Miller, Wed, 3 Nov 2010
    `active-state <http://code.activestate.com/recipes/577452-a-memoize-decorator-for-instance-methods/>`_

    - ``-``: Can not be used for frozen object. To use it for frozen object, use the method once before
        freeze the object.
    - ``-``: Can not be used for numpy.ndarray inputs.
    - ``+``: faster than memoize1.

    - ``-``: can not be dumped when used together with @accepts for one method.
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return functools.partial(self, obj)

    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res
