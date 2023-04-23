# -*- coding: utf-8 -*-
"""A collection of all local decorators.

decorators return new functions, therefore will change .__code__ and so on.

@author: Yi Zhang (collecting). Created on Thu Feb 22 22:37:03 2018
         Department of Aerodynamics
         Faculty of Aerospace Engineering
         TU Delft,
         Delft, Netherlands

"""
import functools
lru_cache = functools.lru_cache
from tools.decorators.timeit.timeit_1 import timeit1
from tools.decorators.timeit.timeit_2 import timeit2

from tools.decorators.memoize.memoize_1 import memoize1
from tools.decorators.memoize.memoize_2 import memoize2
from tools.decorators.memoize.memoize_3 import memoize3
from tools.decorators.memoize.memoize_4 import memoize4
from tools.decorators.memoize.memoize_5 import memoize5

from tools.decorators.accepts import accepts


if __name__ == "__main__":
    # do some tests
    @timeit2
    def network_call(user_id):
        print("(computed)")
        return user_id


    class NetworkEngine(object):
        def __init__(self):
            pass

        @lru_cache()  # test for #1,2
        def search(self, user_id):
            return network_call(user_id)

        @staticmethod
        def test_tt(user_id):
            return network_call(user_id)


    e = NetworkEngine()
    for v in [1, 2, 3, 3, 3, 1, 4, 1, 5, 'abs', 'abs', 'abs']:
        print(e.search(v))


    @accepts('positive_int', (list, float, list, 'ndim=1', 'shape=(3)'))
    def foo(a, b):
        return a, b


    foo(1, [1, 2, 3])


    @memoize4
    @timeit1
    def foo1(a, b, c=-2):
        print('computed')
        return a * b * c

    print(foo1(3, 5))
    print(foo1(4, 5))
    print(foo1(5, 5))
    print(foo1(5, 5))
    print(foo1(4, 5))

    ms = memoize1, memoize2, memoize3, memoize4


    @memoize1
    def fun2(a):
        print(a)


    @memoize5
    def fun5(b):
        print(b)
