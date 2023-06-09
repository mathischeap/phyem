# -*- coding: utf-8 -*-
r"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 7:02 PM on 5/2/2023

# python tests/unittests/msepy/main.py
"""
import os
import sys

if './' not in sys.path:
    sys.path.append('./')

from src.config import SIZE

assert SIZE == 1, f"msepy does not work with multiple ranks."

__all__ = [
    "_1",
    "_2",
    "_3",
]

import tests.unittests.msepy.m1n1 as _1

import tests.unittests.msepy.m2n2 as _2

import tests.unittests.msepy.m3n3 as _3

msepy_path = r'.\tests\unittests\msepy'
codifferential_tests = [
    rf'python {msepy_path}\codifferential_test.py 1 1 outer',
    rf'python {msepy_path}\codifferential_test.py 1 1 inner',
    rf'python {msepy_path}\codifferential_test.py 2 2 inner',
    rf'python {msepy_path}\codifferential_test.py 2 2 outer',
    rf'python {msepy_path}\codifferential_test.py 2 1 inner',
    rf'python {msepy_path}\codifferential_test.py 2 1 outer',
    rf'python {msepy_path}\codifferential_test.py 3 3 outer',
    rf'python {msepy_path}\codifferential_test.py 3 2 outer',
    rf'python {msepy_path}\codifferential_test.py 3 1 outer',
    rf'python {msepy_path}\codifferential_test.py 3 3 inner',
    rf'python {msepy_path}\codifferential_test.py 3 2 inner',
    rf'python {msepy_path}\codifferential_test.py 3 1 inner',
]

for _ in codifferential_tests:
    print(os.popen(_).read())


from tests.unittests.msepy.div_grad._2d_outer_periodic import div_grad_2d_periodic_manufactured_test

errors = div_grad_2d_periodic_manufactured_test(3, 4)
assert all([_ < 0.01 for _ in errors]), f"div_grad_2d_periodic_manufactured_test!"


if __name__ == '__main__':
    # python tests/unittests/msepy/main.py
    pass
