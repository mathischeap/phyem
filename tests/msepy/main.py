# -*- coding: utf-8 -*-
r"""
python tests/msepy/main.py
"""
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


import tests.msepy.m1n1 as _1

import tests.msepy.m2n2 as _2

import tests.msepy.m3n3 as _3


from tests.msepy.codifferential_test import codifferential_test
codifferential_test(1, 1, 'outer')
codifferential_test(1, 1, 'inner')
codifferential_test(2, 1, 'outer')
codifferential_test(2, 1, 'inner')
codifferential_test(2, 2, 'outer')
codifferential_test(2, 2, 'inner')
codifferential_test(3, 1, 'outer')
codifferential_test(3, 1, 'inner')
codifferential_test(3, 2, 'outer')
codifferential_test(3, 2, 'inner')
codifferential_test(3, 3, 'outer')
codifferential_test(3, 3, 'inner')


from tests.msepy.div_grad._2d_outer_periodic import div_grad_2d_periodic_manufactured_test
from tests.msepy.div_grad._2d_outer import div_grad_2d_general_bc_manufactured_test


a, b = div_grad_2d_periodic_manufactured_test(3, 4)
errors = [a, b]
assert all([_ < 0.01 for _ in errors]), f"div_grad_2d_periodic_manufactured_test!"

a, b = div_grad_2d_general_bc_manufactured_test(3, 4)
assert a < 0.01, f"div_grad_2d_general_bc_manufactured_test!"
assert b < 0.06, f"div_grad_2d_general_bc_manufactured_test!"

print("\t\t <-> phyem tests for implementation msepy passed.\n")
