# -*- coding: utf-8 -*-
r"""
A collection sphinx doctests functions.
"""

__all__ = [
    "div_grad_2d_periodic_manufactured_test",
    "canonical_linear_pH_3d_periodic_manufactured_test",
]

from tests.msepy.div_grad._2d_outer_periodic import div_grad_2d_periodic_manufactured_test
from tests.msepy.canonical_linear_pH._3d import canonical_linear_pH_3d_periodic_manufactured_test


if __name__ == '__main__':
    # python _sdt.py
    import doctest
    doctest.testmod()
