# -*- coding: utf-8 -*-
r"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 7:02 PM on 5/2/2023

# python tests/unittests/msepy/main.py
"""

import sys

if './' not in sys.path:
    sys.path.append('./')

__all__ = [
    'm1n1',
]

from src.config import SIZE

assert SIZE == 1, f"msepy does not work with multiple ranks."

import tests.unittests.msepy.m1n1 as m1n1


if __name__ == '__main__':
    # python tests/unittests/msepy/main.py
    pass
