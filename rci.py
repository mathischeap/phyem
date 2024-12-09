# -*- coding: utf-8 -*-
r"""
python rci.py filename
"""
import sys
import __init__ as ph


def _rci(filename):
    ph.reveal_phc(filename)


if __name__ == '__main__':
    # python rci.py filename
    _rci(sys.argv[1])
