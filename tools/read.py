# -*- coding: utf-8 -*-
"""
"""
import pickle
from src.config import SIZE


def read(filename):
    """Read from objects."""
    assert SIZE == 1, f"ph.read works for COMM.SIZE == 1. Now it is {SIZE}."
    with open(filename, 'rb') as inputs:
        objs = pickle.load(inputs)
    inputs.close()
    return objs
