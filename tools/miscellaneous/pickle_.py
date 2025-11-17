# -*- coding: utf-8 -*-
r"""
A wrapper of pickle package in the phyem environment.
"""

import pickle

from phyem.src.config import RANK, MASTER_RANK, COMM


def ms(filename, obj):
    r"""Master save.

    This will only call pickle in the master rank and save what is in the master rank only.
    """
    if RANK != MASTER_RANK:
        return
    else:
        pass

    with open(filename, 'wb') as output:
        # noinspection PyTypeChecker
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    output.close()


def mr(filename):
    r"""master read. Read to the master rank only, return None in other ranks."""
    if RANK == MASTER_RANK:

        with open(filename, 'rb') as inputs:
            obj = pickle.load(inputs)
        inputs.close()

    else:
        obj = None

    return obj


def r(filename):
    r"""Read a file to all ranks."""
    if RANK == MASTER_RANK:

        with open(filename, 'rb') as inputs:
            obj = pickle.load(inputs)
        inputs.close()

    else:
        obj = None

    return COMM.bcast(obj, root=MASTER_RANK)
