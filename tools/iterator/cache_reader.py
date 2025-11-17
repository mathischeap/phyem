# -*- coding: utf-8 -*-
r"""

"""
import pickle
import os

from phyem.src.config import RANK, MASTER_RANK


def cache_reader(filename):
    r""""""
    if '.' not in filename:
        filename += '.phc'
    else:
        pass
    assert filename[-4:] == '.phc', f"I need a file with .phc as extension."
    assert os.path.isfile(filename), f"{filename} does not exist."

    with open(filename, 'rb') as f:
        cache = pickle.load(f)
    f.close()
    return cache


def print_cache_log(filename):
    r""""""
    if RANK != MASTER_RANK:
        return
    else:
        cache = cache_reader(filename)
        if 'log' in cache:
            log = cache['log']
        else:
            log = ''
        print(log, flush=True)
