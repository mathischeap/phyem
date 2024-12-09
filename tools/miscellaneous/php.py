# -*- coding: utf-8 -*-
r"""
phyem print.
"""
from src.config import RANK, MASTER_RANK


def php(str, flush=True):
    r""""""
    if RANK == MASTER_RANK:
        print(str, flush=flush)
    else:
        pass
