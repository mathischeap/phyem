# -*- coding: utf-8 -*-
r"""
phyem print.
"""
from phyem.src.config import RANK, MASTER_RANK


def php(*str_, flush=True):
    r"""Ph print: only to print the input from the master rank."""
    if RANK == MASTER_RANK:
        print(*str_, flush=flush)
    else:
        pass
