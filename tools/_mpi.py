# -*- coding: utf-8 -*-
r"""
"""
from src.config import MASTER_RANK, RANK, COMM


def merge_dict(*dictionaries, root=MASTER_RANK):
    """"""
    merged_dictionaries = list()
    for d in dictionaries:
        D = COMM.gather(d, root=root)
        if RANK == root:
            temp = dict()
            for D_rank in D:
                temp.update(D_rank)
            D = temp
        else:
            pass

        merged_dictionaries.append(D)

        COMM.barrier()

    return merged_dictionaries
