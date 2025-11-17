# -*- coding: utf-8 -*-
r"""
"""
from phyem.src.config import MASTER_RANK, RANK, COMM


def merge_dict(*dictionaries, root=MASTER_RANK):
    r""""""
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


def merge_list(*lists, root=MASTER_RANK):
    r""""""
    merged_lists = ()
    for L in lists:
        LIST = COMM.gather(L, root=root)
        if RANK == root:
            temp = list()
            for L_rank in LIST:
                temp.extend(L_rank)
            LIST = temp
        else:
            pass
        merged_lists += (LIST,)

        COMM.barrier()

    return merged_lists
