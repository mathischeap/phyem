# -*- coding: utf-8 -*-
"""
"""

from src.config import RANK, MASTER_RANK, COMM, SIZE
import numpy as np


def gathering_matrix_Lambda__m2n2k2(tpm, degree):
    """Do the numbering for the 0-form on a 2d mesh in 2d space."""
    tgm = tpm._tgm
    # do the numbering in the master rank only
    if RANK == MASTER_RANK:
        global_type = tgm._global_element_type_dict
        element_distribution = tgm._element_distribution
        global_numbering = {}
        current = 0
        for e in global_type:
            etype = global_type[e]
            # --------- call the element class to do the particular numbering -----------
            if e in tpm.composition.global_element_range:
                if etype in ('unique msepy curvilinear quadrilateral', 'orthogonal rectangle'):
                    global_numbering[e], current = ___gm222_msepy_quadrilateral___(
                        current, degree,
                    )
                else:
                    raise NotImplementedError()
            else:
                global_numbering[e] = None

        # --------- split numbering ----------------------------------
        rank_numbering = list()
        for rank in range(SIZE):
            rank_element_indices = element_distribution[rank]
            numbering = {}
            for e in rank_element_indices:
                numbering[e] = global_numbering[e]
            rank_numbering.append(numbering)

    else:
        rank_numbering = None

    # distribute to ranks ----------------
    rank_numbering = COMM.scatter(rank_numbering, root=MASTER_RANK)
    return rank_numbering


def ___gm222_msepy_quadrilateral___(current, degree):
    """"""
    if isinstance(degree, int):
        px, py = degree, degree
    else:
        raise NotImplementedError(f"cannot parse degree={degree} for 2d msepy quadrilateral element.")
    numbering = np.arange(current, current + px * py)
    current += px * py
    return numbering, current
