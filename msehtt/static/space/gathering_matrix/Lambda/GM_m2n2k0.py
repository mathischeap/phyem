# -*- coding: utf-8 -*-
"""
"""

from src.config import RANK, MASTER_RANK, COMM, SIZE
import numpy as np


def gathering_matrix_Lambda__m2n2k0(tpm, degree):
    """Do the numbering for the 0-form on a 2d mesh in 2d space."""
    tgm = tpm._tgm
    # do the numbering in the master rank only
    if RANK == MASTER_RANK:
        global_map = tgm._global_element_map_dict
        global_type = tgm._global_element_type_dict
        element_distribution = tgm._element_distribution

        global_numbering = {}
        edge_numbering_pool = {}
        node_numbering_pool = {}
        current = 0
        for e in global_map:
            etype = global_type[e]
            map_ = global_map[e]
            # --------- call the element class to do the particular numbering -----------
            if e in tpm.composition.global_element_range:
                if etype in ('unique msepy curvilinear quadrilateral', 'orthogonal rectangle'):
                    global_numbering[e], current = ___gm220_msepy_quadrilateral___(
                        map_, edge_numbering_pool, node_numbering_pool, current, degree,
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


def ___gm220_msepy_quadrilateral___(map_, edge_numbering_pool, node_numbering_pool, current, degree):
    """"""
    if isinstance(degree, int):
        px, py = degree, degree
    else:
        raise NotImplementedError(f"cannot parse degree={degree} for 2d msepy quadrilateral element.")
    numbering = -np.ones((px+1, py+1), dtype=int)

    # (x-, y-) corner ---------------------------------
    corner = map_[0]
    if corner in node_numbering_pool:
        number = node_numbering_pool[corner]
    else:
        number = current
        current += 1
        node_numbering_pool[corner] = number
    numbering[0, 0] = number

    # y- edge -----------------------
    edge__ = (map_[0], map_[1])
    edge_r = (map_[1], map_[0])
    if edge__ in edge_numbering_pool:
        number = edge_numbering_pool[edge__]
    elif edge_r in edge_numbering_pool:
        number = edge_numbering_pool[edge_r]
    else:
        number = np.arange(current, current + px - 1)
        current += px - 1
        edge_numbering_pool[edge__] = number
    numbering[1:-1, 0] = number

    # (x+, y-) corner ---------------------------------
    corner = map_[1]
    if corner in node_numbering_pool:
        number = node_numbering_pool[corner]
    else:
        number = current
        current += 1
        node_numbering_pool[corner] = number
    numbering[-1, 0] = number

    # x- edge -----------------------
    edge__ = (map_[0], map_[2])
    edge_r = (map_[2], map_[0])
    if edge__ in edge_numbering_pool:
        number = edge_numbering_pool[edge__]
    elif edge_r in edge_numbering_pool:
        number = edge_numbering_pool[edge_r]
    else:
        number = np.arange(current, current + py - 1)
        current += py - 1
        edge_numbering_pool[edge__] = number
    numbering[0, 1:-1] = number

    # internal nodes --------------------
    number = np.arange(current, current + (px-1)*(py-1)).reshape(((px-1), (py-1)), order='F')
    current += (px - 1) * (py - 1)
    numbering[1:-1, 1:-1] = number

    # x+ edge -----------------------
    edge__ = (map_[1], map_[3])
    edge_r = (map_[3], map_[1])
    if edge__ in edge_numbering_pool:
        number = edge_numbering_pool[edge__]
    elif edge_r in edge_numbering_pool:
        number = edge_numbering_pool[edge_r]
    else:
        number = np.arange(current, current + py - 1)
        current += py - 1
        edge_numbering_pool[edge__] = number
    numbering[-1, 1:-1] = number

    # (x-, y+) corner ---------------------------------
    corner = map_[2]
    if corner in node_numbering_pool:
        number = node_numbering_pool[corner]
    else:
        number = current
        current += 1
        node_numbering_pool[corner] = number
    numbering[0, -1] = number

    # y+ edge -----------------------
    edge__ = (map_[2], map_[3])
    edge_r = (map_[3], map_[2])
    if edge__ in edge_numbering_pool:
        number = edge_numbering_pool[edge__]
    elif edge_r in edge_numbering_pool:
        number = edge_numbering_pool[edge_r]
    else:
        number = np.arange(current, current + px - 1)
        current += px - 1
        edge_numbering_pool[edge__] = number
    numbering[1:-1, -1] = number

    # (x+, y+) corner ---------------------------------
    corner = map_[3]
    if corner in node_numbering_pool:
        number = node_numbering_pool[corner]
    else:
        number = current
        current += 1
        node_numbering_pool[corner] = number
    numbering[-1, -1] = number

    numbering = numbering.ravel('F')
    assert -1 not in numbering, f"at least one dof is not numbered."
    return numbering, current
