# -*- coding: utf-8 -*-
r"""
"""

from src.config import RANK, MASTER_RANK, COMM, SIZE
import numpy as np
from msehtt.tools.gathering_matrix import MseHttGatheringMatrix

from src.spaces.main import _degree_str_maker
_cache_ = {}


def gathering_matrix_Lambda__m2n2k2(tpm, degree):
    """Do the numbering for the 0-form on a 2d mesh in 2d space."""
    key = tpm.__repr__() + _degree_str_maker(degree)
    if key in _cache_:
        return _cache_[key]

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
                if etype in (
                        5,
                        9,
                        'unique msepy curvilinear quadrilateral',
                        'orthogonal rectangle',
                        "unique msepy curvilinear triangle",
                        'unique curvilinear quad',
                ):
                    global_numbering[e], current = ___gm222_msepy_quadrilateral___(
                        current, degree,
                    )
                else:
                    raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
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
    _cache_[key] = MseHttGatheringMatrix(rank_numbering)
    return _cache_[key]


from msehtt.static.mesh.great.elements.types.orthogonal_rectangle import MseHttGreatMeshOrthogonalRectangleElement


def ___gm222_msepy_quadrilateral___(current, degree):
    """"""
    p, _ = MseHttGreatMeshOrthogonalRectangleElement.degree_parser(degree)
    px, py = p
    numbering = np.arange(current, current + px * py)
    current += px * py
    return numbering, current
