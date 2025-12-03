# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.src.config import RANK, MASTER_RANK, COMM, SIZE
from phyem.msehtt.tools.gathering_matrix import MseHttGatheringMatrix
from phyem.src.spaces.main import _degree_str_maker

_cache_ = {}


def gathering_matrix_Lambda__m3n3k3(tpm, degree):
    """Do the numbering for the 3-form on a 3d mesh in 3d space."""
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
                    11,                         # same to 'orthogonal hexahedron'
                    'orthogonal hexahedron',    # same to 11
                    "unique msepy curvilinear hexahedron",
                ):
                    global_numbering[e], current = ___gm333_msepy_quadrilateral___(
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


from phyem.msehtt.static.mesh.great.elements.types.orthogonal_hexahedron import MseHttGreatMeshOrthogonalHexahedronElement


def ___gm333_msepy_quadrilateral___(current, degree):
    """"""
    p, _ = MseHttGreatMeshOrthogonalHexahedronElement.degree_parser(degree)
    px, py, pz = p
    numbering = np.arange(current, current + px * py * pz)
    current += px * py * pz
    return numbering, current
