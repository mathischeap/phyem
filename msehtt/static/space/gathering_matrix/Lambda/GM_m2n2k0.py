# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.src.config import RANK, MASTER_RANK, COMM, SIZE
from phyem.msehtt.tools.gathering_matrix import MseHttGatheringMatrix
from phyem.src.spaces.main import _degree_str_maker

_cache_ = {}


def gathering_matrix_Lambda__m2n2k0(tpm, degree):
    """Do the numbering for the 0-form on a 2d mesh in 2d space."""
    key = tpm.__repr__() + _degree_str_maker(degree)
    if key in _cache_:
        return _cache_[key]

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
                if etype in (
                        'unique msepy curvilinear quadrilateral',
                        'orthogonal rectangle',
                ):
                    global_numbering[e], current = ___gm220_msepy_quadrilateral___(
                        map_, edge_numbering_pool, node_numbering_pool, current, degree,
                    )

                elif etype in (
                        5,
                        "unique curvilinear triangle",
                        "unique msepy curvilinear triangle",
                ):
                    global_numbering[e], current = ___gm220_vtu_5___(
                        map_, edge_numbering_pool, node_numbering_pool, current, degree,
                    )

                elif etype in (
                        9,
                        'unique curvilinear quad',
                ):
                    global_numbering[e], current = ___gm220_quad_9___(
                        map_, edge_numbering_pool, node_numbering_pool, current, degree,
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


from phyem.msehtt.static.mesh.great.elements.types.orthogonal_rectangle import MseHttGreatMeshOrthogonalRectangleElement


def ___gm220_msepy_quadrilateral___(map_, edge_numbering_pool, node_numbering_pool, current, degree):
    """"""
    p, _ = MseHttGreatMeshOrthogonalRectangleElement.degree_parser(degree)
    px, py = p

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
        number = number[::-1]
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
        number = number[::-1]
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
        number = number[::-1]
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
        number = number[::-1]
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


from phyem.msehtt.static.mesh.great.elements.types.vtu_9_quad import Vtu9Quad


def ___gm220_quad_9___(map_, edge_numbering_pool, node_numbering_pool, current, degree):
    """"""
    p, _ = Vtu9Quad.degree_parser(degree)
    px, py = p

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

    # y- edge : West -----------------------
    edge__ = (map_[0], map_[1])
    edge_r = (map_[1], map_[0])
    if edge__ in edge_numbering_pool:
        number = edge_numbering_pool[edge__]
    elif edge_r in edge_numbering_pool:
        number = edge_numbering_pool[edge_r]
        number = number[::-1]
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

    # x- edge : North face -----------------------
    edge__ = (map_[0], map_[3])
    edge_r = (map_[3], map_[0])
    if edge__ in edge_numbering_pool:
        number = edge_numbering_pool[edge__]
    elif edge_r in edge_numbering_pool:
        number = edge_numbering_pool[edge_r]
        number = number[::-1]
    else:
        number = np.arange(current, current + py - 1)
        current += py - 1
        edge_numbering_pool[edge__] = number
    numbering[0, 1:-1] = number

    # internal nodes --------------------
    number = np.arange(current, current + (px-1)*(py-1)).reshape(((px-1), (py-1)), order='F')
    current += (px - 1) * (py - 1)
    numbering[1:-1, 1:-1] = number

    # x+ edge: South face -----------------------
    edge__ = (map_[1], map_[2])
    edge_r = (map_[2], map_[1])
    if edge__ in edge_numbering_pool:
        number = edge_numbering_pool[edge__]
    elif edge_r in edge_numbering_pool:
        number = edge_numbering_pool[edge_r]
        number = number[::-1]
    else:
        number = np.arange(current, current + py - 1)
        current += py - 1
        edge_numbering_pool[edge__] = number
    numbering[-1, 1:-1] = number

    # (x-, y+) corner ---------------------------------
    corner = map_[3]
    if corner in node_numbering_pool:
        number = node_numbering_pool[corner]
    else:
        number = current
        current += 1
        node_numbering_pool[corner] = number
    numbering[0, -1] = number

    # y+ edge : East -----------------------
    edge__ = (map_[3], map_[2])
    edge_r = (map_[2], map_[3])
    if edge__ in edge_numbering_pool:
        number = edge_numbering_pool[edge__]
    elif edge_r in edge_numbering_pool:
        number = edge_numbering_pool[edge_r]
        number = number[::-1]
    else:
        number = np.arange(current, current + px - 1)
        current += px - 1
        edge_numbering_pool[edge__] = number
    numbering[1:-1, -1] = number

    # (x+, y+) corner ---------------------------------
    corner = map_[2]
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


from phyem.msehtt.static.mesh.great.elements.types.vtu_5_triangle import Vtu5Triangle


def ___gm220_vtu_5___(map_, edge_numbering_pool, node_numbering_pool, current, degree):
    """
    -----------------------> et
    |
    |     0         0         0
    |     ---------------------
    |     |         |         |
    |   1 -----------3--------- 5
    |     |         |         |
    |   2 -----------4--------- 6
    |
    v

    xi

    Thus, the north edge is mapped into a node.

    Parameters
    ----------
    map_
    edge_numbering_pool
    node_numbering_pool
    current
    degree

    Returns
    -------

    """
    p, _ = Vtu5Triangle.degree_parser(degree)
    px, py = p
    assert px == py, f"must be px == py for triangle element."

    numbering = -np.ones((px+1, py+1), dtype=int)

    # node 0 ---------------------------------
    corner = map_[0]
    if corner in node_numbering_pool:
        number = node_numbering_pool[corner]
    else:
        number = current
        current += 1
        node_numbering_pool[corner] = number
    numbering[0, :] = number

    # edge 0 -----------------------
    edge__ = (map_[0], map_[1])
    edge_r = (map_[1], map_[0])
    if edge__ in edge_numbering_pool:
        number = edge_numbering_pool[edge__]
    elif edge_r in edge_numbering_pool:
        number = edge_numbering_pool[edge_r]
        number = number[::-1]
    else:
        number = np.arange(current, current + px - 1)
        current += px - 1
        edge_numbering_pool[edge__] = number
    numbering[1:-1, 0] = number

    # node 1 ---------------------------------
    corner = map_[1]
    if corner in node_numbering_pool:
        number = node_numbering_pool[corner]
    else:
        number = current
        current += 1
        node_numbering_pool[corner] = number
    numbering[-1, 0] = number

    # internal nodes --------------------
    number = np.arange(current, current + (px-1)*(py-1)).reshape(((px-1), (py-1)), order='F')
    current += (px - 1) * (py - 1)
    numbering[1:-1, 1:-1] = number

    # edge 1 -----------------------
    edge__ = (map_[1], map_[2])
    edge_r = (map_[2], map_[1])
    if edge__ in edge_numbering_pool:
        number = edge_numbering_pool[edge__]
    elif edge_r in edge_numbering_pool:
        number = edge_numbering_pool[edge_r]
        number = number[::-1]
    else:
        number = np.arange(current, current + py - 1)
        current += py - 1
        edge_numbering_pool[edge__] = number
    numbering[-1, 1:-1] = number

    # edge 2 -----------------------
    edge__ = (map_[0], map_[2])
    edge_r = (map_[2], map_[0])
    if edge__ in edge_numbering_pool:
        number = edge_numbering_pool[edge__]
    elif edge_r in edge_numbering_pool:
        number = edge_numbering_pool[edge_r]
        number = number[::-1]
    else:
        number = np.arange(current, current + px - 1)
        current += px - 1
        edge_numbering_pool[edge__] = number
    numbering[1:-1, -1] = number

    # node 2 ---------------------------------
    corner = map_[2]
    if corner in node_numbering_pool:
        number = node_numbering_pool[corner]
    else:
        number = current
        current += 1
        node_numbering_pool[corner] = number
    numbering[-1, -1] = number

    numbering = np.concatenate(
        [np.array([numbering[0, 0], ]), numbering[1:, :].ravel('F')]
    )
    assert -1 not in numbering, f"at least one dof is not numbered."
    return numbering, current
