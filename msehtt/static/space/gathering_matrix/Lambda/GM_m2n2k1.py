# -*- coding: utf-8 -*-
"""
"""

from src.config import RANK, MASTER_RANK, COMM, SIZE
from numpy import arange, ones, concatenate
from msehtt.tools.gathering_matrix import MseHttGatheringMatrix

from src.spaces.main import _degree_str_maker
_cache_o_ = {}


def gathering_matrix_Lambda__m2n2k1_outer(tpm, degree):
    """Do the numbering for the outer 1-form on a 2d mesh in 2d space."""
    key = tpm.__repr__() + _degree_str_maker(degree)
    if key in _cache_o_:
        return _cache_o_[key]

    tgm = tpm._tgm
    # do the numbering in the master rank only
    if RANK == MASTER_RANK:
        global_map = tgm._global_element_map_dict
        global_type = tgm._global_element_type_dict
        element_distribution = tgm._element_distribution

        global_numbering = {}
        edge_numbering_pool = {}
        current = 0
        for e in global_map:
            etype = global_type[e]
            map_ = global_map[e]
            # --------- call the element class to do the particular numbering -----------
            if e in tpm.composition.global_element_range:
                if etype in ('unique msepy curvilinear quadrilateral', 'orthogonal rectangle'):
                    global_numbering[e], current = ___gm_outer_msepy_quadrilateral___(
                        map_, edge_numbering_pool, current, degree,
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
    _cache_o_[key] = MseHttGatheringMatrix(rank_numbering)
    return _cache_o_[key]


from msehtt.static.mesh.great.elements.types.orthogonal_rectangle import MseHttGreatMeshOrthogonalRectangleElement


def ___gm_outer_msepy_quadrilateral___(map_, edge_numbering_pool, current, degree):
    """"""
    p, _ = MseHttGreatMeshOrthogonalRectangleElement.degree_parser(degree)
    px, py = p

    numbering_dy = -ones((px+1, py), dtype=int)
    numbering_dx = -ones((px, py+1), dtype=int)

    # (0, 0) face ---------------------------------
    edge_nodes = (map_[0], map_[2])
    edge_nodes_reverse = (map_[2], map_[0])
    if edge_nodes in edge_numbering_pool:
        edge_numbering = edge_numbering_pool[edge_nodes]
    elif edge_nodes_reverse in edge_numbering_pool:
        edge_numbering = edge_numbering_pool[edge_nodes_reverse][::-1]
    else:
        edge_numbering = arange(current, current + py)
        current += py
        edge_numbering_pool[edge_nodes] = edge_numbering
    numbering_dy[0, :] = edge_numbering

    # internal dy edges ----------------------------
    numbering_dy[1:-1, :] = arange(current, current + (px-1)*py).reshape((px-1, py), order='F')
    current += (px - 1) * py

    # (0, 1) face ---------------------------------
    edge_nodes = (map_[1], map_[3])
    edge_nodes_reverse = (map_[3], map_[1])
    if edge_nodes in edge_numbering_pool:
        edge_numbering = edge_numbering_pool[edge_nodes]
    elif edge_nodes_reverse in edge_numbering_pool:
        edge_numbering = edge_numbering_pool[edge_nodes_reverse][::-1]
    else:
        edge_numbering = arange(current, current + py)
        current += py
        edge_numbering_pool[edge_nodes] = edge_numbering
    numbering_dy[-1, :] = edge_numbering

    # (1, 0) face ---------------------------------
    edge_nodes = (map_[0], map_[1])
    edge_nodes_reverse = (map_[1], map_[0])
    if edge_nodes in edge_numbering_pool:
        edge_numbering = edge_numbering_pool[edge_nodes]
    elif edge_nodes_reverse in edge_numbering_pool:
        edge_numbering = edge_numbering_pool[edge_nodes_reverse][::-1]
    else:
        edge_numbering = arange(current, current + px)
        current += px
        edge_numbering_pool[edge_nodes] = edge_numbering
    numbering_dx[:, 0] = edge_numbering

    # internal dx edges ----------------------------
    numbering_dx[:, 1:-1] = arange(current, current + px * (py-1)).reshape((px, py-1), order='F')
    current += px * (py-1)

    # (1, 1) face ---------------------------------
    edge_nodes = (map_[2], map_[3])
    edge_nodes_reverse = (map_[3], map_[2])
    if edge_nodes in edge_numbering_pool:
        edge_numbering = edge_numbering_pool[edge_nodes]
    elif edge_nodes_reverse in edge_numbering_pool:
        edge_numbering = edge_numbering_pool[edge_nodes_reverse][::-1]
    else:
        edge_numbering = arange(current, current + px)
        current += px
        edge_numbering_pool[edge_nodes] = edge_numbering
    numbering_dx[:, -1] = edge_numbering

    numbering = concatenate([numbering_dy.ravel('F'), numbering_dx.ravel('F')])
    assert -1 not in numbering, f'At least one dof is not numbered.'
    return numbering, current


_cache_i_ = {}


def gathering_matrix_Lambda__m2n2k1_inner(tpm, degree):
    """Do the numbering for the outer 1-form on a 2d mesh in 2d space."""
    key = tpm.__repr__() + _degree_str_maker(degree)
    if key in _cache_i_:
        return _cache_i_[key]

    tgm = tpm._tgm
    # do the numbering in the master rank only
    if RANK == MASTER_RANK:
        global_map = tgm._global_element_map_dict
        global_type = tgm._global_element_type_dict
        element_distribution = tgm._element_distribution

        global_numbering = {}
        edge_numbering_pool = {}
        current = 0
        for e in global_map:
            etype = global_type[e]
            map_ = global_map[e]
            # --------- call the element class to do the particular numbering -----------
            if e in tpm.composition.global_element_range:
                if etype in ('unique msepy curvilinear quadrilateral', 'orthogonal rectangle'):
                    global_numbering[e], current = ___gm_inner_msepy_quadrilateral___(
                        map_, edge_numbering_pool, current, degree,
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
    _cache_i_[key] = MseHttGatheringMatrix(rank_numbering)
    return _cache_i_[key]


def ___gm_inner_msepy_quadrilateral___(map_, edge_numbering_pool, current, degree):
    """"""
    p, _ = MseHttGreatMeshOrthogonalRectangleElement.degree_parser(degree)
    px, py = p

    numbering_dx = -ones((px, py+1), dtype=int)
    numbering_dy = -ones((px+1, py), dtype=int)

    # (1, 0) face ---------------------------------
    edge_nodes = (map_[0], map_[1])
    edge_nodes_reverse = (map_[1], map_[0])
    if edge_nodes in edge_numbering_pool:
        edge_numbering = edge_numbering_pool[edge_nodes]
    elif edge_nodes_reverse in edge_numbering_pool:
        edge_numbering = edge_numbering_pool[edge_nodes_reverse][::-1]
    else:
        edge_numbering = arange(current, current + px)
        current += px
        edge_numbering_pool[edge_nodes] = edge_numbering
    numbering_dx[:, 0] = edge_numbering

    # internal dx edges ----------------------------
    numbering_dx[:, 1:-1] = arange(current, current + px * (py-1)).reshape((px, py-1), order='F')
    current += px * (py-1)

    # (1, 1) face ---------------------------------
    edge_nodes = (map_[2], map_[3])
    edge_nodes_reverse = (map_[3], map_[2])
    if edge_nodes in edge_numbering_pool:
        edge_numbering = edge_numbering_pool[edge_nodes]
    elif edge_nodes_reverse in edge_numbering_pool:
        edge_numbering = edge_numbering_pool[edge_nodes_reverse][::-1]
    else:
        edge_numbering = arange(current, current + px)
        current += px
        edge_numbering_pool[edge_nodes] = edge_numbering
    numbering_dx[:, -1] = edge_numbering

    # (0, 0) face ---------------------------------
    edge_nodes = (map_[0], map_[2])
    edge_nodes_reverse = (map_[2], map_[0])
    if edge_nodes in edge_numbering_pool:
        edge_numbering = edge_numbering_pool[edge_nodes]
    elif edge_nodes_reverse in edge_numbering_pool:
        edge_numbering = edge_numbering_pool[edge_nodes_reverse][::-1]
    else:
        edge_numbering = arange(current, current + py)
        current += py
        edge_numbering_pool[edge_nodes] = edge_numbering
    numbering_dy[0, :] = edge_numbering

    # internal dy edges ----------------------------
    numbering_dy[1:-1, :] = arange(current, current + (px-1)*py).reshape((px-1, py), order='F')
    current += (px - 1) * py

    # (0, 1) face ---------------------------------
    edge_nodes = (map_[1], map_[3])
    edge_nodes_reverse = (map_[3], map_[1])
    if edge_nodes in edge_numbering_pool:
        edge_numbering = edge_numbering_pool[edge_nodes]
    elif edge_nodes_reverse in edge_numbering_pool:
        edge_numbering = edge_numbering_pool[edge_nodes_reverse][::-1]
    else:
        edge_numbering = arange(current, current + py)
        current += py
        edge_numbering_pool[edge_nodes] = edge_numbering
    numbering_dy[-1, :] = edge_numbering

    numbering = concatenate([numbering_dx.ravel('F'), numbering_dy.ravel('F')])
    assert -1 not in numbering, f'At least one dof is not numbered.'
    return numbering, current
