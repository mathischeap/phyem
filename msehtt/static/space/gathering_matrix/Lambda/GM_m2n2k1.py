# -*- coding: utf-8 -*-
r"""
"""

from src.config import RANK, MASTER_RANK, COMM, SIZE
from numpy import arange, ones, concatenate
from msehtt.tools.gathering_matrix import MseHttGatheringMatrix
from msehtt.static.mesh.great.main import MseHttGreatMesh
from msehtt.static.mesh.partial.main import MseHttMeshPartial

from src.spaces.main import _degree_str_maker

from msehtt.static.mesh.great.elements.types.orthogonal_rectangle import MseHttGreatMeshOrthogonalRectangleElement
from msehtt.static.mesh.great.elements.types.vtu_5_triangle import Vtu5Triangle


# ----------------- outer m2n2 1-form --------------------------------------------------


_cache_o_ = {}


def gathering_matrix_Lambda__m2n2k1_outer(tpm_or_tgm, degree, do_cache=True):
    """Do the numbering for the outer 1-form on a 2d mesh in 2d space."""
    if do_cache:
        key = tpm_or_tgm.__repr__() + _degree_str_maker(degree)
        if key in _cache_o_:
            return _cache_o_[key]
    else:
        pass

    if tpm_or_tgm.__class__ is MseHttMeshPartial:
        tgm = tpm_or_tgm._tgm
        ELEMENT_RANGE = tpm_or_tgm.composition.global_element_range

    elif tpm_or_tgm.__class__ is MseHttGreatMesh:
        tgm = tpm_or_tgm
        if RANK == MASTER_RANK:
            ELEMENT_RANGE = tgm._global_element_map_dict.keys()
        else:
            ELEMENT_RANGE = {}
    else:
        raise NotImplementedError()

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
            if e in ELEMENT_RANGE:
                if etype in (
                        'unique msepy curvilinear quadrilateral',
                        'orthogonal rectangle',
                ):
                    global_numbering[e], current = ___gm_outer_msepy_quadrilateral___(
                        map_, edge_numbering_pool, current, degree,
                    )

                elif etype in (
                        5,
                        'unique curvilinear triangle',
                        'unique msepy curvilinear triangle',
                ):
                    global_numbering[e], current = ___gm_outer_vtu_5___(
                        map_, edge_numbering_pool, current, degree,
                    )

                elif etype in (
                        9,
                        'unique curvilinear quad',
                ):
                    global_numbering[e], current = ___gm_outer_quad_9___(
                        map_, edge_numbering_pool, current, degree,
                    )

                else:
                    raise NotImplementedError(f"{__name__} not implemented for etype={etype}")

            else:
                global_numbering[e] = None

        if tpm_or_tgm.__class__ is MseHttMeshPartial:
            # --------- split numbering ----------------------------------
            rank_numbering = list()
            for rank in range(SIZE):
                rank_element_indices = element_distribution[rank]
                numbering = {}
                for e in rank_element_indices:
                    numbering[e] = global_numbering[e]
                rank_numbering.append(numbering)
        else:
            pass

    else:
        rank_numbering = None

    if tpm_or_tgm.__class__ is MseHttMeshPartial:
        # distribute to ranks ----------------
        # noinspection PyUnboundLocalVariable
        rank_numbering = COMM.scatter(rank_numbering, root=MASTER_RANK)
        NUMBERING = MseHttGatheringMatrix(rank_numbering)

    elif tpm_or_tgm.__class__ is MseHttGreatMesh:
        if RANK == MASTER_RANK:
            # noinspection PyUnboundLocalVariable
            NUMBERING = MseHttGatheringMatrix(global_numbering)
        else:
            NUMBERING = MseHttGatheringMatrix({})
    else:
        raise Exception()

    if do_cache:
        # noinspection PyUnboundLocalVariable
        _cache_o_[key] = NUMBERING
    else:
        pass
    return NUMBERING


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


def ___gm_outer_quad_9___(map_, edge_numbering_pool, current, degree):
    """"""
    p, _ = MseHttGreatMeshOrthogonalRectangleElement.degree_parser(degree)
    px, py = p

    numbering_dy = -ones((px+1, py), dtype=int)
    numbering_dx = -ones((px, py+1), dtype=int)

    # (0, 0) face: North face ---------------------------------
    edge_nodes = (map_[0], map_[3])
    edge_nodes_reverse = (map_[3], map_[0])
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

    # (0, 1) face: South face ---------------------------------
    edge_nodes = (map_[1], map_[2])
    edge_nodes_reverse = (map_[2], map_[1])
    if edge_nodes in edge_numbering_pool:
        edge_numbering = edge_numbering_pool[edge_nodes]
    elif edge_nodes_reverse in edge_numbering_pool:
        edge_numbering = edge_numbering_pool[edge_nodes_reverse][::-1]
    else:
        edge_numbering = arange(current, current + py)
        current += py
        edge_numbering_pool[edge_nodes] = edge_numbering
    numbering_dy[-1, :] = edge_numbering

    # (1, 0) face : West face ---------------------------------
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

    # (1, 1) face : East face ---------------------------------
    edge_nodes = (map_[3], map_[2])
    edge_nodes_reverse = (map_[2], map_[3])
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


def ___gm_outer_vtu_5___(map_, edge_numbering_pool, current, degree):
    r"""
        ______________________> et
        |           0        the north edge is collapsed into node 0
        |          /\
        |         /  \                 >   edge 0: positive direction: 0->1
        | edge0  /    \ edge 2         >>  edge 1: positive direction: 1->2
        |       /      \               >>> edge 2: positive direction: 0->2
        |      /        \
        |     ------------
        v     1   edge1   2
        xi


        -----------------------> et
        |
        |
        |     ---------------------
        |     |         |         |
        |     |4        | 6       | 8
        |     |         |         |
        |     -----0----- ----2----
        |     |         |         |
        |     |5        | 7       | 9
        |     |         |         |
        |     -----1----------3----
        |
        v

        xi


    Parameters
    ----------
    map_
    edge_numbering_pool
    current
    degree

    Returns
    -------

    """
    p, _ = Vtu5Triangle.degree_parser(degree)
    px, py = p

    numbering_dy = -ones((px, py), dtype=int)
    numbering_dx = -ones((px, py+1), dtype=int)

    # internal dy edges ----------------------------
    numbering_dy[:-1, :] = arange(current, current + (px-1)*py).reshape((px-1, py), order='F')
    current += (px - 1) * py

    # South face: edge 1 ---------------------------------
    edge_nodes = (map_[1], map_[2])
    edge_nodes_reverse = (map_[2], map_[1])
    if edge_nodes in edge_numbering_pool:
        edge_numbering = edge_numbering_pool[edge_nodes]
    elif edge_nodes_reverse in edge_numbering_pool:
        edge_numbering = edge_numbering_pool[edge_nodes_reverse][::-1]
    else:
        edge_numbering = arange(current, current + py)
        current += py
        edge_numbering_pool[edge_nodes] = edge_numbering
    numbering_dy[-1, :] = edge_numbering

    # West face: edge 0 ---------------------------------
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

    # East face: edge 2 ---------------------------------
    edge_nodes = (map_[0], map_[2])
    edge_nodes_reverse = (map_[2], map_[0])
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


# ----------------- inner m2n2 1-form --------------------------------------------------


_cache_i_ = {}


def gathering_matrix_Lambda__m2n2k1_inner(tpm_or_tgm, degree, do_cache=True):
    """Do the numbering for the outer 1-form on a 2d mesh in 2d space."""
    if do_cache:
        key = tpm_or_tgm.__repr__() + _degree_str_maker(degree)
        if key in _cache_i_:
            return _cache_i_[key]
    else:
        pass

    if tpm_or_tgm.__class__ is MseHttMeshPartial:
        tgm = tpm_or_tgm._tgm
        ELEMENT_RANGE = tpm_or_tgm.composition.global_element_range

    elif tpm_or_tgm.__class__ is MseHttGreatMesh:
        tgm = tpm_or_tgm
        if RANK == MASTER_RANK:
            ELEMENT_RANGE = tgm._global_element_map_dict.keys()
        else:
            ELEMENT_RANGE = {}

    else:
        raise NotImplementedError()

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
            if e in ELEMENT_RANGE:
                if etype in (
                        'unique msepy curvilinear quadrilateral',
                        'orthogonal rectangle',
                ):
                    global_numbering[e], current = ___gm_inner_msepy_quadrilateral___(
                        map_, edge_numbering_pool, current, degree,
                    )

                elif etype in (
                        5,
                        'unique curvilinear triangle',
                        'unique msepy curvilinear triangle',
                ):
                    global_numbering[e], current = ___gm_inner_vtu_5___(
                        map_, edge_numbering_pool, current, degree,
                    )

                elif etype in (
                        9,
                        'unique curvilinear quad',
                ):
                    global_numbering[e], current = ___gm_inner_quad_9___(
                        map_, edge_numbering_pool, current, degree,
                    )

                else:
                    raise NotImplementedError(f"{__name__} not implemented for etype={etype}")

            else:
                global_numbering[e] = None

        if tpm_or_tgm.__class__ is MseHttMeshPartial:
            # --------- split numbering ----------------------------------
            rank_numbering = list()
            for rank in range(SIZE):
                rank_element_indices = element_distribution[rank]
                numbering = {}
                for e in rank_element_indices:
                    numbering[e] = global_numbering[e]
                rank_numbering.append(numbering)
        else:
            pass

    else:
        rank_numbering = None

    if tpm_or_tgm.__class__ is MseHttMeshPartial:
        # distribute to ranks ----------------
        # noinspection PyUnboundLocalVariable
        rank_numbering = COMM.scatter(rank_numbering, root=MASTER_RANK)
        NUMBERING = MseHttGatheringMatrix(rank_numbering)

    elif tpm_or_tgm.__class__ is MseHttGreatMesh:
        if RANK == MASTER_RANK:
            # noinspection PyUnboundLocalVariable
            NUMBERING = MseHttGatheringMatrix(global_numbering)
        else:
            NUMBERING = MseHttGatheringMatrix({})
    else:
        raise Exception()

    if do_cache:
        # noinspection PyUnboundLocalVariable
        _cache_i_[key] = NUMBERING
    else:
        pass
    return NUMBERING


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


def ___gm_inner_quad_9___(map_, edge_numbering_pool, current, degree):
    """"""
    p, _ = MseHttGreatMeshOrthogonalRectangleElement.degree_parser(degree)
    px, py = p

    numbering_dx = -ones((px, py+1), dtype=int)
    numbering_dy = -ones((px+1, py), dtype=int)

    # (1, 0) face: West face ---------------------------------
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

    # (1, 1) face: East face---------------------------------
    edge_nodes = (map_[3], map_[2])
    edge_nodes_reverse = (map_[2], map_[3])
    if edge_nodes in edge_numbering_pool:
        edge_numbering = edge_numbering_pool[edge_nodes]
    elif edge_nodes_reverse in edge_numbering_pool:
        edge_numbering = edge_numbering_pool[edge_nodes_reverse][::-1]
    else:
        edge_numbering = arange(current, current + px)
        current += px
        edge_numbering_pool[edge_nodes] = edge_numbering
    numbering_dx[:, -1] = edge_numbering

    # (0, 0) face : North face ---------------------------------
    edge_nodes = (map_[0], map_[3])
    edge_nodes_reverse = (map_[3], map_[0])
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

    # (0, 1) face : South---------------------------------
    edge_nodes = (map_[1], map_[2])
    edge_nodes_reverse = (map_[2], map_[1])
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


def ___gm_inner_vtu_5___(map_, edge_numbering_pool, current, degree):
    r"""

        ______________________> et
        |           0        the north edge is collapsed into node 0
        |          /\
        |         /  \                 >   edge 0: positive direction: 0->1
        | edge0  /    \ edge 2         >>  edge 1: positive direction: 1->2
        |       /      \               >>> edge 2: positive direction: 0->2
        |      /        \
        |     ------------
        v     1   edge1   2
         xi


        -----------------------> et
        |
        |
        |     ---------------------
        |     |         |         |
        |     |0        | 2       | 4
        |     |         |         |
        |     -----6----- ----8----
        |     |         |         |
        |     |1        | 3       | 5
        |     |         |         |
        |     -----7----------9----
        |
        v
         xi

    Parameters
    ----------
    map_
    edge_numbering_pool
    current
    degree

    Returns
    -------

    """
    p, _ = Vtu5Triangle.degree_parser(degree)
    px, py = p

    numbering_dx = -ones((px, py+1), dtype=int)
    numbering_dy = -ones((px, py), dtype=int)

    # West face: edge 0 ---------------------------------
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

    # East face: edge 2 ---------------------------------
    edge_nodes = (map_[0], map_[2])
    edge_nodes_reverse = (map_[2], map_[0])
    if edge_nodes in edge_numbering_pool:
        edge_numbering = edge_numbering_pool[edge_nodes]
    elif edge_nodes_reverse in edge_numbering_pool:
        edge_numbering = edge_numbering_pool[edge_nodes_reverse][::-1]
    else:
        edge_numbering = arange(current, current + px)
        current += px
        edge_numbering_pool[edge_nodes] = edge_numbering
    numbering_dx[:, -1] = edge_numbering

    # internal dy edges ----------------------------
    numbering_dy[:-1, :] = arange(current, current + (px-1)*py).reshape((px-1, py), order='F')
    current += (px - 1) * py

    # South face : edge 1 ---------------------------------
    edge_nodes = (map_[1], map_[2])
    edge_nodes_reverse = (map_[2], map_[1])
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
