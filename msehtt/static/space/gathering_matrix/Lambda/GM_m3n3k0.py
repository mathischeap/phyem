# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.src.config import RANK, MASTER_RANK, COMM, SIZE
from phyem.msehtt.tools.gathering_matrix import MseHttGatheringMatrix
from phyem.src.spaces.main import _degree_str_maker

_cache_ = {}


def gathering_matrix_Lambda__m3n3k0(tpm, degree):
    """Do the numbering for the 0-form on a 3d mesh in 3d space."""
    key = tpm.__repr__() + _degree_str_maker(degree)
    if key in _cache_:
        return _cache_[key]

    tgm = tpm._tgm
    # do the numbering in the master rank only
    if RANK == MASTER_RANK:
        global_map = tgm._global_element_map_dict
        global_type = tgm._global_element_type_dict
        element_distribution = tgm._element_distribution

        element_face_topology_mismatch = tgm.elements._element_face_topology_mismatch

        global_numbering = {}

        face_numbering_pool = {}
        edge_numbering_pool = {}
        node_numbering_pool = {}
        current = 0
        for e in global_map:
            etype = global_type[e]
            map_ = global_map[e]
            # --------- call the element class to do the particular numbering -----------
            if e in tpm.composition.global_element_range:
                if etype in (
                        'orthogonal hexahedron',
                        "unique msepy curvilinear hexahedron",
                ):
                    global_numbering[e], current = ___gm330_msepy_quadrilateral___(
                        map_,
                        face_numbering_pool, edge_numbering_pool, node_numbering_pool,
                        current, degree,
                        element_face_topology_mismatch
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


def ___gm330_msepy_quadrilateral___(
        map_,
        face_numbering_pool, edge_numbering_pool, node_numbering_pool,
        current, degree,
        element_face_topology_mismatch
):
    """"""

    p, _ = MseHttGreatMeshOrthogonalHexahedronElement.degree_parser(degree)
    px, py, pz = p

    numbering = - np.ones((px+1, py+1, pz+1), dtype=int)

    if element_face_topology_mismatch:
        raise NotImplementedError()

    else:
        pass

    # ----------- element_face_topology_mismatch is False; perfect element topology! --------------

    # ----------- CORNERS: 8 ------------------------------------------------------------------------

    # (x-, y-, z-) corner #0 ---------------------------------
    corner = map_[0]
    if corner in node_numbering_pool:
        number = node_numbering_pool[corner]
    else:
        number = current
        current += 1
        node_numbering_pool[corner] = number
    numbering[0, 0, 0] = number

    # (x+, y-, z-) corner #01 ---------------------------------
    corner = map_[1]
    if corner in node_numbering_pool:
        number = node_numbering_pool[corner]
    else:
        number = current
        current += 1
        node_numbering_pool[corner] = number
    numbering[-1, 0, 0] = number

    # (x-, y+, z-) corner #2 ---------------------------------
    corner = map_[2]
    if corner in node_numbering_pool:
        number = node_numbering_pool[corner]
    else:
        number = current
        current += 1
        node_numbering_pool[corner] = number
    numbering[0, -1, 0] = number

    # (x+, y+, z-) corner #3 ---------------------------------
    corner = map_[3]
    if corner in node_numbering_pool:
        number = node_numbering_pool[corner]
    else:
        number = current
        current += 1
        node_numbering_pool[corner] = number
    numbering[-1, -1, 0] = number

    # (x-, y-, z+) corner #4 ---------------------------------
    corner = map_[4]
    if corner in node_numbering_pool:
        number = node_numbering_pool[corner]
    else:
        number = current
        current += 1
        node_numbering_pool[corner] = number
    numbering[0, 0, -1] = number

    # (x+, y-, z+) corner #5 ---------------------------------
    corner = map_[5]
    if corner in node_numbering_pool:
        number = node_numbering_pool[corner]
    else:
        number = current
        current += 1
        node_numbering_pool[corner] = number
    numbering[-1, 0, -1] = number

    # (x-, y+, z+) corner #6 ---------------------------------
    corner = map_[6]
    if corner in node_numbering_pool:
        number = node_numbering_pool[corner]
    else:
        number = current
        current += 1
        node_numbering_pool[corner] = number
    numbering[0, -1, -1] = number

    # (x+, y+, z+) corner #7 ---------------------------------
    corner = map_[7]
    if corner in node_numbering_pool:
        number = node_numbering_pool[corner]
    else:
        number = current
        current += 1
        node_numbering_pool[corner] = number
    numbering[-1, -1, -1] = number

    # ----------- EDGES: 12 ------------------------------------------------------------------------

    # dx #0: (y-, z-) edge -----------------------
    edge = (map_[0], map_[1])
    if edge in edge_numbering_pool:
        number = edge_numbering_pool[edge]
    else:
        number = np.arange(current, current + px - 1)
        current += px - 1
        edge_numbering_pool[edge] = number
    numbering[1:-1, 0, 0] = number

    # dx #1: (y+, z-) edge -----------------------
    edge = (map_[2], map_[3])
    if edge in edge_numbering_pool:
        number = edge_numbering_pool[edge]
    else:
        number = np.arange(current, current + px - 1)
        current += px - 1
        edge_numbering_pool[edge] = number
    numbering[1:-1, -1, 0] = number

    # dx #2: (y-, z+) edge -----------------------
    edge = (map_[4], map_[5])
    if edge in edge_numbering_pool:
        number = edge_numbering_pool[edge]
    else:
        number = np.arange(current, current + px - 1)
        current += px - 1
        edge_numbering_pool[edge] = number
    numbering[1:-1, 0, -1] = number

    # dx #3: (y+, z+) edge -----------------------
    edge = (map_[6], map_[7])
    if edge in edge_numbering_pool:
        number = edge_numbering_pool[edge]
    else:
        number = np.arange(current, current + px - 1)
        current += px - 1
        edge_numbering_pool[edge] = number
    numbering[1:-1, -1, -1] = number

    # dy #0: (x-, z-) edge -----------------------
    edge = (map_[0], map_[2])
    if edge in edge_numbering_pool:
        number = edge_numbering_pool[edge]
    else:
        number = np.arange(current, current + py - 1)
        current += py - 1
        edge_numbering_pool[edge] = number
    numbering[0, 1:-1, 0] = number

    # dy #1: (x+, z-) edge -----------------------
    edge = (map_[1], map_[3])
    if edge in edge_numbering_pool:
        number = edge_numbering_pool[edge]
    else:
        number = np.arange(current, current + py - 1)
        current += py - 1
        edge_numbering_pool[edge] = number
    numbering[-1, 1:-1, 0] = number

    # dy #2: (x-, z+) edge -----------------------
    edge = (map_[4], map_[6])
    if edge in edge_numbering_pool:
        number = edge_numbering_pool[edge]
    else:
        number = np.arange(current, current + py - 1)
        current += py - 1
        edge_numbering_pool[edge] = number
    numbering[0, 1:-1, -1] = number

    # dy #3: (x+, z+) edge -----------------------
    edge = (map_[5], map_[7])
    if edge in edge_numbering_pool:
        number = edge_numbering_pool[edge]
    else:
        number = np.arange(current, current + py - 1)
        current += py - 1
        edge_numbering_pool[edge] = number
    numbering[-1, 1:-1, -1] = number

    # dz #0: (x-, y-) edge -----------------------
    edge = (map_[0], map_[4])
    if edge in edge_numbering_pool:
        number = edge_numbering_pool[edge]
    else:
        number = np.arange(current, current + pz - 1)
        current += pz - 1
        edge_numbering_pool[edge] = number
    numbering[0, 0, 1:-1] = number

    # dz #1: (x+, y-) edge -----------------------
    edge = (map_[1], map_[5])
    if edge in edge_numbering_pool:
        number = edge_numbering_pool[edge]
    else:
        number = np.arange(current, current + pz - 1)
        current += pz - 1
        edge_numbering_pool[edge] = number
    numbering[-1, 0, 1:-1] = number

    # dz #2: (x-, y+) edge -----------------------
    edge = (map_[2], map_[6])
    if edge in edge_numbering_pool:
        number = edge_numbering_pool[edge]
    else:
        number = np.arange(current, current + pz - 1)
        current += pz - 1
        edge_numbering_pool[edge] = number
    numbering[0, -1, 1:-1] = number

    # dz #3: (x+, y+) edge -----------------------
    edge = (map_[3], map_[7])
    if edge in edge_numbering_pool:
        number = edge_numbering_pool[edge]
    else:
        number = np.arange(current, current + pz - 1)
        current += pz - 1
        edge_numbering_pool[edge] = number
    numbering[-1, -1, 1:-1] = number

    # ----------- FACES: 6 ------------------------------------------------------------------------

    # dy^dz face #0 (x-) ---------------------------------------------------------------
    face = (map_[0], map_[2], map_[4], map_[6])
    if face in face_numbering_pool:
        number = face_numbering_pool[face]
    else:
        number = np.arange(current, current + (py - 1) * (pz - 1)).reshape(
            ((py - 1), (pz - 1)), order='F')
        current += (py - 1) * (pz - 1)
        face_numbering_pool[face] = number
    numbering[0, 1:-1, 1:-1] = number

    # dy^dz face #1 (x+) ---------------------------------------------------------------
    face = (map_[1], map_[3], map_[5], map_[7])
    if face in face_numbering_pool:
        number = face_numbering_pool[face]
    else:
        number = np.arange(current, current + (py - 1) * (pz - 1)).reshape(
            ((py - 1), (pz - 1)), order='F')
        current += (py - 1) * (pz - 1)
        face_numbering_pool[face] = number
    numbering[-1, 1:-1, 1:-1] = number

    # dz^dx face #2 (y-) ---------------------------------------------------------------
    face = (map_[0], map_[1], map_[4], map_[5])
    if face in face_numbering_pool:
        number = face_numbering_pool[face]
    else:
        number = np.arange(current, current + (pz - 1) * (px - 1)).reshape(
            ((px - 1), (pz - 1)), order='F')
        current += (pz - 1) * (px - 1)
        face_numbering_pool[face] = number
    numbering[1:-1, 0, 1:-1] = number

    # dz^dx face #3 (y+) ---------------------------------------------------------------
    face = (map_[2], map_[3], map_[6], map_[7])
    if face in face_numbering_pool:
        number = face_numbering_pool[face]
    else:
        number = np.arange(current, current + (pz - 1) * (px - 1)).reshape(
            ((px - 1), (pz - 1)), order='F')
        current += (pz - 1) * (px - 1)
        face_numbering_pool[face] = number
    numbering[1:-1, -1, 1:-1] = number

    # dx^dy face #4 (z-) ---------------------------------------------------------------
    face = (map_[0], map_[1], map_[2], map_[3])
    if face in face_numbering_pool:
        number = face_numbering_pool[face]
    else:
        number = np.arange(current, current + (px - 1) * (py - 1)).reshape(
            ((px - 1), (py - 1)), order='F')
        current += (px - 1) * (py - 1)
        face_numbering_pool[face] = number
    numbering[1:-1, 1:-1, 0] = number

    # dx^dy face #5 (z+) ---------------------------------------------------------------
    face = (map_[4], map_[5], map_[6], map_[7])
    if face in face_numbering_pool:
        number = face_numbering_pool[face]
    else:
        number = np.arange(current, current + (px - 1) * (py - 1)).reshape(
            ((px - 1), (py - 1)), order='F')
        current += (px - 1) * (py - 1)
        face_numbering_pool[face] = number
    numbering[1:-1, 1:-1, -1] = number

    # ------ INTERNAL nodes ------------------------------------------------------------------------------
    numbering[1:-1, 1:-1, 1:-1] = np.arange(current, current + (px - 1) * (py - 1) * (pz - 1)).reshape(
        ((px - 1), (py - 1), (pz - 1)), order='F'
    )
    current += (px - 1) * (py - 1) * (pz - 1)

    # ====================================================================================================
    numbering = numbering.ravel('F')
    assert -1 not in numbering, f"at least one dof is not numbered."
    return numbering, current
