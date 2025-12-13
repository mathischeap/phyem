# -*- coding: utf-8 -*-
r"""
Global dof correspondence.
"""
from phyem.msehtt.static.form.main import MseHttForm
from phyem.src.spaces.main import _degree_str_maker
from phyem.src.config import RANK, MASTER_RANK, COMM


def _globalDofCorresponding__msehtt_static_form_(tgm, f0, f1):
    r"""
    CASE 0:
    This returns a dict, for example, `global_dof_cor`.

    Then `global_dof_cor[i]` is for the global dof #i of f0. And let's say, if `global_dof_cor[i] = A`.
    Then A is a list of all global dof indices of f1 that is completely on the global dof #i of f0.

    So, basically, we are trying to find the global dofs of f1 that are on each global dof of f0.

    This requires that the mesh of f0 is coarser than that of f1. And also, it is a nested refinement.

    CASE 1:
    But if f0 is f1 or f0.space is f1.space, then return 'same'

    CASE 2:
    Return None.
    """
    assert isinstance(f0, MseHttForm) and isinstance(f1, MseHttForm)
    lvl0 = tgm.find_level(f0)  # just to make sure that the form is indeed on this great multigrid mesh.
    lvl1 = tgm.find_level(f1)  # just to make sure that the form is indeed on this great multigrid mesh.

    degree0 = _degree_str_maker(f0.degree)
    degree1 = _degree_str_maker(f1.degree)

    if f0.space is f1.space and degree0 == degree1:
        return 'complete', 'same'
    else:
        pass

    method_ = tgm._configuration['method']
    if method_ == 'uniform' and lvl0 > lvl1:
        return 'empty', None
    else:
        pass

    return ___globalDofCorresponding__computer___(f0, f1)


def ___globalDofCorresponding__computer___(f0, f1):
    r""""""
    # -------- FIND RANK GEOMETRIES s0 ---------------------------------------------------------
    s0 = f0.space
    gm0 = f0.cochain.gathering_matrix
    dof_geometries_0 = {}
    for e in gm0:
        local_dofs = gm0[e]
        for i, global_dof in enumerate(local_dofs):
            geo = s0.find_dof_geometry.local_dof(f0.degree, e, i)
            if global_dof in dof_geometries_0:
                if ___geo_in_geometries___(geo, dof_geometries_0[global_dof]):
                    pass
                else:
                    dof_geometries_0[global_dof].append(geo)
            else:
                dof_geometries_0[global_dof] = [geo, ]

    # -------- FIND RANK GEOMETRIES s1 ---------------------------------------------------------
    s1 = f1.space
    gm1 = f1.cochain.gathering_matrix
    dof_geometries_1 = {}
    for e in gm1:
        local_dofs = gm1[e]
        for i, global_dof in enumerate(local_dofs):
            geo = s1.find_dof_geometry.local_dof(f1.degree, e, i)
            if global_dof in dof_geometries_1:
                if ___geo_in_geometries___(geo, dof_geometries_1[global_dof]):
                    pass
                else:
                    dof_geometries_1[global_dof].append(geo)
            else:
                dof_geometries_1[global_dof] = [geo, ]

    # -------- gather geometries s0 ---------------------------------------------------------
    DOF_geometries_0 = COMM.gather(dof_geometries_0, root=MASTER_RANK)
    if RANK == MASTER_RANK:
        DOF_GEO_0 = dict()
        for DICT in DOF_geometries_0:
            for dof in DICT:
                if dof in DOF_GEO_0:
                    new_geo_S = DICT[dof]
                    to_be_extended = []
                    for ng in new_geo_S:
                        if ___geo_in_geometries___(ng, DOF_GEO_0[dof]):
                            pass
                        else:
                            to_be_extended.append(ng)
                    DOF_GEO_0[dof].extend(to_be_extended)
                else:
                    DOF_GEO_0[dof] = DICT[dof]
    else:  # slave ranks
        DOF_GEO_0 = []

    del DOF_geometries_0

    # -------- gather geometries s1 ---------------------------------------------------------
    DOF_geometries_1 = COMM.gather(dof_geometries_1, root=MASTER_RANK)
    if RANK == MASTER_RANK:
        DOF_GEO_1 = dict()
        DOF_GEO_1_COPY = dict()
        for DICT in DOF_geometries_1:
            for dof in DICT:
                if dof in DOF_GEO_1:
                    new_geo_S = DICT[dof]
                    to_be_extended = []
                    for ng in new_geo_S:
                        if ___geo_in_geometries___(ng, DOF_GEO_1[dof]):
                            pass
                        else:
                            to_be_extended.append(ng)
                    DOF_GEO_1[dof].extend(to_be_extended)
                    DOF_GEO_1_COPY[dof].extend(to_be_extended)
                else:
                    DOF_GEO_1[dof] = DICT[dof]
                    DOF_GEO_1_COPY[dof] = DICT[dof]
    else:  # slave ranks
        DOF_GEO_1 = []
        DOF_GEO_1_COPY = []

    del DOF_geometries_1

    # gather dof groups -------------------------------------------------------------------
    if RANK == MASTER_RANK:
        group = {}   # dofs in a group could be subset of each other

        for dof1 in DOF_GEO_1:
            geometries1 = DOF_GEO_1[dof1]
            geo1 = geometries1[0]
            if hasattr(geo1, '___possible_subset_group_key___'):
                key = geo1.___possible_subset_group_key___()

                if key in group:
                    assert dof1 not in group[key]
                    group[key].append(dof1)
                else:
                    group[key] = [dof1, ]
            else:
                # to make sure that if one geometry has it, all geometries should have that.
                assert len(group) == 0, \
                    f"{geo1} of type {geo1.__class__} misses property ___possible_subset_group_key___."

        if len(group) == 0:
            group = None
        else:
            pass
    else:
        group = {}  # do nothing, just to avoid pycharm warning

    # -------- FIND correspondence --------------------------------------------------------
    if RANK == MASTER_RANK:

        if group is None:
            GLOBAL_SEARCH = True
        else:
            GLOBAL_SEARCH = False

        globalDofCorresponding = dict()
        possible_complete = True
        for dof0 in DOF_GEO_0:
            dof0 = int(dof0)
            geometries0 = DOF_GEO_0[dof0]
            found_dofs = []

            if GLOBAL_SEARCH:
                SEARCH_RANGE = DOF_GEO_1
            else:
                geo0 = geometries0[0]
                key = geo0.___possible_subset_group_key___()
                if key not in group:
                    SEARCH_RANGE = []
                else:
                    SEARCH_RANGE = group[key]

            for dof1 in SEARCH_RANGE:
                if dof1 in DOF_GEO_1:
                    geometries1 = DOF_GEO_1[dof1]
                    if ___there_is_a_geo_in_geometries1_that_is_a_part_of_one_in_geometries0___(
                            geometries0, geometries1):
                        found_dofs.append(int(dof1))
                    else:
                        pass
                else:
                    pass

            if len(found_dofs) == 0:
                if possible_complete:
                    possible_complete = False
                else:
                    pass
            else:
                globalDofCorresponding[dof0] = found_dofs
                for found_dof1 in found_dofs:
                    del DOF_GEO_1[found_dof1]

        # ~~~~ check complete ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 'complete' mean each dof of f0 is fully covered by dofs of f1
        if possible_complete:   # we should do this check when there is still possibility
            complete = True
            for dof0 in globalDofCorresponding:
                geometries0 = DOF_GEO_0[dof0]
                dofs1 = globalDofCorresponding[dof0]
                GEOMETRIES1 = []
                for dof1 in dofs1:
                    ___ = DOF_GEO_1_COPY[dof1]
                    for _g_ in ___:
                        if ___geo_in_geometries___(_g_, GEOMETRIES1):
                            pass
                        else:
                            GEOMETRIES1.append(_g_)

                geo0 = geometries0[0]
                # we can just use one geometry of dof0 there may be multiple for periodic domain.
                if geo0.is_equal_to_a_union_of_a_part_of_separate_geometries_of_same_type(GEOMETRIES1):
                    pass
                else:
                    complete = False
                    break
        else:
            complete = False

    else:  # slave ranks
        complete = None
        globalDofCorresponding = None

    # -------- distribute to ranks ------------------------------------------------------------
    complete = COMM.bcast(complete, root=MASTER_RANK)
    globalDofCorresponding = COMM.bcast(globalDofCorresponding, root=MASTER_RANK)

    rank_globalDofCorresponding = {}
    for dof0 in globalDofCorresponding:
        if dof0 in dof_geometries_0:
            DOFs1 = globalDofCorresponding[dof0]
            for dof1 in DOFs1:
                assert dof1 in dof_geometries_1, f"must be, dof at the same place must be in same rank"
            rank_globalDofCorresponding[dof0] = globalDofCorresponding[dof0]

    if complete:
        complete_or_incomplete_ind = 'complete'
    else:
        complete_or_incomplete_ind = 'incomplete'

    # print(complete_or_incomplete_ind)

    return complete_or_incomplete_ind, rank_globalDofCorresponding


def ___geo_in_geometries___(geo, geometries):
    r""""""
    for GEO in geometries:
        if geo == GEO:
            return True
        else:
            pass
    return False


def ___there_is_a_geo_in_geometries1_that_is_a_part_of_one_in_geometries0___(geometries0, geometries1):
    r""""""
    for geo in geometries1:
        if ___geo_is_a_part_of_one_in_geometries___(geo, geometries0):
            return True
        else:
            pass
    return False


def ___geo_is_a_part_of_one_in_geometries___(geo, geometries):
    r"""check if there is a geometry in `geometries0` that completely contain `geo`. If yes, return True. Otherwise,
    return False.

    Remember, if two geometries are different types, they cannot be a part of each other. For example, a point
    on a face is not regarded as a part of the face. An edge on a face is not neither. A point on an edge is not
    neither.
    """
    for GEO in geometries:
        if GEO.whether_contain__input__as_a_part_of_same_type(geo):
            return True
        else:
            pass
    return False
