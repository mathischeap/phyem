# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from msehtt.static.mesh.great.config.msepy_trf import _parse_trf
from tools.frozen import Frozen
from msehtt.static.mesh.great.config.msehtt_ import MseHtt_Static_PreDefined_Config
from msehtt.static.mesh.great.elements.types.distributor import MseHttGreatMeshElementDistributor
from msehtt.static.mesh.great.elements.types.distributor import Vtu9Quad
from msehtt.static.mesh.great.elements.types.distributor import UniqueCurvilinearQuad


___a_cache___ = {
    'boundary_faces': None
}


class MseHtt_Static_PreDefined_Trf_Config(Frozen):
    r""""""

    def __init__(self, indicator):
        r""""""
        self._msehtt_config = MseHtt_Static_PreDefined_Config(indicator)
        self._freeze()

    def __call__(self, element_layout, **kwargs):
        r""""""
        assert 'trf' in kwargs, f"must be!"
        trf = kwargs['trf']
        else_kwargs = {}
        for key in kwargs:
            if key != 'trf':
                else_kwargs[key] = kwargs[key]
            else:
                pass

        base_mesh = self._msehtt_config(element_layout, **else_kwargs)
        # element_type_dict, element_parameter_dict, element_map_dict = base_mesh

        rff, rft, rcm = _parse_trf(trf)

        mn, elements = _refining_(base_mesh, rff, rft, rcm)

        if mn == (2, 2):
            element_map = _make_element_map_m2n2_(base_mesh, elements)
        else:
            raise NotImplementedError()

        if mn == (2, 2):
            Element_Type_Dict, Element_Parameter_Dict, Element_Map_Dict = _finalize_m2n2_(
                base_mesh, element_map,
            )
        else:
            raise NotImplementedError()

        return Element_Type_Dict, Element_Parameter_Dict, Element_Map_Dict


def _finalize_m2n2_(base_mesh, element_map):
    r"""

    Parameters
    ----------
    base_mesh
    element_map

    Returns
    -------

    """
    element_type_dict, element_parameter_dict, element_map_dict = base_mesh
    Element_Type_Dict, Element_Parameter_Dict, Element_Map_Dict = {}, {}, {}

    for e in element_map:
        if isinstance(e, tuple):  # triangle (vtu-5 or unique)
            base_element_index = e[0]
            b_etype = element_type_dict[base_element_index]
            b_para = element_parameter_dict[base_element_index]

            if b_etype == 9:  # the msehtt_base_element is a quad:9 element.
                Element_Type_Dict[e] = 5  # the four sub-elements are triangle:5-type elements
                xi, et = _parse_ref_vortices_of_triangle_element_(e)
                xi = np.array(xi)
                et = np.array(et)
                xy = Vtu9Quad._find_mapping_(b_para, xi, et)
                xy = np.array(xy).T
                Element_Parameter_Dict[e] = xy
                Element_Map_Dict[e] = element_map[e]

            elif b_etype == 'unique curvilinear quad':
                # the four sub-elements are unique curvilinear triangles.
                Element_Type_Dict[e] = 'unique curvilinear triangle'
                xi, et = _parse_ref_vortices_of_triangle_element_(e)
                triangle_ct_in_unique_quad = ___parse_TCTinUQ___(b_para, xi, et)
                parameter = {
                    'mapping': triangle_ct_in_unique_quad.mapping,
                    'Jacobian_matrix': triangle_ct_in_unique_quad.Jacobian_matrix,
                }
                Element_Parameter_Dict[e] = parameter
                Element_Map_Dict[e] = element_map[e]

            else:  # unique msepy-unique-triangle
                raise Exception(f"cannot parse element indicator={e}")

        else:  # take msehtt element
            Element_Type_Dict[e] = element_type_dict[e]
            Element_Parameter_Dict[e] = element_parameter_dict[e]
            Element_Map_Dict[e] = element_map[e]

    assert len(Element_Type_Dict) == len(Element_Parameter_Dict) == len(Element_Map_Dict), \
        f"info length dis-match."
    assert len(Element_Type_Dict) > 0, f"I at least needs one element, right?"
    return Element_Type_Dict, Element_Parameter_Dict, Element_Map_Dict


def _refining_(base_mesh, rff, rft, rcm):
    r""""""
    element_type_dict, element_parameter_dict, element_map_dict = base_mesh

    implemented_element_types = MseHttGreatMeshElementDistributor.implemented_element_types()

    M = []
    N = []

    for ele_ind in element_type_dict:

        assert not isinstance(ele_ind, tuple), \
            (f"pls do not use tuple element index for a base msehtt mesh, "
             f"as we will use tuple element index to indicate a triangle.")

        etype = element_type_dict[ele_ind]
        ele_class = implemented_element_types[etype]

        m = ele_class.m()
        n = ele_class.n()

        if m in M:
            pass
        else:
            M.append(m)

        if n in N:
            pass
        else:
            N.append(n)

    if M == [2] and M == [2]:
        for ele_ind in element_type_dict:
            etype = element_type_dict[ele_ind]
            assert etype in (9, 'unique curvilinear quad'), \
                f"Can only based on a msehtt mesh of quad or unique curvilinear quad."
        mn = (2, 2)
    else:
        raise NotImplementedError()

    # --- do the first level refining, different from higher levels -------------------------
    elements_2be_refined = _level_one_refining_(mn, base_mesh, rff, rft[0], rcm)
    new_element_indices = _parse_level_one_elements_(mn, elements_2be_refined, base_mesh)

    for rft_i in rft[1:]:
        elements_2be_refined = _level_high_refining_(mn, base_mesh, rff, rft_i, rcm, new_element_indices)
        new_element_indices = _parse_level_elements_(mn, elements_2be_refined, new_element_indices)

    ___a_cache___['boundary_faces'] = None  # clear the cache

    return mn, new_element_indices


def _level_one_refining_(mn, base_mesh, rrf, rft0, rcm):
    r""""""
    if mn == (2, 2):
        elements_2be_refined = _level_one_refining_m2n2_(base_mesh, rrf, rft0, rcm)
    else:
        raise NotImplementedError()

    return elements_2be_refined


def _level_one_refining_m2n2_(base_mesh, rrf, rft0, rcm):
    r""""""
    element_type_dict, element_parameter_dict, element_map_dict = base_mesh

    elements_2be_refined = list()

    if rcm == 'center':

        for ele_ind in element_type_dict:
            etype = element_type_dict[ele_ind]
            para = element_parameter_dict[ele_ind]
            if etype == 9:
                center = np.sum(np.array(para), axis=0) / 4

            elif etype == 'unique curvilinear quad':
                center = UniqueCurvilinearQuad._find_element_center_coo(para)

            else:
                raise Exception()

            rfv = rrf(*center)

            if rfv >= rft0:
                elements_2be_refined.append(ele_ind)
            else:
                pass

    else:
        raise NotImplementedError()

    return elements_2be_refined


def _parse_level_one_elements_(mn, elements_2be_refined, base_mesh):
    r""""""

    if mn == (2, 2):
        elements = _parse_level_one_elements_m2n2_(elements_2be_refined, base_mesh)
    else:
        raise NotImplementedError()

    return elements


def _parse_level_one_elements_m2n2_(elements_2be_refined, base_mesh):
    r""""""
    element_type_dict, element_parameter_dict, element_map_dict = base_mesh

    new_element_indices = list()

    for ele_ind in element_type_dict:
        etype = element_type_dict[ele_ind]
        if ele_ind in elements_2be_refined:
            if etype == 9:
                split_elements = [  # split into 4 elements
                    (ele_ind, 'N'),  # i. north triangle
                    (ele_ind, 'W'),  # iii. west triangle
                    (ele_ind, 'E'),  # iv. east triangle
                    (ele_ind, 'S'),  # ii. south triangle
                ]

            elif etype == 'unique curvilinear quad':

                split_elements = [  # split into 4 elements
                    (ele_ind, 'N'),  # i. north triangle
                    (ele_ind, 'W'),  # iii. west triangle
                    (ele_ind, 'E'),  # iv. east triangle
                    (ele_ind, 'S'),  # ii. south triangle
                ]

            else:
                raise Exception(f"msehtt base mesh can only of quad or unique curvilinear quad elements.")

            new_element_indices.extend(split_elements)

        else:
            new_element_indices.append(ele_ind)

    return new_element_indices


def _parse_level_elements_(mn, elements_2be_refined, old_element_indices):
    r""""""
    if mn == (2, 2):
        new_element_indices = _parse_level_elements_m2n2_(elements_2be_refined, old_element_indices)
    else:
        raise NotImplementedError()

    return new_element_indices


def _parse_level_elements_m2n2_(elements_2be_refined, old_element_indices):
    r""""""
    new_elements = list()
    for ei in old_element_indices:
        if ei in elements_2be_refined:
            split_elements = [  # split into 2 elements
                ei + (2, ),     # ii. the triangle attached to the east edge (edge #2).
                ei + (0, ),     # i. the triangle attached to the west edge (edge #0).
            ]

            new_elements.extend(split_elements)
        else:
            new_elements.append(ei)
    return new_elements


def _level_high_refining_(mn, base_mesh, rff, rft_i, rcm, element_indices):
    r"""Refine levels > 1 (the triangular refining)."""
    if mn == (2, 2):
        elements_2be_refined = _level_high_refining_m2n2_(base_mesh, rff, rft_i, rcm, element_indices)
    else:
        raise NotImplementedError()

    return elements_2be_refined


def _level_high_refining_m2n2_(base_mesh, rff, rft_i, rcm, element_indices):
    r""""""
    element_type_dict, element_parameter_dict, element_map_dict = base_mesh
    refined_able_pairs, elements_facing_boundary = _find_refined_able_pairs_m2n2_(base_mesh, element_indices)

    elements_2be_refined = list()

    for e in element_indices:
        if e in elements_2be_refined:
            pass
        else:
            if isinstance(e, tuple):  # not a msepy base element
                if e in refined_able_pairs or e in elements_facing_boundary:

                    etype = element_type_dict[e[0]]
                    e_par = element_parameter_dict[e[0]]

                    if rcm == 'center':

                        ref_center_coo = _parse_triangle_ref_center_coo_(e)

                        if etype == 9:
                            val_coo = Vtu9Quad._find_mapping_(e_par, *ref_center_coo)

                        elif etype == 'unique curvilinear quad':
                            val_coo = UniqueCurvilinearQuad._find_mapping_(e_par, *ref_center_coo)

                        else:
                            raise Exception()

                        refine_checking_val = rff(*val_coo)

                    else:
                        raise NotImplementedError(f"_level_high_refining_ not implemented for rcm={rcm}.")

                    if refine_checking_val >= rft_i:
                        if e in refined_able_pairs:
                            elements_2be_refined.append(e)
                            elements_2be_refined.append(refined_able_pairs[e])
                        elif e in elements_facing_boundary:
                            elements_2be_refined.append(e)
                        else:
                            pass
                    else:
                        pass
                else:
                    pass

            else:  # skip base msepy element
                pass

    assert len(elements_2be_refined) == len(set(elements_2be_refined)), f"repeated elements to be refined?"
    return elements_2be_refined


def _find_refined_able_pairs_m2n2_(base_mesh, element_indices):
    r""""""
    element_map = _make_element_map_m2n2_(base_mesh, element_indices)

    if ___a_cache___['boundary_faces'] is None:
        element_type_dict, _, element_map_dict = base_mesh
        boundary_faces = _parse_boundary_faces_of_a_patch_of_elements_(
            element_map_dict, element_type_dict, list(element_type_dict.keys()))

        boundary_map = {}
        for d in boundary_faces:
            ei = d['element index']
            fid = d['face id']
            if ei in boundary_map:
                boundary_map[ei].append(fid)
            else:
                boundary_map[ei] = [fid]

        # noinspection PyTypedDict
        ___a_cache___['boundary_faces'] = boundary_map

    else:
        boundary_map = ___a_cache___['boundary_faces']

    elements_facing_boundary = []
    for e in element_map:
        if isinstance(e, tuple):  # triangle element (vtu-5 or unique)
            if e[0] in boundary_map:
                X, Y = _parse_ref_vortices_of_triangle_element_(e)
                _, x1, x2 = X
                _, y1, y2 = Y

                BM = boundary_map[e[0]]

                if x1 == x2 == -1:  # on north face
                    if 0 in BM:
                        elements_facing_boundary.append(e)
                    else:
                        pass
                elif x1 == x2 == 1:  # on south face
                    if 1 in BM:
                        elements_facing_boundary.append(e)
                    else:
                        pass
                elif y1 == y2 == -1:  # on West face
                    if 2 in BM:
                        elements_facing_boundary.append(e)
                    else:
                        pass
                elif y1 == y2 == 1:  # on East face
                    if 3 in BM:
                        elements_facing_boundary.append(e)
                    else:
                        pass
                else:
                    pass
        else:  # it is not a triangle element, skip it
            pass

    bottom_edge_positions = {}
    for e in element_map:
        if isinstance(e, tuple):  # triangle element (vtu-5 or unique)
            nodes = element_map[e]
            bottom_nodes = [nodes[1], nodes[2]]
            bottom_nodes.sort()
            bottom_nodes = tuple(bottom_nodes)

            if bottom_nodes in bottom_edge_positions:
                bottom_edge_positions[bottom_nodes].append(e)
            else:
                bottom_edge_positions[bottom_nodes] = [e]
        else:  # it is not a triangle element, skip it
            pass

    refined_able_pairs = {}

    not_facing_triangle_triangles = []
    for bottom_nodes in bottom_edge_positions:
        e_e = bottom_edge_positions[bottom_nodes]
        LEN = len(e_e)
        if LEN == 1:
            assert e_e[0] not in not_facing_triangle_triangles, f'must be!'
            not_facing_triangle_triangles.append(e_e[0])
        elif LEN == 2:
            refined_able_pairs[e_e[0]] = e_e[1]
            refined_able_pairs[e_e[1]] = e_e[0]
        else:
            raise Exception

    for e in elements_facing_boundary:
        assert e in not_facing_triangle_triangles, f'must be!'

    return refined_able_pairs, elements_facing_boundary


def _make_element_map_m2n2_(base_mesh, element_indices):
    r""""""
    _, _, element_map_dict = base_mesh
    element_vortices = element_map_dict

    MAP = {}
    for e in element_indices:
        if isinstance(e, tuple):  # this is a triangle element
            element_nodes = element_vortices[e[0]]
            X, Y = _parse_ref_vortices_of_triangle_element_(e)

            MAP[e] = list()
            for i in range(3):
                x, y = X[i], Y[i]
                if x == y == -1:   # this vortex is the North-West corner of the quad element
                    node_indicator = element_nodes[0]
                elif x == 1 and y == -1:  # this vortex is the South-West corner of the quad element
                    node_indicator = element_nodes[1]
                elif x == -1 and y == 1:  # this vortex is the North-East corner of the quad element
                    node_indicator = element_nodes[3]
                elif x == 1 and y == 1:  # this vortex is the South-East corner of the quad element
                    node_indicator = element_nodes[2]
                elif x == -1:  # on north edge
                    face_nodes = [element_nodes[0], element_nodes[3]]
                    face_nodes.sort()
                    node_indicator = tuple(face_nodes) + (y, )
                elif x == 1:  # on South edge
                    face_nodes = [element_nodes[1], element_nodes[2]]
                    face_nodes.sort()
                    node_indicator = tuple(face_nodes) + (y, )
                elif y == -1:  # on West edge
                    face_nodes = [element_nodes[0], element_nodes[1]]
                    face_nodes.sort()
                    node_indicator = tuple(face_nodes) + (x, )
                elif y == 1:  # on East edge
                    face_nodes = [element_nodes[3], element_nodes[2]]
                    face_nodes.sort()
                    node_indicator = tuple(face_nodes) + (x, )
                else:
                    node_indicator = (e[0], x, y)

                MAP[e].append(node_indicator)

        else:  # this must be a msepy element
            assert e in element_map_dict, f"element #{e} is not a base element?"
            MAP[e] = element_vortices[e]

    numbering = {}
    current = 0
    for e in MAP:
        for indicator in MAP[e]:
            if indicator in numbering:
                pass
            else:
                numbering[indicator] = current
                current += 1

    element_map = {}
    for e in MAP:
        element_map[e] = list()
        for indicator in MAP[e]:
            element_map[e].append(numbering[indicator])

    return element_map


_msehtt_trf_cache_vortices_of_triangle_ = {}


def _parse_ref_vortices_of_triangle_element_(triangle_element_indicator):
    r"""

    Parameters
    ----------
    triangle_element_indicator :

    Returns
    -------
    It returns a tuple of two list:
        (
            [a, b, c],
            [d, e, f],
        ),
    where a, b, c, d, e, f are floats;
    and (a, d), (b, e), (c, f) are coordinates (in reference domain) of node #0, #1, #2 of the triangle (vtu-5) element.

    """
    position_indicators = triangle_element_indicator[1:]

    if position_indicators in _msehtt_trf_cache_vortices_of_triangle_:
        return _msehtt_trf_cache_vortices_of_triangle_[position_indicators]
    else:
        pass

    split4_indicator = position_indicators[0]
    if split4_indicator == 'N':
        vortices = (
            [0, -1, -1],
            [0, 1, -1],
        )
    elif split4_indicator == 'S':
        vortices = (
            [0, 1, 1],
            [0, -1, 1],
        )
    elif split4_indicator == 'W':
        vortices = (
            [0, -1, 1],
            [0, -1, -1],
        )
    elif split4_indicator == 'E':
        vortices = (
            [0, 1, -1],
            [0, 1, 1],
        )
    else:
        raise Exception

    for triangle_indicator in position_indicators[1:]:
        x0, x1, x2 = vortices[0]
        y0, y1, y2 = vortices[1]
        bx, by = (x1 + x2) / 2, (y1 + y2) / 2

        if triangle_indicator == 0:
            vortices = (
                [bx, x0, x1],
                [by, y0, y1],
            )
        elif triangle_indicator == 2:
            vortices = (
                [bx, x2, x0],
                [by, y2, y0],
            )
        else:
            raise Exception

    _msehtt_trf_cache_vortices_of_triangle_[position_indicators] = vortices
    return vortices


def _parse_boundary_faces_of_a_patch_of_elements_(element_map_dict, element_type_dict, element_range):
    """"""

    global_map = element_map_dict
    global_etype = element_type_dict

    # ----- find all element faces: keys are the face nodes -------------------------------------
    all_element_faces = dict()
    implemented_element_types = MseHttGreatMeshElementDistributor.implemented_element_types()
    for i in element_range:
        map_ = global_map[i]
        element_class = implemented_element_types[global_etype[i]]
        element_face_setting = element_class.face_setting()
        element_n = element_class.n()
        for face_id in element_face_setting:
            if element_n == 2:  # 2d element: only have two face nodes.
                face_start_index, face_end_index = element_face_setting[face_id]
                face_nodes = (map_[face_start_index], map_[face_end_index])
                node0 = min(face_nodes)
                node1 = max(face_nodes)
                undirected_face_indices = (node0, node1)
            elif element_n == 3:  # 3d elements
                face_node_indices = element_face_setting[face_id]
                face_nodes = [map_[_] for _ in face_node_indices]
                face_nodes.sort()
                undirected_face_indices = tuple(face_nodes)
            else:
                raise NotImplementedError()
            if undirected_face_indices in all_element_faces:
                pass
            else:
                all_element_faces[undirected_face_indices] = 0
            all_element_faces[undirected_face_indices] += 1

    # -------- find those element faces only appear once ---------------------------------------
    boundary_element_face_undirected_indices = []
    for indices in all_element_faces:
        if all_element_faces[indices] == 1:
            if len(indices) == 2:
                # for those faces only have two nodes, we add (n0, n1) and (n1, n0) to boundary face indicators.
                indices_reverse = (indices[1], indices[0])
                boundary_element_face_undirected_indices.extend([indices, indices_reverse])
            else:
                boundary_element_face_undirected_indices.append(indices)
        else:
            pass

    # ------- pick up those faces on boundary -------------------------------------------------
    boundary_faces = []
    for i in element_range:
        map_ = global_map[i]
        element_class = implemented_element_types[global_etype[i]]
        element_face_setting = element_class.face_setting()
        element_n = element_class.n()
        for face_id in element_face_setting:
            if element_n == 2:  # 2d element: only have two face nodes.
                face_start_index, face_end_index = element_face_setting[face_id]
                face_nodes = (map_[face_start_index], map_[face_end_index])
                if face_nodes in boundary_element_face_undirected_indices:
                    boundary_faces.append(
                        {
                            'element index': i,        # this face is on the element indexed ``i``.
                            'face id': face_id,        # this face is of this face id in element indexed ``i``.
                            'local node indices': element_face_setting[face_id],   # face nodes local indices
                            'global node numbering': face_nodes,                    # face node global numbering.
                        }
                    )
                else:
                    pass
            else:
                face_node_indices = element_face_setting[face_id]
                face_nodes = [map_[_] for _ in face_node_indices]
                face_nodes.sort()
                face_nodes = tuple(face_nodes)
                if face_nodes in boundary_element_face_undirected_indices:
                    boundary_faces.append(
                        {
                            'element index': i,        # this face is on the element indexed ``i``.
                            'face id': face_id,        # this face is of this face id in element indexed ``i``.
                            'local node indices': face_node_indices,   # face nodes local indices
                            'global node numbering': tuple([map_[_] for _ in face_node_indices]),
                            # face node global numbering.
                        }
                    )
                else:
                    pass

    # =========================================================================================
    return boundary_faces


_msehtt_trf_cache_ref_center_of_triangle_ = {}


def _parse_triangle_ref_center_coo_(triangle_element_indicator):
    r"""

    Parameters
    ----------
    triangle_element_indicator :
        For example,
            (220, 'S') means the triangle (vtu-5) attached to `South` edge of msepy element #220.
            (220, 'S', 0) means the triangle attached to edge #0 of element (220, 'S').

    Returns
    -------

    """
    position_indicators = triangle_element_indicator[1:]
    if position_indicators in _msehtt_trf_cache_ref_center_of_triangle_:
        return _msehtt_trf_cache_ref_center_of_triangle_[position_indicators]
    else:
        pass
    X, Y = _parse_ref_vortices_of_triangle_element_(triangle_element_indicator)
    x0, x1, x2 = X
    y0, y1, y2 = Y
    bx, by = (x1 + x2) / 2, (y1 + y2) / 2
    dx = x0 - bx
    dy = y0 - by
    center = (bx + dx / 3, by + dy / 3)
    _msehtt_trf_cache_ref_center_of_triangle_[position_indicators] = center
    return center


def ___parse_TCTinUQ___(uq_mapping_JM, vortices_xi, vortices_et):
    r""""""
    return ___TCTinUQ___(uq_mapping_JM, vortices_xi, vortices_et)


___invA___ = np.array([
    [-0.5,    0,  0.5],
    [0.25, -0.5, 0.25],
    [0.25,  0.5, 0.25]
])


class ___TCTinUQ___(Frozen):
    r""""""
    def __init__(self, uq_mapping_JM, vortices_xi, vortices_et):
        r""""""

        X = np.vstack((vortices_xi, vortices_et, [1., 1., 1.]))

        T = X @ ___invA___

        self._t00 = T[0, 0]
        self._t01 = T[0, 1]
        self._t02 = T[0, 2]

        self._t10 = T[1, 0]
        self._t11 = T[1, 1]
        self._t12 = T[1, 2]

        self._bmp_ = uq_mapping_JM
        self._freeze()

    def mapping(self, xi, et):
        r""""""
        # [-1, 1]^2 -> a reference triangle.
        r = xi
        s = et * (xi + 1) / 2
        # --- then to the physical triangle. ------------------------
        x = self._t00 * r + self._t01 * s + self._t02 * 1
        y = self._t10 * r + self._t11 * s + self._t12 * 1
        return self._bmp_['mapping'](x, y)

    def Jacobian_matrix(self, xi, et):
        r""""""

        # [-1, 1]^2 -> a reference triangle.
        r = xi
        s = et * (xi + 1) / 2
        # --- then to the physical triangle. ------------------------
        x = self._t00 * r + self._t01 * s + self._t02 * 1
        y = self._t10 * r + self._t11 * s + self._t12 * 1

        dx_dr = self._t00
        dx_ds = self._t01

        dy_dr = self._t10
        dy_ds = self._t11

        dr_dxi = 1
        ds_dxi = et / 2
        ds_det = (xi + 1) / 2

        dx_dxi = dx_dr * dr_dxi + dx_ds * ds_dxi
        dx_det = dx_ds * ds_det

        dy_dxi = dy_dr * dr_dxi + dy_ds * ds_dxi
        dy_det = dy_ds * ds_det

        dX, dY = self._bmp_['Jacobian_matrix'](x, y)
        dX_dx, dX_dy = dX
        dY_dx, dY_dy = dY

        dX_dxi = dX_dx * dx_dxi + dX_dy * dy_dxi
        dX_det = dX_dx * dx_det + dX_dy * dy_det

        dY_dxi = dY_dx * dx_dxi + dY_dy * dy_dxi
        dY_det = dY_dx * dx_det + dY_dy * dy_det

        return (
            [dX_dxi, dX_det],
            [dY_dxi, dY_det]
        )
