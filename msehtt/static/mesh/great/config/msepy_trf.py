# -*- coding: utf-8 -*-
r"""We config the msehtt great mesh as a msepy mesh + triangular refining.
"""
import numpy as np
from tools.frozen import Frozen
from msehtt.static.mesh.great.config.msepy_ import MseHttMsePyConfig


# noinspection PyUnusedLocal
def ___default_trf_1___(x, *args, **kwargs):
    r""""""
    return np.ones_like(x)


class MseHttMsePy_Trf_Config(Frozen):
    r""""""

    def __init__(self, tgm, domain_indicator):
        r""""""
        self._msepy_config = MseHttMsePyConfig(tgm, domain_indicator)
        self._freeze()

    def __call__(self, element_layout, trf=1, **kwargs):
        r""""""
        _, _, element_map_dict, msepy_manifold, msepy_mesh = (
            self._msepy_config(element_layout, msepy_mesh_manifold_only=False, **kwargs)
        )

        rff, rft, rcm = _parse_trf(trf)

        # ------- parse the refining -----------------------------------------------------
        if msepy_mesh.elements._element_vortices_numbering_ is None:
            msepy_mesh.elements._element_vortices_numbering_ = element_map_dict
        else:
            pass

        elements = _refining_(msepy_mesh, rff, rft, rcm)

        element_map = _make_element_map_(msepy_mesh, elements)

        Element_Type_Dict, Element_Parameter_Dict, Element_Map_Dict = _finalize_(
            msepy_mesh, element_map,
        )

        return Element_Type_Dict, Element_Parameter_Dict, Element_Map_Dict, msepy_manifold


def _parse_trf(trf):
    r"""

    Parameters
    ----------
    trf

    Returns
    -------
    rff :
        refining function
    rft :
        refining threshold
    rcm :
        refine-checking-method; which method to check whether an element needs refining.

        Can be one of:
            'center': only check the value at the element center.

    """

    if isinstance(trf, int):           # do one-layer refining for all elements.

        if trf == 0:  # no refinement -----
            rff = ___default_trf_1___      # refining function
            rft = [10, ]  # refining threshold, `10` will stop all refinement for `___default_trf_1___`.
            rcm = 'center'  # refine-checking-method; which method to check whether an element needs refining.

        else:
            assert trf > 0, r"when trf is a int, we do `trf`-level refining for all elements. So `trf` must be > 0."
            rff = ___default_trf_1___      # refining function
            rft = [0 for _ in range(trf)]  # refining threshold
            rcm = 'center'   # refine-checking-method; which method to check whether an element needs refining.

    elif isinstance(trf, dict):
        assert 'rff' in trf and 'rft' in trf, \
            (f"When provided trf as a dict, "
             f"trf must be a dict of keys 'rff' and 'rft' representing refining function "
             f"and refining threshold.")
        rff = trf['rff']
        rft = trf['rft']
        if 'rcm' in trf:
            rcm = trf['rcm']
        else:
            rcm = 'center'
        assert callable(rff), f'rff = {rff} must be callable.'
        if isinstance(rft, (int, float)):
            rft = [rft, ]
        else:
            assert isinstance(rft, (list, tuple)) and all([isinstance(_, (int, float)) for _ in rft]), \
                f"rft={rft} wrong, must be a list or tuple of all increasing numbers."
            if len(rft) > 1:
                assert all(np.diff(np.array(rft)) > 0), \
                    f"rft={rft} wrong, must be a list or tuple of all increasing numbers."

    else:
        raise NotImplementedError(f"trf={trf} is not implemented.")

    # ------ check rft -------------------------------------------------------------
    rft = np.array(rft)
    assert rft.ndim == 1 and np.all(np.diff(rft) >= 0), \
        f"refining threshold = {rft} is illegal, it must be an increasing 1d array."

    return rff, rft, rcm


def _finalize_(bmm, element_map):
    r"""

    Parameters
    ----------
    bmm
    element_map

    Returns
    -------

    """
    ndim = bmm.ndim
    Element_Type_Dict, Element_Parameter_Dict, Element_Map_Dict = {}, {}, {}

    regions = bmm._manifold.regions

    for e in element_map:
        if isinstance(e, tuple) and ndim == 2:  # triangle (vtu-5 or unique)
            msepy_element_index = e[0]
            msepy_element = bmm.elements[msepy_element_index]

            if msepy_element.metric_signature is None:  # the msepy_element is a 2d unique curvilinear triangle
                # unique curvilinear triangle
                Element_Type_Dict[e] = 'unique msepy curvilinear triangle'
                Element_Parameter_Dict[e] = {
                    'region': msepy_element._region,
                    'origin': msepy_element.ct._origin,
                    'delta': msepy_element.ct._delta,
                    'xy': _parse_ref_vortices_of_triangle_element_(e),
                }
                Element_Map_Dict[e] = element_map[e]

            elif msepy_element.metric_signature[:7] == 'Linear:':  # msepy_element is a triangle
                # vtu-5 triangle
                X, Y = _parse_ref_vortices_of_triangle_element_(e)
                X, Y = msepy_element.ct.mapping(np.array(X), np.array(Y))
                x0, x1, x2 = X
                y0, y1, y2 = Y
                Element_Type_Dict[e] = 5
                Element_Parameter_Dict[e] = (
                    (x0, y0),
                    (x1, y1),
                    (x2, y2),
                )
                Element_Map_Dict[e] = element_map[e]

            else:  # unique msepy-unique-triangle
                raise NotImplementedError(f"cannot parse element indicator={e}")

        else:  # take msepy element
            msepy_element = bmm.elements[e]
            if ndim == 2:
                if msepy_element.metric_signature is None:  # the msepy_element is a 2d unique-msepy-element
                    # unique msepy curvilinear quadrilateral
                    Element_Type_Dict[e] = 'unique msepy curvilinear quadrilateral'
                    Element_Parameter_Dict[e] = {
                        'region': msepy_element._region,
                        'origin': msepy_element.ct._origin,
                        'delta': msepy_element.ct._delta,
                    }
                    Element_Map_Dict[e] = element_map[e]

                elif msepy_element.metric_signature[:7] == 'Linear:':  # msepy_element is a 2d rectangle-msepy-element
                    # orthogonal rectangle element
                    rct = regions[msepy_element._region]._ct
                    _origin = msepy_element.ct._origin
                    _delta = msepy_element.ct._delta
                    end = (_origin[0] + _delta[0], _origin[1] + _delta[1])
                    origin = rct.mapping(*_origin)
                    end = rct.mapping(*end)
                    delta = (end[0] - origin[0], end[1] - origin[1])
                    Element_Type_Dict[e] = 'orthogonal rectangle'
                    Element_Parameter_Dict[e] = {
                        'origin': origin,
                        'delta': delta,
                    }
                    Element_Map_Dict[e] = element_map[e]

                else:
                    raise NotImplementedError(
                        f"take direct a msepy element. ndim={ndim}, "
                        f"metric_signature={msepy_element.metric_signature}."
                    )

            else:
                raise NotImplementedError(f"Not implemented for msepy mesh ndim = {ndim}")

    assert len(Element_Type_Dict) == len(Element_Parameter_Dict) == len(Element_Map_Dict), \
        f"info length dis-match."
    assert len(Element_Type_Dict) > 0, f"I at least needs one element, right?"
    return Element_Type_Dict, Element_Parameter_Dict, Element_Map_Dict


def _refining_(bmm, rff, rft, rcm):
    r"""

    Parameters
    ----------
    bmm :
        base msepy mesh
    rff :
        "refining function": A function that can be evaluated all over the domain. Its value in an
        element will be used to decide whether the element needs to be refined.
    rft :
        "refinement thresholds". The thresholds to decide how many levels to be refined.
    rcm :
        "refine-checking-method"; we use which method to check whether an element needs refining?

        It can be one of
            'center': we will only check the value of `rff` at the center of each element, if this value is
                not-lower than a threshold, refine!

    Returns
    -------
    A list of element indicators in the final mesh.

    An element indicator can be one of:

        In 2d:
            (220, 'S') means the triangle (vtu-5) attached to `South` edge of msepy element #220.
            (186, 'W', 0) means the triangle attached to edge #0 of element (186, 'W').
            (7, 'E', 2) means the triangle attached to edge #2 of element (7, 'E').

        And in 3d:
            tbc.

    """
    ndim = bmm.ndim
    elements = [e for e in bmm.elements]

    # --- do the first level refining, different from higher levels -------------------------
    elements_2be_refined = _level_one_refining_(bmm, rff, rft[0], rcm)
    elements = _parse_level_elements_(ndim, elements_2be_refined, elements, is_base=True)

    for rft_i in rft[1:]:
        elements_2be_refined = _level_high_refining_(bmm, rff, rft_i, rcm, elements)
        elements = _parse_level_elements_(ndim, elements_2be_refined, elements)

    return elements


def _make_element_map_(bmm, elements):
    r""""""
    if bmm.ndim == 2:
        return _make_2d_element_map_(bmm, elements)
    else:
        raise NotImplementedError()


def _make_2d_element_map_(bmm, elements):
    r""""""
    assert bmm.elements._element_vortices_numbering_ is not None, \
        f"pls first maka element_vortices_numbering for the msepy mesh elements."
    element_vortices = bmm.elements._element_vortices_numbering_

    MAP = {}
    for e in elements:
        if isinstance(e, tuple):  # this is a triangle element
            element_nodes = element_vortices[e[0]]
            X, Y = _parse_ref_vortices_of_triangle_element_(e)

            MAP[e] = list()
            for i in range(3):
                x, y = X[i], Y[i]
                if x == y == -1:   # this vortex is the North-West corner of msepy element
                    node_indicator = element_nodes[0]
                elif x == 1 and y == -1:  # this vortex is the South-West corner of msepy element
                    node_indicator = element_nodes[1]
                elif x == -1 and y == 1:  # this vortex is the North-East corner of msepy element
                    node_indicator = element_nodes[2]
                elif x == 1 and y == 1:  # this vortex is the South-East corner of msepy element
                    node_indicator = element_nodes[3]
                elif x == -1:  # on north edge
                    face_nodes = [element_nodes[0], element_nodes[2]]
                    face_nodes.sort()
                    node_indicator = tuple(face_nodes) + (y, )
                elif x == 1:  # on South edge
                    face_nodes = [element_nodes[1], element_nodes[3]]
                    face_nodes.sort()
                    node_indicator = tuple(face_nodes) + (y, )
                elif y == -1:  # on West edge
                    face_nodes = [element_nodes[0], element_nodes[1]]
                    face_nodes.sort()
                    node_indicator = tuple(face_nodes) + (x, )
                elif y == 1:  # on East edge
                    face_nodes = [element_nodes[2], element_nodes[3]]
                    face_nodes.sort()
                    node_indicator = tuple(face_nodes) + (x, )
                else:
                    node_indicator = (e[0], x, y)

                MAP[e].append(node_indicator)

        else:  # this must be a msepy element
            assert e in bmm.elements, f"element #{e} is not a msepy element?"
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


def _find_refined_able_pairs_(bmm, elements):
    r""""""
    if bmm.ndim == 2:
        return _find_refined_able_pairs_2d_(bmm, elements)
    else:
        raise NotImplementedError()


def _find_refined_able_pairs_2d_(bmm, elements):
    r""""""
    element_map = _make_element_map_(bmm, elements)
    msepy_map = bmm.elements.map

    elements_facing_boundary = []
    for e in element_map:
        if isinstance(e, tuple):  # triangle element (vtu-5 or unique)
            X, Y = _parse_ref_vortices_of_triangle_element_(e)
            _, x1, x2 = X
            _, y1, y2 = Y
            if x1 == x2 == -1:  # on north face
                if msepy_map[e[0]][0] == -1:
                    elements_facing_boundary.append(e)
                else:
                    pass
            elif x1 == x2 == 1:  # on south face
                if msepy_map[e[0]][1] == -1:
                    elements_facing_boundary.append(e)
                else:
                    pass
            elif y1 == y2 == -1:  # on West face
                if msepy_map[e[0]][2] == -1:
                    elements_facing_boundary.append(e)
                else:
                    pass
            elif y1 == y2 == 1:  # on East face
                if msepy_map[e[0]][3] == -1:
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


def _parse_level_elements_(ndim, elements_2be_refined, base_elements, is_base=False):
    r""""""
    new_elements = list()
    for i, ei in enumerate(base_elements):
        if ei in elements_2be_refined:
            if is_base:  # refining the base msepy elements
                if ndim == 2:
                    split_elements = [  # split into 4 elements
                        (ei, 'N'),      # i. north triangle
                        (ei, 'W'),      # iii. west triangle
                        (ei, 'E'),      # iv. east triangle
                        (ei, 'S'),      # ii. south triangle
                    ]
                else:
                    raise NotImplementedError(ndim)
            else:
                if ndim == 2:
                    split_elements = [  # split into 2 elements
                        ei + (2, ),     # ii. the triangle attached to the east edge (edge #2).
                        ei + (0, ),     # i. the triangle attached to the west edge (edge #0).
                    ]
                else:
                    raise NotImplementedError(ndim)

            new_elements.extend(split_elements)

        else:
            new_elements.append(ei)

    return new_elements


def _level_one_refining_(bmm, rff, rft0, rcm):
    r""""""
    if rcm == 'center':  # use rff value at element center to decide whether this element needs refinement.
        if bmm.ndim == 2:
            ref_center_coo = (0, 0)
        elif bmm.ndim == 3:
            ref_center_coo = (0, 0, 0)
        else:
            raise Exception()
    else:
        raise NotImplementedError(f'_level_one_refining_ not coded for rcm={rcm}')

    elements_2be_refined = list()
    elements_sofar = bmm.elements
    for e in elements_sofar:
        element = elements_sofar[e]

        if rcm == 'center':  # use rff value at element center to decide whether this element needs refinement.
            refine_checking_coo = element.ct.mapping(*ref_center_coo)
            refine_checking_val = rff(*refine_checking_coo)
        else:
            raise NotImplementedError(f'_level_one_refining_ not coded for rcm={rcm}')

        if refine_checking_val >= rft0:  # we refine this base element.
            elements_2be_refined.append(e)
        else:
            pass

    return elements_2be_refined


def _level_high_refining_(bmm, rff, rft, rcm, elements_sofar):
    r"""

    Parameters
    ----------
    bmm
    rff
    rft
    rcm
    elements_sofar :
        The refined elements so far. We will add another level on it.

    Returns
    -------

    """
    refined_able_pairs, elements_facing_boundary = _find_refined_able_pairs_(bmm, elements_sofar)

    ndim = bmm.ndim
    msepy_elements = bmm.elements
    elements_2be_refined = list()

    for e in elements_sofar:
        if e in elements_2be_refined:
            pass
        else:

            if isinstance(e, tuple):  # not a msepy base element
                if e in refined_able_pairs or e in elements_facing_boundary:
                    msepy_element = msepy_elements[e[0]]
                    if rcm == 'center':
                        if ndim == 2:
                            ref_center_coo = _parse_triangle_ref_center_coo_(e)
                        else:
                            raise NotImplementedError()
                        refine_checking_coo = msepy_element.ct.mapping(*ref_center_coo)
                        refine_checking_val = rff(*refine_checking_coo)

                    else:
                        raise NotImplementedError(f"_level_high_refining_ not implemented for rcm={rcm}.")

                    if refine_checking_val >= rft:
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


_cache_vortices_of_triangle_ = {}


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

    if position_indicators in _cache_vortices_of_triangle_:
        return _cache_vortices_of_triangle_[position_indicators]
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

    _cache_vortices_of_triangle_[position_indicators] = vortices
    return vortices


_cache_ref_center_of_triangle_ = {}


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
    if position_indicators in _cache_ref_center_of_triangle_:
        return _cache_ref_center_of_triangle_[position_indicators]
    else:
        pass
    X, Y = _parse_ref_vortices_of_triangle_element_(triangle_element_indicator)
    x0, x1, x2 = X
    y0, y1, y2 = Y
    bx, by = (x1 + x2) / 2, (y1 + y2) / 2
    dx = x0 - bx
    dy = y0 - by
    center = (bx + dx / 3, by + dy / 3)
    _cache_ref_center_of_triangle_[position_indicators] = center
    return center
